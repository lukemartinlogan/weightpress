"""Input sources: safetensors checkpoints, .npy arrays, and raw binary.

Every source is presented as one flat float32 stream plus a manifest recording
where each original tensor lives in it, so decompression can rebuild the
checkpoint with its original dtypes and shapes.

Widening fp16/bf16 to fp32 is exact, so the error bound is measured in fp32
space.  Compression ratios are always reported against the *source* byte count
(``source_bytes``), never the widened one -- reporting against fp32 would make a
fp16 checkpoint look twice as compressible as it is.
"""

from __future__ import annotations

import dataclasses
import json
import os
import struct
from typing import Any, Iterator

import numpy as np

#: safetensors dtype tag -> (numpy view dtype, bytes per element).  BF16 has no
#: numpy equivalent, so it is read as uint16 and widened by hand.
_ST_DTYPES: dict[str, tuple[np.dtype, int]] = {
    "F64": (np.dtype("<f8"), 8),
    "F32": (np.dtype("<f4"), 4),
    "F16": (np.dtype("<f2"), 2),
    "BF16": (np.dtype("<u2"), 2),
    "I64": (np.dtype("<i8"), 8),
    "I32": (np.dtype("<i4"), 4),
    "I16": (np.dtype("<i2"), 2),
    "I8": (np.dtype("i1"), 1),
    "U8": (np.dtype("u1"), 1),
    "BOOL": (np.dtype("?"), 1),
}


def read_safetensors_header(path: str) -> tuple[dict[str, Any], int]:
    """Parse the JSON header and return it with the data-section offset.

    ``safetensors.safe_open`` is avoided deliberately: it costs tens of seconds
    on a multi-hundred-MB checkpoint here, while the header is a few hundred
    microseconds to parse directly.
    """
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def _widen(raw: np.ndarray, tag: str) -> np.ndarray:
    """Convert a raw tensor view to float32."""
    if tag == "BF16":
        # bfloat16 is the top 16 bits of a float32, so this is exact.
        return (raw.astype(np.uint32) << 16).view(np.float32)
    return raw.astype(np.float32)


@dataclasses.dataclass
class TensorEntry:
    name: str
    dtype: str
    shape: list[int]
    offset: int  # in values, into the flat fp32 stream
    numel: int
    itemsize: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class WeightStream:
    """A flat float32 view over a checkpoint, sliced into fixed-size windows."""

    def __init__(self, path: str, limit_values: int | None = None):
        self.path = path
        self.limit_values = limit_values
        self.entries: list[TensorEntry] = []
        self.total_values = 0
        self.source_bytes = 0

    # -- construction ---------------------------------------------------
    @staticmethod
    def open(path: str, *, dtype: str = "float32", limit_bytes: int | None = None):
        ext = os.path.splitext(path)[1].lower()
        limit_values = None
        if ext == ".safetensors":
            src = _SafeTensorsStream(path)
        elif ext == ".npy":
            src = _NpyStream(path)
        else:
            src = _RawStream(path, dtype)
        if limit_bytes is not None:
            limit_values = max(1, limit_bytes // np.dtype(np.float32).itemsize)
        src.limit_values = limit_values
        src._build()
        return src

    def _build(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _iter_tensors(self) -> Iterator[np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    # -- windowing ------------------------------------------------------
    def windows(self, values_per_window: int) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(index, fp32_array)`` windows of the concatenated stream."""
        buf: list[np.ndarray] = []
        held = 0
        index = 0
        produced = 0
        cap = self.limit_values if self.limit_values is not None else None

        for arr in self._iter_tensors():
            if cap is not None and produced + held + arr.size > cap:
                arr = arr[: max(0, cap - produced - held)]
            if arr.size:
                buf.append(arr)
                held += arr.size
            while held >= values_per_window:
                flat = np.concatenate(buf) if len(buf) > 1 else buf[0]
                yield index, np.ascontiguousarray(flat[:values_per_window])
                index += 1
                produced += values_per_window
                rest = flat[values_per_window:]
                buf = [rest] if rest.size else []
                held = rest.size
            if cap is not None and produced + held >= cap:
                break

        if held:
            flat = np.concatenate(buf) if len(buf) > 1 else buf[0]
            yield index, np.ascontiguousarray(flat)

    def manifest(self) -> dict[str, Any]:
        return {
            "path": os.path.abspath(self.path),
            "total_values": self.total_values,
            "source_bytes": self.source_bytes,
            "tensors": [e.to_dict() for e in self.entries],
        }


class _SafeTensorsStream(WeightStream):
    """Reads a checkpoint by memory-mapping it and slicing at header offsets."""

    def _build(self) -> None:
        header, data_start = read_safetensors_header(self.path)
        self._data_start = data_start
        # Sort by file offset: that makes the stream a single forward scan of
        # the file rather than a scatter of seeks.
        items = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
        self._byte_ranges: list[tuple[int, int, str]] = []
        offset = 0
        for name, info in items:
            tag = info["dtype"]
            if tag not in _ST_DTYPES:
                raise ValueError(f"{self.path}: unsupported tensor dtype {tag!r}")
            _, itemsize = _ST_DTYPES[tag]
            shape = list(info["shape"])
            start, end = info["data_offsets"]
            numel = (end - start) // itemsize
            self.entries.append(
                TensorEntry(name, tag, shape, offset, numel, itemsize)
            )
            self._byte_ranges.append((data_start + start, data_start + end, tag))
            offset += numel
            self.source_bytes += end - start
        self.total_values = offset
        if self.limit_values is not None:
            self.total_values = min(self.total_values, self.limit_values)

    def _iter_tensors(self) -> Iterator[np.ndarray]:
        mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        for (start, end, tag) in self._byte_ranges:
            dt, _ = _ST_DTYPES[tag]
            # Copy the byte range before viewing: a tensor may begin at an
            # offset that is not aligned for its element type.
            raw = np.frombuffer(bytes(mm[start:end]), dtype=dt)
            yield _widen(raw, tag)


class _NpyStream(WeightStream):
    def _build(self) -> None:
        arr = np.load(self.path, mmap_mode="r")
        self.entries = [
            TensorEntry(
                os.path.basename(self.path),
                str(arr.dtype),
                list(arr.shape),
                0,
                int(arr.size),
                arr.dtype.itemsize,
            )
        ]
        self.total_values = int(arr.size)
        self.source_bytes = int(arr.size * arr.dtype.itemsize)
        if self.limit_values is not None:
            self.total_values = min(self.total_values, self.limit_values)

    def _iter_tensors(self) -> Iterator[np.ndarray]:
        arr = np.load(self.path, mmap_mode="r")
        step = 1 << 24
        flat = arr.reshape(-1)
        for lo in range(0, flat.size, step):
            yield np.asarray(flat[lo : lo + step], dtype=np.float32)


class _RawStream(WeightStream):
    def __init__(self, path: str, dtype: str):
        super().__init__(path)
        self.dtype = np.dtype(dtype)

    def _build(self) -> None:
        nbytes = os.path.getsize(self.path)
        n = nbytes // self.dtype.itemsize
        self.entries = [
            TensorEntry(
                os.path.basename(self.path), str(self.dtype), [n], 0, n,
                self.dtype.itemsize,
            )
        ]
        self.total_values = n
        self.source_bytes = n * self.dtype.itemsize
        if self.limit_values is not None:
            self.total_values = min(self.total_values, self.limit_values)

    def _iter_tensors(self) -> Iterator[np.ndarray]:
        step = 1 << 24
        arr = np.memmap(self.path, dtype=self.dtype, mode="r")
        for lo in range(0, arr.size, step):
            yield np.asarray(arr[lo : lo + step], dtype=np.float32)


def restore_tensors(
    values: np.ndarray, manifest: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Slice a reconstructed fp32 stream back into named tensors."""
    out: dict[str, np.ndarray] = {}
    for e in manifest["tensors"]:
        lo, n = e["offset"], e["numel"]
        if lo >= values.size:
            break
        piece = values[lo : lo + n]
        if piece.size < n:
            break
        out[e["name"]] = piece.reshape(e["shape"])
    return out
