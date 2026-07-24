"""On-disk container for a compressed weight stream.

Layout::

    MAGIC | version:u32 | reserved:u32 | <chunk payloads ...> | header_json | header_len:u64 | MAGIC

The header is written last (payload sizes are not known up front) and located by
seeking to the end, so writing stays a single forward pass.

Each chunk payload is the concatenation of its centroid table and the zstd blobs
in a fixed order; the header records every length.  The centroid tables are
*also* written as standalone ``.npz`` sidecars under ``<out>/<name>.kmeans/`` --
the container stays self-describing, and the k-means tables are independently
inspectable per 128 MB chunk as the design calls for.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any, BinaryIO

import numpy as np

from .codec import EncodedChunk
from .config import FORMAT_VERSION, MAGIC

_BLOBS = (
    "labels_blob",
    "code_lo_blob",
    "code_hi_blob",
    "outlier_idx_blob",
    "outlier_val_blob",
)


class ContainerWriter:
    """Streams chunks out in index order."""

    def __init__(self, path: str, meta: dict[str, Any]):
        self.path = path
        self.meta = meta
        self._fh: BinaryIO | None = None
        self._chunks: list[dict[str, Any]] = []

    def __enter__(self) -> ContainerWriter:
        self._fh = open(self.path, "wb")
        self._fh.write(MAGIC)
        self._fh.write(struct.pack("<II", FORMAT_VERSION, 0))
        return self

    def add(self, chunk: EncodedChunk) -> None:
        assert self._fh is not None
        offset = self._fh.tell()
        cent = np.ascontiguousarray(chunk.centroids, dtype=np.float32)
        self._fh.write(cent.tobytes())
        entry = {
            "index": chunk.index,
            "offset": offset,
            "k": chunk.k,
            "tuple_size": chunk.tuple_size,
            "n_values": chunk.n_values,
            "mode": chunk.mode,
            "labels_itemsize": chunk.labels_itemsize,
            "n_outliers": chunk.n_outliers,
            "centroid_bytes": int(cent.nbytes),
        }
        for name in _BLOBS:
            blob = getattr(chunk, name)
            entry[name + "_len"] = len(blob)
            self._fh.write(blob)
        self._chunks.append(entry)

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._fh is not None
        if exc_type is None:
            header = json.dumps({**self.meta, "chunks": self._chunks}).encode()
            self._fh.write(header)
            self._fh.write(struct.pack("<Q", len(header)))
            self._fh.write(MAGIC)
        self._fh.close()
        self._fh = None


class ContainerReader:
    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "rb")
        if self._fh.read(4) != MAGIC:
            raise ValueError(f"{path}: not a weightpress container")
        version, _ = struct.unpack("<II", self._fh.read(8))
        if version != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported format version {version}")
        self._fh.seek(-12, os.SEEK_END)
        (header_len,) = struct.unpack("<Q", self._fh.read(8))
        if self._fh.read(4) != MAGIC:
            raise ValueError(f"{path}: truncated container (bad trailer)")
        self._fh.seek(-12 - header_len, os.SEEK_END)
        self.header: dict[str, Any] = json.loads(self._fh.read(header_len))
        self.chunks: list[dict[str, Any]] = self.header["chunks"]

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> ContainerReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_chunk(self, i: int) -> EncodedChunk:
        e = self.chunks[i]
        self._fh.seek(e["offset"])
        cent = np.frombuffer(
            self._fh.read(e["centroid_bytes"]), dtype=np.float32
        ).reshape(e["k"], e["tuple_size"])
        blobs = {name: self._fh.read(e[name + "_len"]) for name in _BLOBS}
        return EncodedChunk(
            index=e["index"],
            k=e["k"],
            tuple_size=e["tuple_size"],
            n_values=e["n_values"],
            mode=e["mode"],
            centroids=cent,
            labels_itemsize=e["labels_itemsize"],
            n_outliers=e["n_outliers"],
            **blobs,
        )


def write_codebook_sidecar(
    out_dir: str, stem: str, index: int, centroids: np.ndarray, meta: dict[str, Any]
) -> str:
    """Write one chunk's k-means table as a standalone ``.npz``."""
    d = os.path.join(out_dir, f"{stem}.kmeans")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"chunk_{index:06d}.npz")
    np.savez(path, centroids=np.asarray(centroids, dtype=np.float32), meta=json.dumps(meta))
    return path
