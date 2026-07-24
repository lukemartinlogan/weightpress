"""Error-bounded residual quantization and the lossless integer back end.

The k-means codebook supplies a prediction ``p`` for every weight.  The residual
is quantized onto a grid of width ``2 * error_bound``::

    q     = round((x - p) / (2 * eb))
    x_hat = p + q * (2 * eb)          =>   |x - x_hat| <= eb

so the bound holds by construction for every value, whatever k turns out to be.
Code ``0`` is reserved as an escape: residuals too large for the 16-bit code
word are stored verbatim as float32 and reconstruct exactly.

The integer stream is zigzagged (so near-zero residuals become near-zero
unsigned words), split into low and high byte planes, and handed to zstd.  The
split matters: the high plane of a Gaussian residual is almost all zeros and
collapses, while mixing it with the noisy low plane would defeat the entropy
coder.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import zstandard as zstd

from .config import CODE_MAX

__all__ = [
    "EncodedChunk",
    "dequantize",
    "labels_dtype",
    "pack_chunk",
    "quant_step",
    "quantize",
    "reconstruct",
    "unpack_chunk",
    "zigzag_decode",
    "zigzag_encode",
]

#: The grid is made 0.01% finer than the nominal ``2 * error_bound`` so that the
#: float32 rounding of ``pred + q * step`` cannot push a value that is exactly on
#: the bound just past it.  Costs ~0.0001 bits/value and buys headroom of
#: ``1e-4 * eb``, three orders of magnitude above the float32 ulp of typical
#: weights.  The encoder still verifies every value and escapes any that misses.
QUANT_MARGIN = 1e-4


def quant_step(error_bound: float) -> float:
    return 2.0 * error_bound * (1.0 - QUANT_MARGIN)


def labels_dtype(k: int) -> np.dtype:
    if k <= 1 << 8:
        return np.dtype(np.uint8)
    if k <= 1 << 16:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def zigzag_encode(code: np.ndarray) -> np.ndarray:
    """Signed int32 -> uint16, small magnitudes to small words."""
    c = code.astype(np.int32, copy=False)
    return ((c << 1) ^ (c >> 31)).astype(np.uint16)


def zigzag_decode(z: np.ndarray) -> np.ndarray:
    u = z.astype(np.uint32, copy=False)
    return ((u >> 1).astype(np.int32)) ^ (-(u & 1).astype(np.int32))


def reconstruct(code: np.ndarray, pred: np.ndarray, step: float) -> np.ndarray:
    """The decoder's exact arithmetic, float32 throughout.

    The encoder calls this too, so what it measures is what the decoder gets --
    the bound is never checked against a more precise calculation than the one
    that actually runs.
    """
    q = np.where(code > 0, code - 1, code).astype(np.float32)
    return pred.astype(np.float32, copy=False) + q * np.float32(step)


def quantize(x: np.ndarray, pred: np.ndarray, error_bound: float):
    """Return ``(zigzag_codes, outlier_index, outlier_values)``.

    ``x`` and ``pred`` are flat float32 arrays of equal length.  Every value is
    reconstructed with :func:`reconstruct` and checked; anything that still
    misses the bound (an error bound below the float32 ulp of the data, say) is
    escaped and stored verbatim rather than silently violating it.
    """
    step = quant_step(error_bound)
    q = np.rint((x.astype(np.float64) - pred.astype(np.float64)) / step)
    bad = (q < -CODE_MAX) | (q > CODE_MAX - 1) | ~np.isfinite(q)
    qi = np.where(bad, 0, q).astype(np.int32)
    # Shift non-negatives up by one so that code 0 is free to mean "escape".
    code = np.where(qi >= 0, qi + 1, qi).astype(np.int32)
    code[bad] = 0

    # NaN/Inf compare False here, but they were already caught above.
    bad |= np.abs(reconstruct(code, pred, step) - x) > error_bound
    code[bad] = 0

    idx = np.flatnonzero(bad).astype(np.uint32)
    return zigzag_encode(code), idx, x[bad].astype(np.float32)


def dequantize(
    z: np.ndarray,
    pred: np.ndarray,
    error_bound: float,
    outlier_index: np.ndarray,
    outlier_values: np.ndarray,
) -> np.ndarray:
    out = reconstruct(zigzag_decode(z), pred, quant_step(error_bound))
    if outlier_index.size:
        out[outlier_index.astype(np.int64)] = outlier_values
    return out


@dataclasses.dataclass
class EncodedChunk:
    """One window's worth of compressed payload."""

    index: int
    k: int
    tuple_size: int
    n_values: int
    mode: str
    centroids: np.ndarray  # (k, tuple_size) float32
    labels_blob: bytes
    labels_itemsize: int
    code_lo_blob: bytes = b""
    code_hi_blob: bytes = b""
    outlier_idx_blob: bytes = b""
    outlier_val_blob: bytes = b""
    n_outliers: int = 0

    @property
    def codebook_bytes(self) -> int:
        return int(self.centroids.nbytes)

    @property
    def label_bytes(self) -> int:
        return len(self.labels_blob)

    @property
    def code_bytes(self) -> int:
        return len(self.code_lo_blob) + len(self.code_hi_blob)

    @property
    def outlier_bytes(self) -> int:
        return len(self.outlier_idx_blob) + len(self.outlier_val_blob)


def pack_chunk(
    index: int,
    centroids: np.ndarray,
    labels: np.ndarray,
    zcodes: np.ndarray | None,
    outlier_index: np.ndarray,
    outlier_values: np.ndarray,
    *,
    n_values: int,
    tuple_size: int,
    mode: str,
    level: int = 3,
    planes: tuple[np.ndarray, np.ndarray] | None = None,
) -> EncodedChunk:
    """Compress one window.

    Either ``zcodes`` (uint16 zigzag words) or ``planes`` (the low/high uint8
    planes, already split by the GPU path) must be supplied in residual mode.
    """
    cctx = zstd.ZstdCompressor(level=level)
    k = int(centroids.shape[0])
    ldt = labels_dtype(k)
    lab = labels.astype(ldt, copy=False)
    chunk = EncodedChunk(
        index=index,
        k=k,
        tuple_size=tuple_size,
        n_values=n_values,
        mode=mode,
        centroids=np.ascontiguousarray(centroids, dtype=np.float32),
        labels_blob=cctx.compress(lab.tobytes()),
        labels_itemsize=ldt.itemsize,
    )
    if planes is not None:
        lo, hi = planes
        chunk.code_lo_blob = cctx.compress(np.ascontiguousarray(lo, np.uint8).tobytes())
        chunk.code_hi_blob = cctx.compress(np.ascontiguousarray(hi, np.uint8).tobytes())
    elif zcodes is not None:
        z = np.ascontiguousarray(zcodes, dtype=np.uint16)
        chunk.code_lo_blob = cctx.compress((z & 0xFF).astype(np.uint8).tobytes())
        chunk.code_hi_blob = cctx.compress((z >> 8).astype(np.uint8).tobytes())
    if outlier_index.size:
        # Indices are sorted and sparse: delta coding keeps them tiny.
        deltas = np.diff(outlier_index.astype(np.int64), prepend=0).astype(np.uint32)
        chunk.outlier_idx_blob = cctx.compress(deltas.tobytes())
        chunk.outlier_val_blob = cctx.compress(
            np.ascontiguousarray(outlier_values, dtype=np.float32).tobytes()
        )
        chunk.n_outliers = int(outlier_index.size)
    return chunk


def unpack_chunk(chunk: EncodedChunk, error_bound: float) -> np.ndarray:
    """Reconstruct the window's float32 values."""
    dctx = zstd.ZstdDecompressor()
    ldt = np.dtype({1: np.uint8, 2: np.uint16, 4: np.uint32}[chunk.labels_itemsize])
    # The last window is padded up to a whole number of tuples; the pad is coded
    # like any other value and dropped after reconstruction.
    n_tuples = -(-chunk.n_values // chunk.tuple_size)
    labels = np.frombuffer(dctx.decompress(chunk.labels_blob), dtype=ldt)[:n_tuples]
    pred = chunk.centroids[labels.astype(np.int64)].reshape(-1)

    if chunk.mode == "vq":
        return pred.astype(np.float32)[: chunk.n_values]

    lo = np.frombuffer(dctx.decompress(chunk.code_lo_blob), dtype=np.uint8)
    hi = np.frombuffer(dctx.decompress(chunk.code_hi_blob), dtype=np.uint8)
    z = (lo.astype(np.uint16)) | (hi.astype(np.uint16) << 8)

    if chunk.n_outliers:
        deltas = np.frombuffer(dctx.decompress(chunk.outlier_idx_blob), dtype=np.uint32)
        oidx = np.cumsum(deltas.astype(np.int64))
        ovals = np.frombuffer(dctx.decompress(chunk.outlier_val_blob), dtype=np.float32)
    else:
        oidx = np.empty(0, dtype=np.int64)
        ovals = np.empty(0, dtype=np.float32)

    return dequantize(z, pred, error_bound, oidx, ovals)[: chunk.n_values]
