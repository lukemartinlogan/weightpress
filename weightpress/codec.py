"""Error-bounded residual quantization and the lossless integer back end.

The k-means codebook supplies a prediction ``p`` for every weight.  The residual
is quantized onto a grid a shade narrower than ``2 * error_bound``::

    q     = round((x - p) / step)
    x_hat = p + q * step              =>   |x - x_hat| <= eb

so the bound holds by construction for every value, whatever k turns out to be.
See :func:`quant_step` for why the grid is narrowed and why the step is stored
per window rather than re-derived from the bound.  Code ``0`` is reserved as an
escape: values the grid cannot represent are stored verbatim as float32 and
reconstruct exactly.

The integer stream is zigzagged (so near-zero residuals become near-zero
unsigned words), split into byte planes, and handed to zstd.  The split matters:
the high planes of a Gaussian residual are almost all zeros and collapse, while
mixing them with the noisy low plane would defeat the entropy coder.  The number
of planes is chosen per window from the residual range actually observed.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import zstandard as zstd

from .config import CODE_MAX, MAX_CODE_PLANES

__all__ = [
    "EncodedChunk",
    "dequantize",
    "join_planes",
    "labels_dtype",
    "log1p_step",
    "pack_chunk",
    "pack_signs",
    "planes_needed",
    "quant_step",
    "quantize",
    "reconstruct",
    "reconstruct_relative",
    "split_planes",
    "unpack_chunk",
    "unpack_signs",
    "zigzag_decode",
    "zigzag_encode",
]

#: Relative slack in the grid, so a value sitting exactly on the bound is not
#: pushed past it by float32 rounding in ``pred + q * step``.
QUANT_MARGIN = 1e-4
#: Absolute slack, as a fraction of the largest magnitude in the window.  This
#: term is the one that matters when the bound is tight relative to the data:
#: float32 rounding scales with the *value*, not with the bound, so at x~1.0 and
#: eb=1e-5 the relative margin above is 1e-9 while a single ulp is already 6e-8.
#: 2^-22 is four ulps, which also covers the centroid's own rounding.
QUANT_ULP_GUARD = 2.0**-22
#: Ceiling on the guard as a fraction of the bound.  ``max_abs`` is the window's
#: single largest magnitude, but the rounding that matters is at each *value's*
#: magnitude, so sizing the guard off the extreme would narrow the grid for
#: everyone: at eb=1e-5 on gpt2 (max |w| ~ 17) an unclamped guard halved the step
#: and cost a full bit per value.  Clamping keeps the grid within 5% of ideal and
#: lets the rare genuinely-huge value take the escape path instead.
QUANT_MAX_GUARD = 0.05
#: Bytes an escaped value costs: float32 payload plus its delta-coded index.
ESCAPE_COST_BYTES = 8


def quant_step(error_bound: float, max_abs: float = 0.0) -> float:
    """Grid width for a window whose largest magnitude is ``max_abs``.

    The step is stored per chunk rather than re-derived on read: it depends on
    the data, and the decoder must use bit-identical arithmetic to the encoder.
    """
    guard = max(error_bound * QUANT_MARGIN, max_abs * QUANT_ULP_GUARD)
    guard = min(guard, error_bound * QUANT_MAX_GUARD)
    return 2.0 * (error_bound - guard)


def choose_plane_width(
    plane_hists: list[np.ndarray], over_counts: list[int], n_values: int
) -> int:
    """Pick the code width that minimises plane cost plus escape cost.

    Widening the code and escaping the overflow are two ways to pay for the same
    outliers, and which is cheaper depends on the window.  At eb=1e-4 a 2-byte
    code escapes ~50 values in 137M and a third plane would be pure overhead; at
    eb=1e-5 the residual range grows until the escape list is a third of the
    output.  Choosing by estimated bytes handles both without a magic constant.

    ``plane_hists[i]`` is a 256-bin byte histogram for plane ``i`` (it may come
    from a subsample; only the shape of the distribution is used).  Costs are in
    bytes: plane ``i`` contributes its zeroth-order entropy, which is what the
    zstd literal coder gets close to, and each overflow costs an escape entry.
    """
    best_w, best_cost = 1, None
    cumulative = 0.0
    for w in range(1, len(plane_hists) + 1):
        cumulative += _byte_entropy(plane_hists[w - 1]) * n_values / 8.0
        cost = cumulative + ESCAPE_COST_BYTES * over_counts[w - 1]
        if best_cost is None or cost < best_cost:
            best_w, best_cost = w, cost
    return best_w


def plane_limit(width: int) -> int | None:
    """Smallest zigzag code that will not fit in ``width`` bytes, or None."""
    return None if width >= MAX_CODE_PLANES else 1 << (8 * width)


#: Values sampled when histogramming a plane; only the shape is needed.
PLANE_SAMPLE = 1 << 21


def plane_stats(z: np.ndarray) -> tuple[list[np.ndarray], list[int], int]:
    """Per-plane byte histograms (subsampled) and exact overflow counts."""
    stride = max(1, z.size // PLANE_SAMPLE)
    zs = z[::stride]
    hists = [
        np.bincount(((zs >> (8 * i)) & 0xFF).astype(np.uint8), minlength=256)
        for i in range(MAX_CODE_PLANES)
    ]
    over = []
    for w in range(1, MAX_CODE_PLANES + 1):
        limit = plane_limit(w)
        over.append(0 if limit is None else int((z >= limit).sum()))
    return hists, over, int(z.size)


def _byte_entropy(hist: np.ndarray) -> float:
    """Zeroth-order entropy of a byte stream, in bits per byte."""
    counts = np.asarray(hist, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def labels_dtype(k: int) -> np.dtype:
    if k <= 1 << 8:
        return np.dtype(np.uint8)
    if k <= 1 << 16:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def zigzag_encode(code: np.ndarray) -> np.ndarray:
    """Signed int32 -> uint32, small magnitudes to small words."""
    c = code.astype(np.int32, copy=False)
    return ((c << 1) ^ (c >> 31)).astype(np.uint32)


def zigzag_decode(z: np.ndarray) -> np.ndarray:
    u = z.astype(np.uint32, copy=False)
    return ((u >> 1).astype(np.int32)) ^ (-(u & 1).astype(np.int32))


def planes_needed(max_zigzag: int) -> int:
    """Byte planes required to hold the widest code in a window."""
    n = 1
    while n < MAX_CODE_PLANES and max_zigzag >= (1 << (8 * n)):
        n += 1
    return n


def split_planes(z: np.ndarray, n_planes: int) -> list[np.ndarray]:
    zz = z.astype(np.uint32, copy=False)
    return [((zz >> (8 * i)) & 0xFF).astype(np.uint8) for i in range(n_planes)]


def join_planes(planes: list[np.ndarray]) -> np.ndarray:
    z = np.zeros(planes[0].size, dtype=np.uint32)
    for i, p in enumerate(planes):
        z |= p.astype(np.uint32) << (8 * i)
    return z


def reconstruct(code: np.ndarray, pred: np.ndarray, step: float) -> np.ndarray:
    """The decoder's exact arithmetic, float32 throughout.

    The encoder calls this too, so what it measures is what the decoder gets --
    the bound is never checked against a more precise calculation than the one
    that actually runs.
    """
    q = np.where(code > 0, code - 1, code).astype(np.float32)
    return pred.astype(np.float32, copy=False) + q * np.float32(step)


def quantize(x: np.ndarray, pred: np.ndarray, error_bound: float):
    """Return ``(zigzag_codes, outlier_index, outlier_values, step)``.

    ``x`` and ``pred`` are flat float32 arrays of equal length.  Every value is
    reconstructed with :func:`reconstruct` and checked; anything that still
    misses the bound (an error bound below the float32 ulp of the data, say) is
    escaped and stored verbatim rather than silently violating it.
    """
    finite = x[np.isfinite(x)]
    step = quant_step(error_bound, float(np.abs(finite).max()) if finite.size else 0.0)
    q = np.rint((x.astype(np.float64) - pred.astype(np.float64)) / step)
    bad = (q < -CODE_MAX) | (q > CODE_MAX - 1) | ~np.isfinite(q)
    qi = np.where(bad, 0, q).astype(np.int32)
    # Shift non-negatives up by one so that code 0 is free to mean "escape".
    code = np.where(qi >= 0, qi + 1, qi).astype(np.int32)
    code[bad] = 0

    # NaN/Inf compare False here, but they were already caught above.
    bad |= np.abs(reconstruct(code, pred, step) - x) > error_bound
    code[bad] = 0
    z = zigzag_encode(code)

    # Decide how wide the code should be, then escape whatever does not fit.
    width = choose_plane_width(*plane_stats(z))
    limit = plane_limit(width)
    if limit is not None:
        wide = z >= limit
        if wide.any():
            bad |= wide
            code[wide] = 0
            z = zigzag_encode(code)

    idx = np.flatnonzero(bad).astype(np.uint32)
    return z, idx, x[bad].astype(np.float32), step


def dequantize(
    z: np.ndarray,
    pred: np.ndarray,
    step: float,
    outlier_index: np.ndarray,
    outlier_values: np.ndarray,
) -> np.ndarray:
    out = reconstruct(zigzag_decode(z), pred, step)
    if outlier_index.size:
        out[outlier_index.astype(np.int64)] = outlier_values
    return out


def log1p_step(error_bound: float, max_abs_log: float = 0.0) -> float:
    """Log-domain grid width that yields a relative error at most ``error_bound``.

    A residual within +/-d in log space reconstructs to a linear relative error
    of ``exp(d) - 1``, so a full grid step of ``2 * ln(1 + eb)`` bounds it at
    ``eb``.  The margin (finer grid) and the per-window magnitude guard come from
    :func:`quant_step`, here applied in log space.
    """
    return quant_step(float(np.log1p(error_bound)), max_abs_log)


def reconstruct_relative(
    code: np.ndarray, pred_log: np.ndarray, step: float, sign: np.ndarray
) -> np.ndarray:
    """Decoder arithmetic for the relative (log-domain) path.

    ``sign`` is +1.0 / -1.0 per value.  Must be bit-identical to what the encoder
    checked against, so the same float32 ``exp`` runs on both sides.
    """
    u_hat = reconstruct(code, pred_log, step)
    return (sign * np.exp(u_hat)).astype(np.float32)


def pack_signs(neg: np.ndarray) -> bytes:
    """Bit-pack a boolean 'is negative' mask (LSB-first)."""
    return np.packbits(np.ascontiguousarray(neg, dtype=bool), bitorder="little").tobytes()


def unpack_signs(blob: bytes, n: int) -> np.ndarray:
    """Return +1.0 / -1.0 per value from a packed sign mask."""
    bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8), bitorder="little")[:n]
    return np.where(bits.astype(bool), np.float32(-1.0), np.float32(1.0))


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
    #: Grid width used by this window; the decoder must reuse it exactly.
    step: float = 0.0
    #: "relative" (log-domain) or "absolute" (linear grid).
    error_mode: str = "absolute"
    code_plane_blobs: list[bytes] = dataclasses.field(default_factory=list)
    outlier_idx_blob: bytes = b""
    outlier_val_blob: bytes = b""
    #: Bit-packed sign mask, only present in relative mode.
    sign_blob: bytes = b""
    n_outliers: int = 0

    @property
    def n_planes(self) -> int:
        return len(self.code_plane_blobs)

    @property
    def sign_bytes(self) -> int:
        return len(self.sign_blob)

    @property
    def codebook_bytes(self) -> int:
        return int(self.centroids.nbytes)

    @property
    def label_bytes(self) -> int:
        return len(self.labels_blob)

    @property
    def code_bytes(self) -> int:
        return sum(len(b) for b in self.code_plane_blobs)

    @property
    def outlier_bytes(self) -> int:
        return len(self.outlier_idx_blob) + len(self.outlier_val_blob) + len(self.sign_blob)


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
    step: float = 0.0,
    error_mode: str = "absolute",
    signs: np.ndarray | None = None,
    level: int = 3,
    planes: list[np.ndarray] | None = None,
) -> EncodedChunk:
    """Compress one window.

    In residual mode supply either ``zcodes`` (uint32 zigzag words) or
    ``planes`` (uint8 byte planes already split by the GPU path).  In relative
    mode ``signs`` is a boolean 'is negative' mask over every value.
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
        step=float(step),
        error_mode=error_mode,
    )
    if signs is not None:
        # Signs are ~1 incompressible bit each, but zstd still trims the runs of
        # same-sign weights that show up in practice.
        chunk.sign_blob = cctx.compress(pack_signs(signs))
    if planes is None and zcodes is not None:
        z = np.ascontiguousarray(zcodes, dtype=np.uint32)
        planes = split_planes(z, planes_needed(int(z.max()) if z.size else 0))
    if planes is not None:
        chunk.code_plane_blobs = [
            cctx.compress(np.ascontiguousarray(p, np.uint8).tobytes()) for p in planes
        ]
    if outlier_index.size:
        # Indices are sorted and sparse: delta coding keeps them tiny.
        deltas = np.diff(outlier_index.astype(np.int64), prepend=0).astype(np.uint32)
        chunk.outlier_idx_blob = cctx.compress(deltas.tobytes())
        chunk.outlier_val_blob = cctx.compress(
            np.ascontiguousarray(outlier_values, dtype=np.float32).tobytes()
        )
        chunk.n_outliers = int(outlier_index.size)
    return chunk


def unpack_chunk(chunk: EncodedChunk) -> np.ndarray:
    """Reconstruct the window's float32 values.

    Takes no error bound: the chunk carries the exact step the encoder used, so
    the decoder can never disagree with it about the grid.
    """
    dctx = zstd.ZstdDecompressor()
    ldt = np.dtype({1: np.uint8, 2: np.uint16, 4: np.uint32}[chunk.labels_itemsize])
    # The last window is padded up to a whole number of tuples; the pad is coded
    # like any other value and dropped after reconstruction.
    n_tuples = -(-chunk.n_values // chunk.tuple_size)
    labels = np.frombuffer(dctx.decompress(chunk.labels_blob), dtype=ldt)[:n_tuples]
    pred = chunk.centroids[labels.astype(np.int64)].reshape(-1)

    if chunk.mode == "vq":
        return pred.astype(np.float32)[: chunk.n_values]

    z = join_planes(
        [
            np.frombuffer(dctx.decompress(b), dtype=np.uint8)
            for b in chunk.code_plane_blobs
        ]
    )

    if chunk.n_outliers:
        deltas = np.frombuffer(dctx.decompress(chunk.outlier_idx_blob), dtype=np.uint32)
        oidx = np.cumsum(deltas.astype(np.int64))
        ovals = np.frombuffer(dctx.decompress(chunk.outlier_val_blob), dtype=np.float32)
    else:
        oidx = np.empty(0, dtype=np.int64)
        ovals = np.empty(0, dtype=np.float32)

    if chunk.error_mode == "relative":
        n_padded = n_tuples * chunk.tuple_size
        sign = unpack_signs(dctx.decompress(chunk.sign_blob), n_padded)
        out = reconstruct_relative(zigzag_decode(z), pred, chunk.step, sign)
        if oidx.size:
            out[oidx] = ovals
        return out[: chunk.n_values]

    return dequantize(z, pred, chunk.step, oidx, ovals)[: chunk.n_values]
