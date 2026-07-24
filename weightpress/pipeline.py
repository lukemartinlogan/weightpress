"""Window scheduling: several 128 MB windows are clustered concurrently.

Concurrency is chosen from the GPU memory budget (default 80% of free memory)
divided by the estimated peak footprint of one window, capped by
``Config.max_workers``.  Each worker owns a CUDA stream and never shares device
tensors with another worker, so the caching allocator stays safe without
``record_stream`` bookkeeping.

Threads (not processes) are used deliberately: the heavy stages -- CUDA kernel
launches, host/device copies and zstd -- all release the GIL, so the reader,
the GPU and the entropy coder overlap.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from . import kmeans as km
from .codec import (
    PLANE_SAMPLE,
    EncodedChunk,
    choose_plane_width,
    labels_dtype,
    log1p_step,
    pack_chunk,
    plane_limit,
    plane_stats,
    planes_needed,
    quant_step,
    reconstruct_relative,
    split_planes,
    unpack_chunk,
    zigzag_decode,
)
from .config import CODE_MAX, MAX_CODE_PLANES, Config
from .container import ContainerReader, ContainerWriter, write_codebook_sidecar
from .gpu import (
    available_host_memory,
    default_budget,
    device_name,
    estimate_window_bytes,
    plan_concurrency,
    resolve_device,
    torch,
)
from .reader import WeightStream
from .stats import ChunkStats, RunStats

_local = threading.local()


def _stream_ctx(device: str):
    """A per-thread CUDA stream so concurrent windows really do overlap."""
    if not device.startswith("cuda"):
        return contextlib.nullcontext()
    t = torch()
    s = getattr(_local, "stream", None)
    if s is None:
        s = t.cuda.Stream(device=device)
        _local.stream = s
    return t.cuda.stream(s)


def warmup(device: str) -> None:
    """Force CUDA context and cuBLAS kernel selection before timing starts.

    Without this the first window absorbs several seconds of one-off
    initialisation and its reported throughput is meaningless.
    """
    if not device.startswith("cuda"):
        return
    from .config import Config

    cfg = Config(window_size=1 << 20, max_k=64, kmeans_iters=2, zstd_level=1)
    compress_window(0, np.zeros(1 << 16, dtype=np.float32), cfg, device)
    torch().cuda.synchronize()


def _upload(x: np.ndarray, tuple_size: int, device: str):
    """Upload one window as a flat fp32 tensor, edge-padded to a whole tuple."""
    t = torch()
    rem = (-x.size) % tuple_size
    dev = t.from_numpy(np.ascontiguousarray(x, dtype=np.float32).copy()).to(
        device, non_blocking=True
    )
    if rem:
        dev = t.cat([dev, dev[-1:].repeat(rem)])
    return dev


def _to_tuples(x: np.ndarray, tuple_size: int, device: str):
    """The feature the k-means and residual stages see: raw weights as tuples."""
    return _upload(x, tuple_size, device).view(-1, tuple_size)


def _to_log_tuples(x: np.ndarray, tuple_size: int, device: str):
    """Relative mode works in the log domain, where an absolute residual bound
    becomes a relative one.  Returns ``(u2d, raw, sign, escaped)``: log|x| as
    tuples for the predictor, plus the flat linear values, per-value sign, and
    the escape mask for values (zeros, non-finite) that have no useful log."""
    t = torch()
    raw = _upload(x, tuple_size, device)
    absx = raw.abs()
    escaped = (raw == 0) | ~t.isfinite(raw)
    # Clamp only so log() is finite on escaped lanes; those lanes are overwritten.
    u = t.log(t.where(escaped, t.ones_like(absx), absx))
    sign = raw < 0
    return u.view(-1, tuple_size), raw, sign, escaped


def _quantize_gpu(x2d, centroids, labels, error_bound: float, k: int):
    """Residual quantization on the device.

    Returns ``(byte_planes, outlier_index, outlier_values, max_err,
    mean_abs_err, step)``.  The planes are split on the device and only the ones this
    window actually needs are copied back, so the host never materialises a
    wide integer buffer.
    """
    t = torch()
    n, tsize = x2d.shape
    n_values = n * tsize
    # The step depends on the window's largest magnitude, so it is computed here
    # and stored with the chunk rather than re-derived by the decoder.
    finite = x2d[t.isfinite(x2d)]
    max_abs = float(finite.abs().max().item()) if finite.numel() else 0.0
    step = quant_step(error_bound, max_abs)

    zcodes = t.empty(n_values, dtype=t.int32, device=x2d.device)
    max_err = t.zeros((), dtype=t.float32, device=x2d.device)
    abs_sum = t.zeros((), dtype=t.float64, device=x2d.device)
    # One escape mask for the whole window: torch.nonzero forces a device sync,
    # so doing it per tile would stall the stream on every iteration.
    escaped = t.zeros(n_values, dtype=t.bool, device=x2d.device)

    tile = km._resid_tile(n, tsize)
    for start in range(0, n, tile):
        stop = min(n, start + tile)
        src = x2d[start:stop]
        pred = centroids[labels[start:stop]]
        q = t.round((src.double() - pred.double()) / step)
        bad = (q < -CODE_MAX) | (q > CODE_MAX - 1) | ~t.isfinite(q)
        q = t.where(bad, t.zeros_like(q), q).to(t.int32)
        # Shift non-negatives up by one so code 0 is free to mean "escape".
        code = t.where(bad, t.zeros_like(q), t.where(q >= 0, q + 1, q))

        # Reconstruct exactly as the decoder will, then escape anything that
        # still misses the bound.  This is the check that makes the guarantee
        # hold rather than merely being very likely.
        q_back = t.where(code > 0, code - 1, code).to(t.float32)
        bad |= ((pred + q_back * step) - src).abs() > error_bound
        code = t.where(bad, t.zeros_like(code), code)

        # zigzag; |code| <= CODE_MAX keeps this inside int32's positive range
        zcodes[start * tsize : stop * tsize] = ((code << 1) ^ (code >> 31)).reshape(-1)

        q_back = t.where(code > 0, code - 1, code).to(t.float32)
        recon = t.where(bad, src, pred + q_back * step)
        err = (src - recon).abs()
        max_err = t.maximum(max_err, err.max())
        abs_sum += err.sum(dtype=t.float64)
        escaped[start * tsize : stop * tsize] = bad.reshape(-1)

    # Widening the code and escaping the overflow both pay for the same
    # outliers; pick whichever is cheaper for this window, then escape the rest.
    zu = zcodes.view(t.int32)
    width = choose_plane_width(*_plane_stats_gpu(zu, n_values))
    limit = plane_limit(width)
    if limit is not None:
        wide = zu >= limit
        escaped |= wide
        zcodes = t.where(wide, t.zeros_like(zcodes), zcodes)

    pos = t.nonzero(escaped, as_tuple=False).flatten()
    if pos.numel():
        oidx = pos.cpu().numpy().astype(np.uint32)
        oval = x2d.reshape(-1)[pos].cpu().numpy().astype(np.float32)
    else:
        oidx = np.empty(0, dtype=np.uint32)
        oval = np.empty(0, dtype=np.float32)

    # Split on the device and copy only the planes the window actually needs.
    n_planes = planes_needed(int(zcodes.max().item()) if n_values else 0)
    planes = [
        ((zcodes >> (8 * i)) & 0xFF).to(t.uint8).cpu().numpy()
        for i in range(n_planes)
    ]

    return (
        planes,
        oidx,
        oval,
        float(max_err.item()),
        float(abs_sum.item() / max(1, n_values)),
        step,
    )


def _quantize_gpu_relative(u2d, raw, sign, escaped, centroids, labels, eb, k):
    """Log-domain residual quantization for a relative (max percentage) bound.

    ``centroids`` predict ``log|x|``.  A residual within +/-step/2 in log space
    is a relative error of ``exp(step/2)-1``, so ``step = 2*ln(1+eb)`` bounds it.

    The GPU rounds each log-residual to the grid; the authoritative
    reconstruct-and-check runs on the CPU with the decoder's exact numpy ``exp``
    (:func:`_finalize_relative`).  Checking with the same ``exp`` the decoder
    uses is what makes the bound hold -- the GPU ``exp`` differs from numpy's by
    a few ULP, which at a tight bound is enough to cross it.

    Returns ``(planes, oidx, oval, max_rel_err, mean_rel_err, step, sign_np)``.
    """
    t = torch()
    n, tsize = u2d.shape
    n_values = n * tsize
    finite = u2d[t.isfinite(u2d)]
    max_abs_log = float(finite.abs().max().item()) if finite.numel() else 0.0
    step = log1p_step(eb, max_abs_log)

    zcodes = t.empty(n_values, dtype=t.int32, device=u2d.device)
    tile = km._resid_tile(n, tsize)
    for a in range(0, n, tile):
        b = min(n, a + tile)
        pred = centroids[labels[a:b]]
        q = t.round((u2d[a:b].double() - pred.double()) / step)
        over = (q < -CODE_MAX) | (q > CODE_MAX - 1) | ~t.isfinite(q)
        q = t.where(over, t.zeros_like(q), q).to(t.int32)
        code = t.where(over, t.zeros_like(q), t.where(q >= 0, q + 1, q))
        zcodes[a * tsize : b * tsize] = ((code << 1) ^ (code >> 31)).reshape(-1)

    pred_flat = centroids[labels].reshape(-1).cpu().numpy()
    return _finalize_relative(
        zcodes.cpu().numpy().astype(np.uint32),
        pred_flat,
        sign.reshape(-1).cpu().numpy(),
        escaped.reshape(-1).cpu().numpy(),
        raw.cpu().numpy(),
        step, eb, n_values,
    )


#: Values per tile in the relative finalize; bounds host peak on RAM-limited
#: boxes.  The persistent arrays (z, raw, pred, sign, bad) are unavoidable; the
#: float64 reconstruct/compare temporaries are what tiling keeps small.
FINALIZE_TILE = 1 << 22


def _finalize_relative(z, pred_log, neg, escaped, raw, step, eb, n_values):
    """CPU, decoder-exact: reconstruct, escape every value still over the bound,
    then choose the code width.  Because this uses the same numpy ``exp`` the
    decoder does, a value that passes here cannot fail on decode.

    The reconstruct-and-compare is tiled so the float64 temporaries never span
    the whole window at once -- 8 concurrent full-window passes OOM an 11 GiB
    host, and this is the binding constraint in relative mode, not GPU memory.
    """
    bad = escaped | (raw == 0) | ~np.isfinite(raw)
    max_rel = 0.0
    sum_rel = 0.0
    for lo in range(0, n_values, FINALIZE_TILE):
        hi = min(n_values, lo + FINALIZE_TILE)
        sl = slice(lo, hi)
        sign = np.where(neg[sl], np.float32(-1.0), np.float32(1.0))
        x_hat = reconstruct_relative(zigzag_decode(z[sl]), pred_log[sl], step, sign)
        o = raw[sl].astype(np.float64)
        rel = np.abs(x_hat.astype(np.float64) - o) / np.maximum(np.abs(o), 1e-300)
        over = ~bad[sl] & (rel > eb)
        bad[sl] |= over
        rel = np.where(bad[sl], 0.0, rel)  # escapes reconstruct exactly
        max_rel = max(max_rel, float(rel.max()) if rel.size else 0.0)
        sum_rel += float(rel.sum())

    # Escape overflow past the chosen code width, same cost trade-off as linear.
    z = np.where(bad, 0, z).astype(np.uint32)
    width = choose_plane_width(*plane_stats(z))
    limit = plane_limit(width)
    if limit is not None:
        wide = z >= limit
        bad |= wide
        z = np.where(bad, 0, z).astype(np.uint32)

    oidx = np.flatnonzero(bad).astype(np.uint32)
    oval = raw[bad].astype(np.float32)
    planes = split_planes(z, planes_needed(int(z.max()) if z.size else 0))
    return (planes, oidx, oval, max_rel, sum_rel / max(1, n_values), step, neg)


def _plane_stats_gpu(z, n_values: int):
    """Device-side equivalent of :func:`codec.plane_stats`."""
    t = torch()
    stride = max(1, z.numel() // PLANE_SAMPLE)
    zs = z[::stride]
    hists = [
        t.bincount(((zs >> (8 * i)) & 0xFF).to(t.int64), minlength=256).cpu().numpy()
        for i in range(MAX_CODE_PLANES)
    ]
    over = []
    for w in range(1, MAX_CODE_PLANES + 1):
        limit = plane_limit(w)
        over.append(0 if limit is None else int((z >= limit).sum().item()))
    return hists, over, n_values


def _vq_error(x2d, centroids, labels, k: int) -> tuple[float, float]:
    t = torch()
    n = x2d.shape[0]
    max_err = t.zeros((), dtype=t.float32, device=x2d.device)
    abs_sum = t.zeros((), dtype=t.float64, device=x2d.device)
    tile = km._resid_tile(n, x2d.shape[1])
    for lo in range(0, n, tile):
        hi = min(n, lo + tile)
        err = (x2d[lo:hi] - centroids[labels[lo:hi]]).abs()
        max_err = t.maximum(max_err, err.max())
        abs_sum += err.sum(dtype=t.float64)
    return float(max_err.item()), float(abs_sum.item() / max(1, x2d.numel()))


def compress_window(
    index: int, values: np.ndarray, cfg: Config, device: str
) -> tuple[EncodedChunk, ChunkStats]:
    """k search + residual coding for a single window."""
    t = torch()
    t0 = time.time()
    n_values = int(values.size)

    # Pure-VQ mode is a demonstration of the linear construction; it has no
    # residual stage to carry a relative bound, so it stays in linear space.
    relative = cfg.error_mode == "relative" and cfg.mode != "vq"
    with _stream_ctx(device):
        gen = t.Generator(device=device)
        gen.manual_seed(cfg.seed + index)
        if relative:
            x2d, raw, sign, escaped = _to_log_tuples(values, cfg.tuple_size, device)
        else:
            x2d = _to_tuples(values, cfg.tuple_size, device)

        res = km.search_k(x2d, cfg, generator=gen)
        k = int(res.centroids.shape[0])
        centroids_np = res.centroids.cpu().numpy()
        # Narrow before the copy: int64 labels are 8x the bytes actually needed
        # and the transfer dominated the window at k=64.
        narrow = {1: t.uint8, 2: t.int16, 4: t.int32}[labels_dtype(k).itemsize]
        labels_np = res.labels.to(narrow).cpu().numpy().view(labels_dtype(k))
        vq_max, vq_mean = res.evaluation.vq_max_error, res.evaluation.vq_mean_abs_error

        if cfg.mode == "vq":
            vq_max, vq_mean = _vq_error(x2d, res.centroids, res.labels, k)
            chunk = pack_chunk(
                index, centroids_np, labels_np, None,
                np.empty(0, np.uint32), np.empty(0, np.float32),
                n_values=n_values, tuple_size=cfg.tuple_size,
                mode=cfg.mode, level=cfg.zstd_level,
            )
            max_err, mean_err, n_out = vq_max, vq_mean, 0
        elif relative:
            planes, oidx, oval, max_err, mean_err, step, signs = _quantize_gpu_relative(
                x2d, raw, sign, escaped, res.centroids, res.labels, cfg.error_bound, k
            )
            # The search minimised log-domain error; report it as relative too.
            vq_max, vq_mean = float(np.expm1(vq_max)), float(np.expm1(vq_mean))
            chunk = pack_chunk(
                index, centroids_np, labels_np, None, oidx, oval,
                n_values=n_values, tuple_size=cfg.tuple_size, mode=cfg.mode,
                step=step, error_mode="relative", signs=signs,
                level=cfg.zstd_level, planes=planes,
            )
            n_out = int(oidx.size)
        else:
            planes, oidx, oval, max_err, mean_err, step = _quantize_gpu(
                x2d, res.centroids, res.labels, cfg.error_bound, k
            )
            chunk = pack_chunk(
                index, centroids_np, labels_np, None, oidx, oval,
                n_values=n_values, tuple_size=cfg.tuple_size,
                mode=cfg.mode, step=step, error_mode="absolute",
                level=cfg.zstd_level, planes=planes,
            )
            n_out = int(oidx.size)

        del x2d
        if device.startswith("cuda"):
            t.cuda.current_stream().synchronize()

    stats = ChunkStats(
        index=index,
        n_values=n_values,
        raw_bytes=n_values * 4,
        k=k,
        k_trials=[e.to_dict() for e in res.trials],
        vq_max_error=vq_max,
        max_error=max_err,
        mean_abs_error=mean_err,
        n_outliers=n_out,
        codebook_bytes=chunk.codebook_bytes,
        label_bytes=chunk.label_bytes,
        code_bytes=chunk.code_bytes,
        outlier_bytes=chunk.outlier_bytes,
        seconds=time.time() - t0,
    )
    return chunk, stats


def compress(
    source: str,
    cfg: Config,
    *,
    dtype: str = "float32",
    limit_bytes: int | None = None,
    progress: Callable[[ChunkStats], None] | None = None,
) -> tuple[str, RunStats]:
    """Compress ``source`` into ``cfg.output_dir``; returns (container path, stats)."""
    t_start = time.time()
    device = resolve_device(cfg.device)
    out_dir = cfg.resolved_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(source))[0]
    out_path = os.path.join(out_dir, stem + ".wp")

    warmup(device)
    stream = WeightStream.open(source, dtype=dtype, limit_bytes=limit_bytes)
    values_per_window = cfg.window_size // 4  # window is fp32 working bytes

    if device.startswith("cuda"):
        # Blocks the caching allocator still holds from an earlier call in this
        # process count as "used" to mem_get_info.  Without releasing them, a
        # second compress() in the same process reads a stale free figure and
        # then runs right up against the device limit.
        torch().cuda.empty_cache()
    budget = (
        cfg.max_gpu_memory
        if cfg.max_gpu_memory is not None
        else default_budget(device)
    )
    per_window = estimate_window_bytes(
        values_per_window, cfg.tuple_size, km.DIST_BUFFER_BYTES,
        km.RESID_TILE_VALUES,
    )
    concurrency = plan_concurrency(budget, per_window, cfg.max_workers)
    # Relative mode finalises each window on the host, holding several
    # full-window arrays at once (~10 bytes/value: z, codes, pred, raw, sign,
    # x_hat, rel).  On a RAM-limited box this, not GPU memory, is the binding
    # constraint -- 8 workers x ~1.3 GiB OOM-killed a 11 GiB host.
    if cfg.error_mode == "relative" and cfg.mode != "vq":
        # Persistent host arrays per window: z (u32), raw (f32), pred (f32),
        # sign+escape+bad (bool*3), plus tiled temporaries -- ~18 bytes/value.
        host_per_window = values_per_window * 18
        host_budget = int(available_host_memory() * 0.5)
        concurrency = min(
            concurrency, plan_concurrency(host_budget, host_per_window, cfg.max_workers)
        )

    run = RunStats(
        source=os.path.abspath(source),
        error_bound=cfg.error_bound,
        tuple_size=cfg.tuple_size,
        window_size=cfg.window_size,
        mode=cfg.mode,
        concurrency=concurrency,
        gpu_budget_bytes=budget,
    )

    meta = {
        "error_bound": cfg.error_bound,
        "tuple_size": cfg.tuple_size,
        "window_size": cfg.window_size,
        "mode": cfg.mode,
        "error_mode": cfg.error_mode,
        "device": device_name(device),
        "source": stream.manifest(),
    }

    with ContainerWriter(out_path, meta) as writer, ThreadPoolExecutor(
        max_workers=concurrency, thread_name_prefix="wp"
    ) as pool:
        pending: list[Future] = []

        def drain(one: bool) -> None:
            while pending and (one or len(pending) >= concurrency):
                chunk, st = pending.pop(0).result()
                writer.add(chunk)
                write_codebook_sidecar(
                    out_dir, stem, chunk.index, chunk.centroids,
                    {"k": chunk.k, "tuple_size": chunk.tuple_size,
                     "n_values": chunk.n_values, "error_bound": cfg.error_bound},
                )
                run.chunks.append(st)
                if progress:
                    progress(st)
                if one:
                    return

        for index, values in stream.windows(values_per_window):
            if len(pending) >= concurrency:
                drain(one=True)
            pending.append(pool.submit(compress_window, index, values, cfg, device))
        while pending:
            drain(one=True)

    run.seconds = time.time() - t_start
    return out_path, run


def decompress(path: str, *, out: str | None = None) -> tuple[np.ndarray, dict]:
    """Rebuild the full fp32 value stream from a container."""
    with ContainerReader(path) as reader:
        parts = []
        for i in range(len(reader.chunks)):
            parts.append(unpack_chunk(reader.read_chunk(i)))
        values = np.concatenate(parts) if parts else np.empty(0, np.float32)
        header = reader.header
    if out:
        np.save(out, values)
    return values, header


def iter_decompressed(path: str) -> Iterator[np.ndarray]:
    """Stream windows back one at a time (for verification without holding all)."""
    with ContainerReader(path) as reader:
        for i in range(len(reader.chunks)):
            yield unpack_chunk(reader.read_chunk(i))
