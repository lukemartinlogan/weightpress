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
from .codec import EncodedChunk, labels_dtype, pack_chunk, quant_step, unpack_chunk
from .config import CODE_MAX, Config
from .container import ContainerReader, ContainerWriter, write_codebook_sidecar
from .gpu import (
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


def _to_tuples(x: np.ndarray, tuple_size: int, device: str):
    """Upload one window and view it as (n_tuples, tuple_size), padding by edge
    replication so the tail tuple is still cheap to predict."""
    t = torch()
    n = x.size
    rem = (-n) % tuple_size
    dev = t.from_numpy(np.ascontiguousarray(x, dtype=np.float32).copy()).to(
        device, non_blocking=True
    )
    if rem:
        pad = dev[-1:].repeat(rem)
        dev = t.cat([dev, pad])
    return dev.view(-1, tuple_size)


def _quantize_gpu(x2d, centroids, labels, error_bound: float, k: int):
    """Residual quantization on the device.

    Returns ``(lo_plane, hi_plane, outlier_index, outlier_values, max_err,
    mean_abs_err)`` with the byte planes already split, so the host only sees
    two uint8 arrays and never materialises a wide integer buffer.
    """
    t = torch()
    n, tsize = x2d.shape
    n_values = n * tsize
    step = quant_step(error_bound)

    lo = t.empty(n_values, dtype=t.uint8, device=x2d.device)
    hi = t.empty(n_values, dtype=t.uint8, device=x2d.device)
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

        z = (code << 1) ^ (code >> 31)  # zigzag; fits in 16 bits by construction
        flat = z.reshape(-1)
        lo[start * tsize : stop * tsize] = (flat & 0xFF).to(t.uint8)
        hi[start * tsize : stop * tsize] = ((flat >> 8) & 0xFF).to(t.uint8)

        q_back = t.where(code > 0, code - 1, code).to(t.float32)
        recon = t.where(bad, src, pred + q_back * step)
        err = (src - recon).abs()
        max_err = t.maximum(max_err, err.max())
        abs_sum += err.sum(dtype=t.float64)
        escaped[start * tsize : stop * tsize] = bad.reshape(-1)

    pos = t.nonzero(escaped, as_tuple=False).flatten()
    if pos.numel():
        oidx = pos.cpu().numpy().astype(np.uint32)
        oval = x2d.reshape(-1)[pos].cpu().numpy().astype(np.float32)
    else:
        oidx = np.empty(0, dtype=np.uint32)
        oval = np.empty(0, dtype=np.float32)

    return (
        lo.cpu().numpy(),
        hi.cpu().numpy(),
        oidx,
        oval,
        float(max_err.item()),
        float(abs_sum.item() / max(1, n_values)),
    )


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

    with _stream_ctx(device):
        gen = t.Generator(device=device)
        gen.manual_seed(cfg.seed + index)
        x2d = _to_tuples(values, cfg.tuple_size, device)

        res = km.search_k(x2d, cfg, generator=gen)
        k = int(res.centroids.shape[0])
        centroids_np = res.centroids.cpu().numpy()
        # Narrow before the copy: int64 labels are 8x the bytes actually needed
        # and the transfer dominated the window at k=64.
        narrow = {1: t.uint8, 2: t.int16, 4: t.int32}[labels_dtype(k).itemsize]
        labels_np = res.labels.to(narrow).cpu().numpy().view(labels_dtype(k))

        if cfg.mode == "vq":
            vq_max, vq_mean = _vq_error(x2d, res.centroids, res.labels, k)
            chunk = pack_chunk(
                index, centroids_np, labels_np, None,
                np.empty(0, np.uint32), np.empty(0, np.float32),
                n_values=n_values, tuple_size=cfg.tuple_size,
                mode=cfg.mode, level=cfg.zstd_level,
            )
            max_err, mean_err, n_out = vq_max, vq_mean, 0
        else:
            lo, hi, oidx, oval, max_err, mean_err = _quantize_gpu(
                x2d, res.centroids, res.labels, cfg.error_bound, k
            )
            vq_max, vq_mean = res.evaluation.vq_max_error, res.evaluation.vq_mean_abs_error
            chunk = pack_chunk(
                index, centroids_np, labels_np, None, oidx, oval,
                n_values=n_values, tuple_size=cfg.tuple_size,
                mode=cfg.mode, level=cfg.zstd_level, planes=(lo, hi),
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
        values_per_window, cfg.tuple_size, km.DIST_BUFFER_BYTES
    )
    concurrency = plan_concurrency(budget, per_window, cfg.max_workers)

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
        eb = reader.header["error_bound"]
        parts = []
        for i in range(len(reader.chunks)):
            parts.append(unpack_chunk(reader.read_chunk(i), eb))
        values = np.concatenate(parts) if parts else np.empty(0, np.float32)
        header = reader.header
    if out:
        np.save(out, values)
    return values, header


def iter_decompressed(path: str) -> Iterator[np.ndarray]:
    """Stream windows back one at a time (for verification without holding all)."""
    with ContainerReader(path) as reader:
        eb = reader.header["error_bound"]
        for i in range(len(reader.chunks)):
            yield unpack_chunk(reader.read_chunk(i), eb)
