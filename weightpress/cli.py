"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .config import Config


def _parse_size(text: str) -> int:
    """Accept ``128MB``, ``2G``, ``1048576`` ..."""
    s = str(text).strip().upper().replace("IB", "B")
    mult = 1
    for suffix, m in (("KB", 1 << 10), ("MB", 1 << 20), ("GB", 1 << 30), ("TB", 1 << 40)):
        if s.endswith(suffix):
            s, mult = s[: -len(suffix)], m
            break
    else:
        for suffix, m in (("K", 1 << 10), ("M", 1 << 20), ("G", 1 << 30), ("T", 1 << 40)):
            if s.endswith(suffix):
                s, mult = s[: -len(suffix)], m
                break
    return int(float(s) * mult)


def _human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} TiB"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-e", "--error-bound", type=float, default=1e-4,
                   help="absolute error bound per weight (default: 1e-4)")
    p.add_argument("-w", "--window-size", type=_parse_size, default="128MB",
                   help="window size in bytes of the fp32 working stream (default: 128MB)")
    p.add_argument("-t", "--tuple-size", type=int, default=2,
                   help="weights per k-means vector (default: 2)")
    p.add_argument("-g", "--max-gpu-memory", type=_parse_size, default=None,
                   help="GPU memory budget (default: 80%% of free memory)")
    p.add_argument("-o", "--output-dir", default=".",
                   help="where containers and k-means tables are written (default: cwd)")
    p.add_argument("--mode", choices=["residual", "vq"], default="residual",
                   help="residual: hard error bound via quantized residuals; "
                        "vq: pure vector quantization, bound met only by growing k")
    p.add_argument("--k-start", type=int, default=64)
    p.add_argument("--max-k", type=int, default=1 << 16)
    p.add_argument("--k-criterion", choices=["size", "vq"], default="size",
                   help="size: stop doubling when it no longer shrinks the payload; "
                        "vq: stop when pure-VQ max error meets the bound")
    p.add_argument("--min-k-gain", type=float, default=0.02)
    p.add_argument("--k-patience", type=int, default=2,
                   help="non-improving doublings to try before settling on k")
    p.add_argument("--kmeans-iters", type=int, default=25)
    p.add_argument("--zstd-level", type=int, default=3)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="float32", help="element type for raw .bin inputs")
    p.add_argument("--limit-bytes", type=_parse_size, default=None,
                   help="only process the first N bytes of the stream (for quick runs)")
    p.add_argument("--json", dest="json_out", default=None,
                   help="write the full per-chunk report to this JSON file")


def _config_from(args: argparse.Namespace) -> Config:
    return Config(
        error_bound=args.error_bound,
        window_size=args.window_size,
        tuple_size=args.tuple_size,
        max_gpu_memory=args.max_gpu_memory,
        output_dir=args.output_dir,
        k_start=args.k_start,
        max_k=args.max_k,
        k_criterion=args.k_criterion,
        min_k_gain=args.min_k_gain,
        k_patience=args.k_patience,
        kmeans_iters=args.kmeans_iters,
        mode=args.mode,
        zstd_level=args.zstd_level,
        max_workers=args.max_workers,
        device=args.device,
        seed=args.seed,
    )


def _report(run, cfg: Config, out_path: str) -> None:
    print()
    print(f"  container      {out_path}")
    print(f"  k-means tables {os.path.splitext(out_path)[0]}.kmeans/ "
          f"({len(run.chunks)} chunk tables)")
    print(f"  windows        {len(run.chunks)}  x  {_human(cfg.window_size)}")
    print(f"  gpu budget     {_human(run.gpu_budget_bytes)}"
          f"   ({run.concurrency} windows in flight)")
    print(f"  chosen k       {run.k_histogram()}")
    print(f"  raw            {_human(run.raw_bytes)}")
    print(f"  stored         {_human(run.stored_bytes)}")
    print(f"  ratio          {run.ratio:.3f}x   ({run.bits_per_value:.2f} bits/value)")
    print(f"  max error      {run.max_error:.3e}   (bound {cfg.error_bound:.3e})")
    if run.chunks:
        cb = sum(c.codebook_bytes for c in run.chunks)
        lb = sum(c.label_bytes for c in run.chunks)
        qb = sum(c.code_bytes for c in run.chunks)
        ob = sum(c.outlier_bytes for c in run.chunks)
        tot = max(1, cb + lb + qb + ob)
        print(f"  breakdown      codebook {100*cb/tot:5.2f}%  labels {100*lb/tot:5.2f}%"
              f"  residuals {100*qb/tot:5.2f}%  escapes {100*ob/tot:5.2f}%")
        print(f"  escapes        {sum(c.n_outliers for c in run.chunks)} values")
    thru = run.raw_bytes / max(1e-9, run.seconds) / (1 << 20)
    print(f"  time           {run.seconds:.2f}s   ({thru:.1f} MiB/s)")
    ok = run.max_error <= cfg.error_bound
    if cfg.mode == "residual":
        print(f"  error bound    {'HELD' if ok else 'VIOLATED'}")


def cmd_compress(args: argparse.Namespace) -> int:
    from .pipeline import compress

    cfg = _config_from(args)
    done = [0]

    def progress(st) -> None:
        done[0] += 1
        print(f"  [chunk {st.index:4d}] k={st.k:<6d} "
              f"ratio={st.ratio:6.3f}x  max_err={st.max_error:.2e}  "
              f"vq_err={st.vq_max_error:.2e}  {st.seconds:5.2f}s", flush=True)

    print(f"compressing {args.source}")
    out_path, run = compress(
        args.source, cfg, dtype=args.dtype, limit_bytes=args.limit_bytes,
        progress=progress,
    )
    _report(run, cfg, out_path)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(run.to_dict(), fh, indent=2)
        print(f"  report         {args.json_out}")
    return 0 if (cfg.mode != "residual" or run.max_error <= cfg.error_bound) else 1


def cmd_decompress(args: argparse.Namespace) -> int:
    from .pipeline import decompress

    t0 = time.time()
    values, header = decompress(args.container, out=args.out)
    print(f"decompressed {values.size} values in {time.time()-t0:.2f}s")
    if args.out:
        print(f"  wrote {args.out}")
    else:
        print(json.dumps({k: v for k, v in header.items() if k != "chunks"}, indent=2)[:2000])
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Re-read the source and the container side by side and check the bound."""
    from .codec import unpack_chunk
    from .container import ContainerReader
    from .reader import WeightStream

    with ContainerReader(args.container) as reader:
        header = reader.header
        eb = header["error_bound"]
        src_path = args.source or header["source"]["path"]
        stream = WeightStream.open(src_path, dtype=args.dtype)
        vpw = header["window_size"] // 4

        worst = 0.0
        n_seen = 0
        n_over = 0
        n_windows = 0
        for i, (_index, original) in enumerate(stream.windows(vpw)):
            if i >= len(reader.chunks):
                break
            got = unpack_chunk(reader.read_chunk(i))
            m = min(got.size, original.size)
            err = np.abs(got[:m].astype(np.float64) - original[:m].astype(np.float64))
            worst = max(worst, float(err.max()) if m else 0.0)
            # Strict: the bound is a hard contract, so no tolerance is allowed.
            n_over += int((err > eb).sum())
            n_seen += m
            n_windows += 1

    print(f"  checked        {n_seen} values across {n_windows} windows")
    print(f"  bound          {eb:.3e}")
    print(f"  max error      {worst:.6e}")
    print(f"  violations     {n_over}")
    ok = n_over == 0
    print(f"  result         {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="weightpress",
        description="Learned error-bounded lossy compression for model weights.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress", help="compress a checkpoint")
    c.add_argument("source", help=".safetensors, .npy or raw binary")
    _add_common(c)
    c.set_defaults(func=cmd_compress)

    d = sub.add_parser("decompress", help="rebuild the fp32 stream")
    d.add_argument("container")
    d.add_argument("-o", "--out", default=None, help="write values to this .npy")
    d.set_defaults(func=cmd_decompress)

    v = sub.add_parser("verify", help="check the error bound against the source")
    v.add_argument("container")
    v.add_argument("-s", "--source", default=None)
    v.add_argument("--dtype", default="float32")
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
