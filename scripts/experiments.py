#!/usr/bin/env python3
"""Run the weightpress evaluation matrix over real checkpoints.

Emits one JSON file per run plus a summary table.  Every run also verifies the
error bound against the source, so the reported ratios are only ever quoted for
bitstreams that actually decode within the bound.

    python scripts/experiments.py --models-dir /home/iowarp/wp-models --out results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weightpress.codec import unpack_chunk
from weightpress.config import Config
from weightpress.container import ContainerReader
from weightpress.pipeline import compress
from weightpress.reader import WeightStream

MB = 1 << 20


def verify(container: str, source: str) -> tuple[float, int, int]:
    """Re-read source and container together; returns (max_err, violations, n).

    Error is measured relative when the container declares a relative bound.
    """
    with ContainerReader(container) as r:
        eb = r.header["error_bound"]
        relative = r.header.get("error_mode", "absolute") == "relative"
        vpw = r.header["window_size"] // 4
        stream = WeightStream.open(source)
        worst, bad, seen = 0.0, 0, 0
        for i, (_idx, original) in enumerate(stream.windows(vpw)):
            if i >= len(r.chunks):
                break
            got = unpack_chunk(r.read_chunk(i))
            m = min(got.size, original.size)
            o = original[:m].astype(np.float64)
            err = np.abs(got[:m].astype(np.float64) - o)
            if relative:
                err = np.divide(err, np.abs(o), out=np.zeros_like(err), where=o != 0)
            worst = max(worst, float(err.max()) if m else 0.0)
            bad += int((err > eb).sum())
            seen += m
    return worst, bad, seen


def run(name: str, source: str, out_dir: str, *, limit_bytes=None, **kw) -> dict:
    cfg = Config(output_dir=os.path.join(out_dir, name), **kw)
    os.makedirs(cfg.output_dir, exist_ok=True)
    t0 = time.time()
    container, stats = compress(source, cfg, limit_bytes=limit_bytes)
    wall = time.time() - t0
    worst, bad, seen = verify(container, source)

    row = {
        "name": name,
        "model": os.path.basename(source),
        "error_bound": cfg.error_bound,
        "tuple_size": cfg.tuple_size,
        "window_mb": cfg.window_size // MB,
        "k_criterion": cfg.k_criterion,
        "k_start": cfg.k_start,
        "max_k": cfg.max_k,
        "mode": cfg.mode,
        "chosen_k": stats.k_histogram(),
        "n_windows": len(stats.chunks),
        "raw_bytes": stats.raw_bytes,
        "source_bytes": None,
        "stored_bytes": stats.stored_bytes,
        "ratio_vs_fp32": stats.ratio,
        "bits_per_value": stats.bits_per_value,
        "vq_max_error": max((c.vq_max_error for c in stats.chunks), default=float("nan")),
        "max_error": worst,
        "violations": bad,
        "values_checked": seen,
        "escapes": sum(c.n_outliers for c in stats.chunks),
        "occupied_clusters": max((c.occupied_clusters for c in stats.chunks), default=0),
        "label_bytes": sum(c.label_bytes for c in stats.chunks),
        "code_bytes": sum(c.code_bytes for c in stats.chunks),
        "codebook_bytes": sum(c.codebook_bytes for c in stats.chunks),
        "outlier_bytes": sum(c.outlier_bytes for c in stats.chunks),
        "seconds": wall,
        "trials": stats.chunks[0].k_trials if stats.chunks else [],
    }
    src = WeightStream.open(source)
    scale = min(1.0, row["raw_bytes"] / max(1, src.total_values * 4))
    row["source_bytes"] = int(src.source_bytes * scale)
    row["ratio_vs_source"] = row["source_bytes"] / max(1, row["stored_bytes"])

    with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
        json.dump({"summary": row, "full": stats.to_dict()}, fh, indent=2)
    print(
        f"  {name:34s} k={row['chosen_k']!s:14s} "
        f"{row['bits_per_value']:6.2f} b/val  {row['ratio_vs_source']:5.2f}x  "
        f"maxerr={worst:.3e}  viol={bad}  {wall:6.1f}s",
        flush=True,
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="/home/iowarp/wp-models")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sweep-limit", type=int, default=256 * MB,
                    help="bytes of the stream used for parameter sweeps")
    ap.add_argument("--only", default=None, help="comma-separated experiment groups")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    md = args.models_dir
    models = {
        "gemma-4-E2B": os.path.join(md, "gemma-4-E2B", "model.safetensors"),
        "gpt2": os.path.join(md, "gpt2.safetensors"),
        "gpt2-medium": os.path.join(md, "gpt2-medium.safetensors"),
        "tinyllama": os.path.join(md, "tinyllama.safetensors"),
    }
    groups = args.only.split(",") if args.only else \
        ["models", "bound", "compare"]
    rows: list[dict] = []
    L = args.sweep_limit

    if "models" in groups:
        print("\n[models] cluster mode, eb=1e-4 relative, 128MB windows -- k is the "
              "number of clusters the search grows to")
        for name, path in models.items():
            if os.path.exists(path):
                rows.append(run(f"model-{name}", path, args.out))

    if "bound" in groups:
        print("\n[bound] cluster count k vs the relative error bound (gpt2)")
        for eb in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            rows.append(run(f"bound-{eb:.0e}", models["gpt2"], args.out,
                            limit_bytes=L, error_bound=eb))

    if "compare" in groups:
        print("\n[compare] the design's clustering vs the residual/vq variants (gpt2)")
        rows.append(run("method-cluster", models["gpt2"], args.out, limit_bytes=L,
                        mode="cluster"))
        rows.append(run("method-residual", models["gpt2"], args.out, limit_bytes=L,
                        mode="residual"))
        rows.append(run("method-vq-pure", models["gpt2"], args.out, limit_bytes=L,
                        mode="vq", k_start=256, max_k=256))

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {os.path.join(args.out, 'summary.json')} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
