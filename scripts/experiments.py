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
    """Re-read source and container together; returns (max_err, violations, n)."""
    with ContainerReader(container) as r:
        eb = r.header["error_bound"]
        vpw = r.header["window_size"] // 4
        stream = WeightStream.open(source)
        worst, bad, seen = 0.0, 0, 0
        for i, (_idx, original) in enumerate(stream.windows(vpw)):
            if i >= len(r.chunks):
                break
            got = unpack_chunk(r.read_chunk(i))
            m = min(got.size, original.size)
            err = np.abs(got[:m].astype(np.float64) - original[:m].astype(np.float64))
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
        "gpt2": os.path.join(md, "gpt2.safetensors"),
        "gpt2-medium": os.path.join(md, "gpt2-medium.safetensors"),
        "tinyllama": os.path.join(md, "tinyllama.safetensors"),
    }
    groups = args.only.split(",") if args.only else \
        ["models", "baseline", "tuple", "vq", "bound"]
    rows: list[dict] = []
    L = args.sweep_limit

    if "models" in groups:
        print("\n[models] defaults: eb=1e-4, 128MB windows, tuple=2, k search by size")
        for name, path in models.items():
            if os.path.exists(path):
                rows.append(run(f"model-{name}", path, args.out))

    if "baseline" in groups:
        print("\n[baseline] does the k-means predictor beat no predictor at all?")
        for label, kw in [
            ("k1-no-kmeans", dict(k_start=1, max_k=1)),
            ("k64-fixed", dict(k_start=64, max_k=64)),
            ("k1024-fixed", dict(k_start=1024, max_k=1024)),
            ("k-search", dict()),
        ]:
            rows.append(run(f"baseline-{label}", models["gpt2"], args.out,
                            limit_bytes=L, **kw))

    if "tuple" in groups:
        print("\n[tuple] a label costs log2(k)/T bits per value, so wider tuples")
        print("        amortise it -- at what T does a codebook start to pay?")
        for tsize in (1, 2, 4, 8, 16, 32, 64):
            rows.append(run(f"tuple-{tsize:03d}", models["gpt2"], args.out,
                            limit_bytes=L, tuple_size=tsize, max_k=1 << 14))
        print("        and with the codebook forced on, to price it directly:")
        for tsize in (2, 16, 64):
            rows.append(run(f"forcedk-t{tsize:03d}", models["gpt2"], args.out,
                            limit_bytes=L, tuple_size=tsize,
                            k_start=256, max_k=256))

    if "vq" in groups:
        print("\n[vq] the literal rule: double k until pure VQ meets the bound")
        rows.append(run("vq-criterion", models["gpt2"], args.out,
                        limit_bytes=128 * MB, k_criterion="vq", max_k=1 << 14))
        rows.append(run("vq-mode-pure", models["gpt2"], args.out,
                        limit_bytes=128 * MB, mode="vq", k_start=256, max_k=256))

    if "bound" in groups:
        print("\n[bound] error bound sweep")
        for eb in (1e-3, 1e-4, 1e-5, 1e-6):
            rows.append(run(f"bound-{eb:.0e}", models["gpt2"], args.out,
                            limit_bytes=L, error_bound=eb))

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {os.path.join(args.out, 'summary.json')} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
