#!/usr/bin/env python3
"""Turn the experiment JSON into the README's results section.

    python scripts/make_report.py --results RESULTS --sweeps SWEEPS \
        --lossless LOSSLESS.json --readme README.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

MB = 1 << 20


def load(d: str) -> dict[str, dict]:
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(p) in ("summary.json", "lossless.json"):
            continue
        with open(p) as fh:
            blob = json.load(fh)
        s = blob["summary"] if "summary" in blob else blob
        out[s["name"]] = s
    return out


def pow2(k: int) -> str:
    return f"2^{int(round(math.log2(k)))}" if k > 1 else str(k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/home/iowarp/wp-results")
    ap.add_argument("--sweeps", default="/home/iowarp/wp-sweeps")
    ap.add_argument("--lossless", default="/home/iowarp/wp-results/lossless.json")
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()

    models = load(args.results)
    sweeps = load(args.sweeps)
    lossless = {}
    if os.path.exists(args.lossless):
        with open(args.lossless) as fh:
            lossless = json.load(fh)

    L: list[str] = []
    A = L.append

    A("Hardware: RTX 5080 (16 GiB), torch 2.11 + CUDA 12.8. Cluster mode, relative")
    A("bound, 128 MiB windows, unless noted. Every row was verified by decoding the")
    A("container and comparing against the source value by value.")
    A("")
    A("### Whole checkpoints (eb=1e-4 relative)")
    A("")
    A("| checkpoint | source | windows | clusters (k) | codebook | bits/val | vs source | max %err | violations |")
    A("|---|---|---|---|---|---|---|---|---|")
    for name in ("model-gpt2", "model-gpt2-medium", "model-tinyllama"):
        r = models.get(name)
        if not r:
            continue
        src = r["model"].replace(".safetensors", "")
        dt = "bf16" if "tinyllama" in name else "fp32"
        k = int(next(iter(r["chosen_k"])))
        A(f"| {src} | {r['source_bytes']/MB:.0f} MiB {dt} | {r['n_windows']} | "
          f"{k:,} ({pow2(k)}) | {r['occupied_clusters']:,} | "
          f"{r['bits_per_value']:.2f} | {r['ratio_vs_source']:.2f}x | "
          f"{r['max_error']:.2e} | {r['violations']} |")
    A("")
    A("`k` is the number of clusters the search grows to (doubling from 64) -- the")
    A("design's k. `codebook` is how many of those cells are actually occupied and")
    A("stored. Each weight becomes one cluster label; the labels are the losslessly")
    A("compressed integer stream and dominate the output (~94%).")
    A("")
    A("bf16 TinyLlama compresses less than the fp32 models: a relative 1e-4 bound")
    A("asks for 0.01% precision while bf16 carries only ~0.4%. It still beats 1x")
    A("because bf16 takes only a few thousand distinct log-magnitudes, so its")
    A("codebook is tiny and the labels compress well.")
    A("")

    if lossless:
        A("For scale, lossless compression of the same bytes (256 MiB, zstd-3):")
        A("")
        A("| checkpoint | zstd | zstd + byte-plane split | weightpress @ 1e-4 |")
        A("|---|---|---|---|")
        for kk, v in lossless.items():
            wp = models.get(f"model-{kk}")
            wpr = f"{wp['ratio_vs_source']:.2f}x" if wp else "-"
            A(f"| {kk} | {v['zstd_ratio']:.2f}x | {v['zstd_split_ratio']:.2f}x | {wpr} |")
        A("")

    bounds = sorted((v for k, v in sweeps.items() if k.startswith("bound-")),
                    key=lambda r: -r["error_bound"])
    if bounds:
        A("### Cluster count vs the bound")
        A("")
        A("gpt2, first 256 MiB. Tightening the bound halves the cell width, so the")
        A('cluster count doubles per decade -- the "double k until it fits" search,')
        A("run to completion.")
        A("")
        A("| bound (rel) | clusters (k) | occupied | bits/val | ratio | max %err |")
        A("|---|---|---|---|---|---|")
        for r in bounds:
            k = int(next(iter(r["chosen_k"])))
            A(f"| {r['error_bound']:.0e} | {k:,} ({pow2(k)}) | "
              f"{r['occupied_clusters']:,} | {r['bits_per_value']:.2f} | "
              f"{r['ratio_vs_source']:.2f}x | {r['max_error']:.2e} |")
        A("")

    methods = [sweeps.get(n) for n in
               ("method-cluster", "method-residual", "method-vq-pure")]
    methods = [m for m in methods if m]
    if methods:
        A("### The clustering vs the alternatives")
        A("")
        A("gpt2, first 256 MiB, eb=1e-4 relative.")
        A("")
        A("| method | k | bits/val | ratio | max %err | violations |")
        A("|---|---|---|---|---|---|")
        names = {"method-cluster": "cluster (the design)",
                 "method-residual": "predictor + residual",
                 "method-vq-pure": "pure VQ, labels only"}
        for m in methods:
            k = int(next(iter(m["chosen_k"])))
            A(f"| {names.get(m['name'], m['name'])} | {k:,} | "
              f"{m['bits_per_value']:.2f} | {m['ratio_vs_source']:.2f}x | "
              f"{m['max_error']:.2e} | {m['violations']:,} |")
        A("")
        A("The **cluster** row is the design as written: VQ clustering, k grown until")
        A("the max percentage error meets the bound, each value stored as its label.")
        A("**predictor + residual** keeps a small k-means codebook and entropy-codes")
        A("the correction instead of the raw label -- a little smaller, same")
        A("guarantee, but k is then the predictor size, not the cluster count. **pure")
        A("VQ** (store the label and stop, no correction) is what the literal reading")
        A("breaks on: at any feasible k the max error is enormous and the bound is")
        A("massively violated. See the note on why clustering alone cannot meet the")
        A("bound below.")
        A("")

    marker = "<!--RESULTS-->"
    text = marker + "\n" + "\n".join(L)
    with open(args.readme) as fh:
        readme = fh.read()
    if marker not in readme:
        raise SystemExit(f"{args.readme}: no {marker}")
    head, _, tail = readme.partition(marker)
    nxt = tail.find("\n## ")
    if nxt < 0:
        raise SystemExit(f"{args.readme}: no section after {marker}")
    with open(args.readme, "w") as fh:
        fh.write(head + text + tail[nxt:])
    print(f"spliced {len(L)} lines into {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
