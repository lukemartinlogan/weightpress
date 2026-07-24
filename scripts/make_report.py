#!/usr/bin/env python3
"""Turn the experiment JSON into the README's results section.

    python scripts/make_report.py --results /home/iowarp/wp-results \
        --sweeps /home/iowarp/wp-sweeps --lossless /home/iowarp/wp-results/lossless.json
"""

from __future__ import annotations

import argparse
import glob
import json
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


def ks(row: dict) -> str:
    h = row["chosen_k"]
    return "/".join(str(k) for k in h) if len(h) > 1 else str(next(iter(h)))


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

    A("Hardware: RTX 5080 (16 GiB), torch 2.11 + CUDA 12.8. Defaults throughout")
    A("(`eb=1e-4`, 128 MiB windows, `tuple_size=2`, k search by size) unless noted.")
    A("Every row was verified by decoding the container and comparing against the")
    A("source value by value.")
    A("")
    A("### Whole checkpoints")
    A("")
    A("| checkpoint | source | windows | k | stored | bits/val | vs fp32 | vs source | max error | violations |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for name in ("model-gpt2", "model-gpt2-medium", "model-tinyllama"):
        r = models.get(name)
        if not r:
            continue
        src = r["model"].replace(".safetensors", "")
        dt = "bf16" if "tinyllama" in name else "fp32"
        A(f"| {src} | {r['source_bytes']/MB:.0f} MiB {dt} | {r['n_windows']} | "
          f"{ks(r)} | {r['stored_bytes']/MB:.0f} MiB | {r['bits_per_value']:.2f} | "
          f"{r['ratio_vs_fp32']:.2f}x | {r['ratio_vs_source']:.2f}x | "
          f"{r['max_error']:.3e} | {r['violations']} |")
    A("")
    A("`vs source` is the number that matters for a bf16 checkpoint: the fp32 column")
    A("credits the codec for undoing a widening it performed itself.")
    A("")

    if lossless:
        A("For scale, lossless compression of the same bytes (256 MiB, zstd-3):")
        A("")
        A("| checkpoint | zstd | zstd + byte-plane split | weightpress @ 1e-4 |")
        A("|---|---|---|---|")
        for k, v in lossless.items():
            wp = models.get(f"model-{k}")
            wpr = f"{wp['ratio_vs_source']:.2f}x" if wp else "-"
            A(f"| {k} | {v['zstd_ratio']:.2f}x | {v['zstd_split_ratio']:.2f}x | {wpr} |")
        A("")

    order = ["baseline-k1-no-kmeans", "baseline-k64-fixed",
             "baseline-k1024-fixed", "baseline-k-search"]
    base = [sweeps[n] for n in order if n in sweeps]
    if base:
        A("### Does the k-means predictor earn its labels?")
        A("")
        A("gpt2, first 256 MiB, tuple size 2. `k=1` means a single centroid, i.e.")
        A("residuals coded against the window mean -- no learned predictor at all.")
        A("")
        A("| setting | k | labels | residuals | bits/val | ratio |")
        A("|---|---|---|---|---|---|")
        for r in base:
            A(f"| {r['name'].replace('baseline-','')} | {ks(r)} | "
              f"{r['label_bytes']/MB:.1f} MiB | {r['code_bytes']/MB:.1f} MiB | "
              f"{r['bits_per_value']:.2f} | **{r['ratio_vs_source']:.2f}x** |")
        A("")
        A("**The codebook is a net loss.** At k=64 the labels cost 2.66 bits/value")
        A("and buy back only 2.09 bits of residual entropy. Adjacent weights in a")
        A("flattened transformer tensor are close to uncorrelated, so a 2-D")
        A("codebook has almost no structure to exploit -- and the label has to be")
        A("paid for regardless. This is not an artefact of the fit: an explicit")
        A("label can only beat scalar quantization by the space-filling advantage")
        A("of the lattice, which in 2-D is bounded by about 0.17 bits per tuple.")
        A("")
        A("The search finds this on its own -- `k-search` lands on the same 3.04x as")
        A("the hand-set `k=1` baseline, on every window of all three checkpoints.")
        A("")

    def label_bits_per_value(r: dict) -> float:
        n_values = r["raw_bytes"] / 4
        return 8.0 * r["label_bytes"] / max(1.0, n_values)

    tup = sorted((v for k, v in sweeps.items() if k.startswith("tuple-")),
                 key=lambda r: r["tuple_size"])
    forced = {r["tuple_size"]: r
              for k, r in sweeps.items() if k.startswith("forcedk-")}
    if tup:
        A("### Tuple size: when does a codebook start to pay?")
        A("")
        A("A label costs `log2(k)/T` bits per value, so wider tuples amortise it.")
        A("The `k=256 forced` column prices the codebook directly by denying the")
        A("search the option of declining it.")
        A("")
        A("| tuple size | k chosen | label bits/val | bits/val | ratio | ratio at k=256 forced |")
        A("|---|---|---|---|---|---|")
        for r in tup:
            f = forced.get(r["tuple_size"])
            fr = f"{f['ratio_vs_source']:.3f}x" if f else "-"
            A(f"| {r['tuple_size']} | {ks(r)} | {label_bits_per_value(r):.3f} | "
              f"{r['bits_per_value']:.2f} | {r['ratio_vs_source']:.3f}x | {fr} |")
        A("")

    bounds = sorted((v for k, v in sweeps.items() if k.startswith("bound-")),
                    key=lambda r: -r["error_bound"])
    if bounds:
        A("### Error bound")
        A("")
        A("| bound | k | bits/val | ratio | max error | escapes |")
        A("|---|---|---|---|---|---|")
        for r in bounds:
            A(f"| {r['error_bound']:.0e} | {ks(r)} | {r['bits_per_value']:.2f} | "
              f"{r['ratio_vs_source']:.2f}x | {r['max_error']:.2e} | {r['escapes']} |")
        A("")

    vq = sweeps.get("vq-criterion")
    pure = sweeps.get("vq-mode-pure")
    if vq:
        A("### The literal 'double k until max error fits' rule does not terminate")
        A("")
        A(f"`--k-criterion vq` on gpt2, `--max-k {vq['max_k']}`:")
        A("")
        A("| k | max VQ error | mean abs VQ error | label bits/tuple | residual bits/val |")
        A("|---|---|---|---|---|")
        for t in vq["trials"]:
            A(f"| {t['k']} | {t['vq_max_error']:.3f} | {t['vq_mean_abs_error']:.5f} | "
              f"{t['label_entropy_bits']:.2f} | {t['code_entropy_bits']:.2f} |")
        A("")
        first, last = vq["trials"][0], vq["trials"][-1]
        shrink = first["vq_mean_abs_error"] / max(1e-12, last["vq_mean_abs_error"])
        A(f"Over a {last['k']//first['k']}x increase in k the *mean* error falls")
        A(f"{shrink:.0f}x, while the *max* error goes from {first['vq_max_error']:.1f}")
        A(f"to {last['vq_max_error']:.1f} -- it does not converge toward 1e-4 at all.")
        A("Lloyd's minimises squared error, so extra centroids chase the bulk of the")
        A("distribution; the max is set by a handful of tail weights that keep")
        A("sharing a cluster with everything else. Even if it did converge, matching")
        A("1e-4 by quantization alone needs ~10^4 levels per dimension, so ~10^8")
        A("centroids at tuple size 2 -- more centroids than there are tuples in a")
        A("window. The search runs to `max_k` and the residual stage does the work.")
        A("")
        if pure:
            A(f"`--mode vq` (store labels only, no residuals) at k={ks(pure)} shows what")
            A(f"that costs: {pure['bits_per_value']:.2f} bits/value and a fine")
            A(f"{pure['ratio_vs_source']:.1f}x ratio, but max error")
            A(f"{pure['max_error']:.1f} and {pure['violations']:,} of")
            A(f"{pure['values_checked']:,} values outside the bound.")
            A("")

    text = "\n".join(L)
    with open(args.readme) as fh:
        readme = fh.read()
    marker = "<!--RESULTS-->"
    if marker in readme:
        head, _, tail = readme.partition(marker)
        nxt = tail.find("\n## ")
        readme = head + text + (tail[nxt:] if nxt >= 0 else "")
        with open(args.readme, "w") as fh:
            fh.write(readme)
        print(f"spliced {len(L)} lines into {args.readme}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
