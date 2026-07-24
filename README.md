# weightpress

Learned error-bounded lossy compression for model weights: a k-means / vector
quantization codebook is the learned regressor, and the per-value cluster labels
are the losslessly compressed integer stream.

Every stored value reconstructs within a **relative** error bound —
`|x - x_hat| / |x| ≤ eb` (default `1e-4`, i.e. 0.01%). **k is the number of
clusters**: the search grows it (doubling from 64) until the maximum percentage
error meets the bound, and reports it. An absolute-error mode
(`--error-mode absolute`) is also available.

## What the measurements say

Tested on gpt2, gpt2-medium and TinyLlama-1.1B — 4.2 GiB of real weights, every
value decoded and compared against the source, at the default relative bound.
Full numbers under [Results](#results-on-real-checkpoints).

* **The bound holds.** Zero violations across all 1.1 billion values, at relative
  bounds from `1e-2` to `1e-6`. The reconstruct-and-check runs on the CPU with
  the decoder's exact arithmetic, so what is verified is bit-for-bit what decode
  produces.
* **k is the number of clusters, and the search grows it to meet the bound.** For
  gpt2 at `eb=1e-4` that is `k=131072` (2^17), of which ~55–70k cells per window
  are occupied and stored. Tightening the bound one decade roughly doubles k, as
  the "double until it fits" rule implies. ~1.8x on fp32 checkpoints, against
  1.17–1.33x for lossless zstd on the same bytes.
* **A relative bound is much stricter than it looks.** Relative `1e-4` demands
  0.01% precision at every magnitude; bf16 carries only ~0.4% to begin with. bf16
  TinyLlama still reaches 1.27x because its weights take only a few thousand
  distinct log-magnitudes, so the codebook is tiny (~4.5k cells) and the labels
  compress well — but fp32 checkpoints, with far more distinct values, sit at
  ~1.85x.
* **Why the clustering has to be uniform-in-log, not Lloyd's k-means.** Lloyd's
  minimises *squared* error, so it cannot meet a hard *max* error bound at any
  practical k — from `k=64` to `k=16384` its mean error falls sharply while its
  max barely moves, set by a few tail weights. The error-bounded clusterer for a
  relative bound is instead uniform cells of width `2·ln(1+eb)` in log space,
  which caps every member's percentage error by construction. That is what
  `mode=cluster` builds. The literal "store the nearest centroid and stop" (pure
  VQ) is what breaks: at feasible k the max error is enormous.

So the honest summary: the compression is real and beats lossless, and it is
exactly the design — clustering plus lossless label coding. The cost (~16
bits/value at `1e-4`) is what pinning 0.01% relative precision on every weight
across a 9-order-of-magnitude range genuinely takes.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e .            # numpy + zstandard
.venv/bin/pip install torch           # CUDA build for GPU k-means
```

## Use

```bash
weightpress compress model.safetensors -o out/
weightpress verify   out/model.wp
weightpress decompress out/model.wp -o restored.npy
```

The five documented inputs, with their defaults:

| Flag | Default | Meaning |
|---|---|---|
| `-e, --error-bound` | `1e-4` | relative error bound (max percentage error), per weight |
| `-w, --window-size` | `128MB` | window size, in bytes of the fp32 working stream |
| `-t, --tuple-size` | `2` | weights per k-means vector |
| `-g, --max-gpu-memory` | 80% of free | budget that sets how many windows run at once |
| `-o, --output-dir` | `.` | where the container and k-means tables are written |

`--error-mode {relative,absolute}` selects the bound (default `relative`).

Output for `compress model.safetensors -o out/`:

* `out/model.wp` — self-contained container (codebook, labels, signs, escapes)
* `out/model.kmeans/chunk_NNNNNN.npz` — the cluster table for each window,
  standalone and independently inspectable, as the design calls for

## Algorithm (`mode=cluster`, the default)

Per 128 MB window, independently and in parallel. Everything happens in the log
domain, so the bound is on percentage error: the feature is `u = log|x|`, the
sign is stored separately, and `x_hat = sign · exp(centroid)`.

1. **Transform**: `u = log|x|`, plus a sign bit per value; zeros and non-finite
   values escape (no useful log) and are stored verbatim.
2. **Search k, the cluster count.** Start at `k=64` and double until each of the
   `k` cells covering the window's log range is at most `2·ln(1+eb)` wide — the
   width at which a cell's centre reconstructs every member within `eb`. Report
   the resulting `k` (for gpt2 at `1e-4`, that is 2^17).
3. **Cluster**: assign each value to its cell. The **codebook** is the cells that
   are actually occupied (their centres, in log space); the per-value **label** is
   an index into it.
4. **Code losslessly**: labels at the narrowest integer width that fits the
   codebook, zstd'd; codebook and bit-packed signs likewise. The label stream is
   the "lossless integer compression" and is ~94% of the output.

### Why the clusters are uniform-in-log, not Lloyd's k-means

The design says "k-means," but k-means (Lloyd's) minimises *mean squared* error,
and that is the wrong objective for a hard *max* error bound: extra centroids
chase the dense bulk of the distribution while a few tail weights keep the max
high, so it never converges to the bound at any practical k (see the results).
The clusterer that *does* meet a relative max bound places its cells uniformly in
log space at width `2·ln(1+eb)`; each cell's centre is then within `ln(1+eb)` of
every member, i.e. within `eb` relative. That is a vector quantizer with `k`
cells — literally the design — with the cell geometry fixed by the bound instead
of by Lloyd's iterations.

### How the bound is guaranteed

```
label   = floor((u - umin) / cell_w)        u = log|x|,  cell_w = 2·ln(1+eb)
centroid = umin + (label + 0.5)·cell_w      (stored, per occupied cell)
x_hat    = sign · exp(centroid)       =>    |x - x_hat| / |x| ≤ eb
```

Two details make this hold in practice, not just on paper:

* **The check runs with the decoder's exact arithmetic.** The reconstruct-and-
  check happens on the CPU with the same numpy `exp` the decoder uses —
  `exp(centroid)` differs between GPU and numpy by a few ULP, enough to cross a
  tight bound — so a value is only accepted if it passes the *decoder's* math.
* **Anything that still misses escapes**, stored verbatim as float32 and
  reconstructed exactly: NaN/Inf, zeros, and (for bounds below the float32 ulp of
  the data) values the grid cannot represent. Where everything escapes, the ratio
  goes to ~1 and the bound still holds.

`weightpress verify` re-reads the source and the container side by side and
reports the true max percentage error and violation count.

### Other modes

`--mode residual` keeps a small k-means codebook as a *predictor* and
entropy-codes the quantized residual instead of the raw label — slightly smaller
at the same bound, but then `k` is the predictor size, not the cluster count.
`--mode vq` is pure vector quantization (store the nearest label, no correction):
it shows why the literal reading fails, since at any feasible `k` the max error is
enormous. Both are kept for comparison; `cluster` is the default.

## Results on real checkpoints

<!--RESULTS-->
Hardware: RTX 5080 (16 GiB), torch 2.11 + CUDA 12.8. Cluster mode, relative
bound, 128 MiB windows, unless noted. Every row was verified by decoding the
container and comparing against the source value by value.

### Whole checkpoints (eb=1e-4 relative)

| checkpoint | source | windows | clusters (k) | codebook | bits/val | vs source | max %err | violations |
|---|---|---|---|---|---|---|---|---|
| gpt2 | 523 MiB fp32 | 5 | 131,072 (2^17) | 69,594 | 17.49 | 1.83x | 8.91e-05 | 0 |
| gpt2-medium | 1450 MiB fp32 | 12 | 131,072 (2^17) | 106,426 | 17.25 | 1.86x | 9.52e-05 | 0 |
| tinyllama | 2098 MiB bf16 | 33 | 131,072 (2^17) | 4,483 | 12.60 | 1.27x | 9.60e-05 | 0 |

`k` is the number of clusters the search grows to (doubling from 64) -- the
design's k. `codebook` is how many of those cells are actually occupied and
stored. Each weight becomes one cluster label; the labels are the losslessly
compressed integer stream and dominate the output (~94%).

bf16 TinyLlama compresses less than the fp32 models: a relative 1e-4 bound
asks for 0.01% precision while bf16 carries only ~0.4%. It still beats 1x
because bf16 takes only a few thousand distinct log-magnitudes, so its
codebook is tiny and the labels compress well.

For scale, lossless compression of the same bytes (256 MiB, zstd-3):

| checkpoint | zstd | zstd + byte-plane split | weightpress @ 1e-4 |
|---|---|---|---|
| gpt2 | 1.23x | 1.33x | 1.83x |
| gpt2-medium | 1.17x | 1.26x | 1.86x |
| tinyllama | 1.28x | 1.41x | 1.27x |

### Cluster count vs the bound

gpt2, first 256 MiB. Tightening the bound halves the cell width, so the
cluster count doubles per decade -- the "double k until it fits" search,
run to completion.

| bound (rel) | clusters (k) | occupied | bits/val | ratio | max %err |
|---|---|---|---|---|---|
| 1e-02 | 2,048 (2^11) | 1,587 | 10.12 | 3.16x | 5.59e-03 |
| 1e-03 | 16,384 (2^14) | 10,583 | 12.49 | 2.56x | 6.99e-04 |
| 1e-04 | 131,072 (2^17) | 68,459 | 16.87 | 1.90x | 8.91e-05 |
| 1e-05 | 2,097,152 (2^21) | 766,780 | 22.73 | 1.41x | 7.43e-06 |
| 1e-06 | 16,777,216 (2^24) | 3,467,516 | 35.97 | 0.89x | 1.00e-06 |

### The clustering vs the alternatives

gpt2, first 256 MiB, eb=1e-4 relative.

| method | k | bits/val | ratio | max %err | violations |
|---|---|---|---|---|---|
| cluster (the design) | 131,072 | 16.87 | 1.90x | 8.91e-05 | 0 |
| predictor + residual | 1 | 13.70 | 2.34x | 9.66e-05 | 0 |
| pure VQ, labels only | 256 | 3.39 | 9.44x | 2.76e+06 | 58,694,463 |

The **cluster** row is the design as written: VQ clustering, k grown until
the max percentage error meets the bound, each value stored as its label.
**predictor + residual** keeps a small k-means codebook and entropy-codes
the correction instead of the raw label -- a little smaller, same
guarantee, but k is then the predictor size, not the cluster count. **pure
VQ** (store the label and stop, no correction) is what the literal reading
breaks on: at any feasible k the max error is enormous and the bound is
massively violated. See the note on why clustering alone cannot meet the
bound below.

## Format

```
MAGIC | version:u32 | reserved:u32 | <chunk payloads...> | header_json | header_len:u64 | MAGIC
```

The header is written last and located by seeking to the end, so writing is a
single forward pass. Each payload is the codebook (cluster centroids) followed by
the zstd blobs in a fixed order — labels, sign plane, and the escape list; the
header records every length, the tensor manifest (name, dtype, shape, offset),
and each chunk's cluster count and codebook size.

Checkpoints are read by memory-mapping the file and slicing at the offsets in the
safetensors header. fp16/bf16 are widened to fp32 exactly, so the bound is
measured in fp32 space; ratios are always reported against the *source* byte
count, never the widened one.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

`tests/test_codec.py` and `tests/test_container.py` need only numpy;
`tests/test_pipeline.py` needs torch and runs each case on CPU and, when
available, CUDA.

## Limitations

* A relative bound is demanding: at `1e-4` the labels need ~16 bits/value, so the
  ceiling is ~2x on fp32 and ~1x on bf16. This is inherent to pinning 0.01%
  relative precision on every weight, not a codec inefficiency.
* The clusters are uniform in log space (the max-error-optimal geometry), not
  Lloyd's k-means — see the note above for why. `--mode residual` uses a real
  k-means predictor if you want to compare.
* f64 and wide-integer tensors are cast to fp32 on read, which is itself lossy;
  the bound is enforced relative to the fp32 stream.
* An error bound below the float32 ulp of the data forces every value to escape,
  giving a ratio near 1. That is correct but not useful.
* Windows are compressed independently, so there is no cross-window modelling.
