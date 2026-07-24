# weightpress

Learned error-bounded lossy compression for model weights: GPU k-means as the
predictor, quantized residuals as the error-bound enforcer, zstd as the lossless
integer back end.

Every stored value is guaranteed to reconstruct within a **relative** error
bound — `|x - x_hat| / |x| ≤ eb` (default `1e-4`, i.e. 0.01%). The k search
minimises the maximum percentage error. The guarantee is structural, not
statistical — see [How the bound is guaranteed](#how-the-bound-is-guaranteed).
An absolute-error mode (`--error-mode absolute`) is also available.

## What the measurements say

Tested on gpt2, gpt2-medium and TinyLlama-1.1B — 4.2 GiB of real weights, every
value decoded and compared against the source, at the default **relative** bound.
Full numbers under [Results](#results-on-real-checkpoints).

* **The bound holds.** Zero violations across all 1.1 billion values, at relative
  bounds from `1e-3` to `1e-6`. The reconstruct-and-check runs on the CPU with
  the decoder's exact arithmetic, so what is verified is bit-for-bit what decode
  produces.
* **~2.3x on fp32 checkpoints** at `eb=1e-4` relative, against 1.17–1.33x for
  lossless zstd on the same bytes.
* **A relative bound is much stricter than it looks — and a bf16 checkpoint can
  barely be compressed under it.** Relative `1e-4` demands 0.01% precision at
  every magnitude; bf16 only carries ~0.4% relative precision to begin with, so
  reproducing it that finely needs nearly its full 16 bits. TinyLlama lands at
  1.04x — honest, not a bug. fp32 checkpoints, which carry more real precision,
  compress ~2.3x.
* **The k-means codebook does not pay for itself, and the search says so.** In
  the log domain too, a `k=64` label costs more than the sharper prediction
  saves, so `k=1` — no predictor at all — wins. The search picks `k=1` on every
  window of every checkpoint tested. This is information-theoretic, not a weak
  fit: an explicit label can only beat scalar quantization by the lattice's
  space-filling gain, ~0.17 bits per 2-D tuple.
* **The literal "double k until the max error fits" rule cannot terminate.**
  Lloyd's minimises squared error, so from `k=64` to `k=16384` the mean error
  falls sharply while the max error barely moves and never approaches the bound —
  it is set by a few tail weights. All the error bounding comes from the residual
  stage.

So the honest summary: the compression is real and beats lossless, but it comes
from error-bounded residual quantization plus entropy coding. The learned
codebook is the part that does not earn its keep here, and the tool is built to
measure and report that rather than assume it.

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

* `out/model.wp` — self-contained container (centroids, labels, residuals)
* `out/model.kmeans/chunk_NNNNNN.npz` — the k-means table for each window,
  standalone and independently inspectable, as the design calls for

## Algorithm

Per 128 MB window, independently and in parallel. In the default **relative**
mode everything below happens in the log domain: the feature is `u = log|x|`,
the sign is stored separately, and `x_hat = sign · exp(u_hat)`. An absolute step
in log space is a relative step in linear space, so a log-domain residual bound
of `ln(1+eb)` gives a linear relative error of `eb`.

1. **Transform** (relative mode): `u = log|x|`, plus a sign bit per value; zeros
   and non-finite values escape (no useful log).
2. **Tuple** the stream into `n/T` vectors of `T` adjacent values (`T=2`).
3. **Fit** k-means on the GPU. Lloyd iterations run on a random subsample so the
   cost scales with `k` rather than with the window; assignment then sweeps every
   tuple. Empty clusters are reseeded onto the currently worst-fit points.
4. **Search k** over powers of two starting at 64 (below). The "max error" it
   minimises is max log error, i.e. max **percentage** error.
5. **Predict** each value with its centroid and **quantize the residual** onto a
   grid just under `2·ln(1+eb)` (relative) or `2·eb` (absolute) wide.
6. **Code losslessly**: labels at the narrowest integer width that fits `k`;
   residuals zigzagged, split into byte planes, and zstd'd; signs bit-packed.

### How the bound is guaranteed

The residual stage is what enforces the bound, not k-means. In log space:

```
q     = round((u - centroid) / step)        u = log|x|,  step ≈ 2·ln(1+eb)
u_hat = centroid + q·step
x_hat = sign · exp(u_hat)              =>   |x - x_hat| / |x| ≤ eb
```

Three details make this hold in practice rather than only on paper:

* **The grid is slightly finer than the nominal step.** A value on the bound
  would otherwise be pushed past it by float32 rounding. The margin has two
  terms — one proportional to `eb`, one to the window's magnitude, since float32
  rounding scales with the value — the second capped so one huge outlier cannot
  narrow the grid for everyone. The step is data-dependent, so it is **stored
  per chunk** rather than re-derived on read.
* **The check runs with the decoder's exact arithmetic.** In relative mode the
  reconstruct-and-check happens on the CPU with the same numpy `exp` the decoder
  uses — `exp(u_hat)` differs between GPU and numpy by a few ULP, enough to cross
  a tight bound, so a value is only accepted if it passes the *decoder's* math.
  Everything else escapes.
* **Anything that still misses escapes**, stored verbatim as float32 and
  reconstructed exactly. Code `0` is reserved as the escape marker. This covers
  NaN/Inf, residuals wider than the chosen code, and error bounds below the
  float32 ulp of the data — where every value escapes, the ratio goes to ~1, and
  the bound still holds.

The code width is not fixed. Widening the code and escaping the overflow are two
ways of paying for the same outliers, so the width is chosen per window from the
estimated cost of each: the entropy of the extra byte plane against the escape
cost of the values that would not fit. With a fixed 16-bit code, `eb=1e-5` on
gpt2 escaped 4.2M values (a third of the output); choosing by cost it escapes 1.

`weightpress verify` re-reads the source and the container side by side and
reports the true max error and violation count.

### The k search

Two stopping rules, `--k-criterion`:

* **`size`** (default) — price `k=1` (see below), then double from `--k-start`
  while each doubling shrinks the estimated payload by `--min-k-gain` (2%), with
  `--k-patience` doublings of slack. This is the rule that matters once residual
  coding is enforcing the bound.
* **`vq`** — the literal rule: double from `--k-start` until the max error of
  *pure* vector quantization meets the bound. On real weights this never
  terminates and the search runs to `--max-k`; see the findings below.

`k=1` means a single centroid, so residuals are coded against the window mean —
scalar quantization with no learned predictor at all. The `size` rule always
prices it alongside the doubling sequence, because a label is only worth paying
for if the sharper prediction saves more than the label costs, and on real
weights that is usually false. Setting `--k-start` equal to `--max-k` pins k and
skips the probe.

## Results on real checkpoints

<!--RESULTS-->
Hardware: RTX 5080 (16 GiB), torch 2.11 + CUDA 12.8. Defaults throughout
(`eb=1e-4` relative, 128 MiB windows, `tuple_size=2`, k search by size) unless noted.
Every row was verified by decoding the container and comparing against the
source value by value.

### Whole checkpoints

| checkpoint | source | windows | k | stored | bits/val | vs fp32 | vs source | max error | violations |
|---|---|---|---|---|---|---|---|---|---|
| gpt2 | 523 MiB fp32 | 5 | 1 | 232 MiB | 14.20 | 2.25x | 2.25x | 9.682e-05 | 0 |
| gpt2-medium | 1450 MiB fp32 | 12 | 1 | 659 MiB | 14.55 | 2.20x | 2.20x | 9.678e-05 | 0 |
| tinyllama | 2098 MiB bf16 | 33 | 1 | 2023 MiB | 15.43 | 2.07x | 1.04x | 9.714e-05 | 0 |

`vs source` is the number that matters for a bf16 checkpoint. TinyLlama sits
near 1x because a relative 1e-4 bound asks for 0.01% precision while bf16
only carries ~0.4% -- reproducing it that finely needs nearly its 16 bits.

For scale, lossless compression of the same bytes (256 MiB, zstd-3):

| checkpoint | zstd | zstd + byte-plane split | weightpress @ 1e-4 |
|---|---|---|---|
| gpt2 | 1.23x | 1.33x | 2.25x |
| gpt2-medium | 1.17x | 1.26x | 2.20x |
| tinyllama | 1.28x | 1.41x | 1.04x |

### Does the k-means predictor earn its labels?

gpt2, first 256 MiB, tuple size 2. `k=1` means a single centroid, i.e.
residuals coded against the window mean -- no learned predictor at all.

| setting | k | labels | residuals | bits/val | ratio |
|---|---|---|---|---|---|
| k1-no-kmeans | 1 | 0.0 MiB | 102.6 MiB | 13.70 | **2.34x** |
| k64-fixed | 64 | 19.6 MiB | 86.8 MiB | 14.17 | **2.26x** |
| k1024-fixed | 1024 | 41.5 MiB | 76.0 MiB | 15.56 | **2.06x** |
| k-search | 1 | 0.0 MiB | 102.6 MiB | 13.70 | **2.34x** |

**The codebook is a net loss.** Every extra label costs more than the
sharper prediction saves: adjacent log-magnitudes in a flattened
transformer tensor are close to uncorrelated, so a 2-D codebook has
almost no structure to exploit, and the label is paid for regardless.
This is not an artefact of the fit: an explicit label can only beat
scalar quantization by the lattice space-filling gain, ~0.17 bits per
2-D tuple.

The search finds this on its own: `k-search` lands on the same 2.34x as the hand-set `k=1` baseline, and chooses k=1 on every
window of all three checkpoints.

### Tuple size: when does a codebook start to pay?

A label costs `log2(k)/T` bits per value, so wider tuples amortise it.
The `k=256 forced` column prices the codebook directly by denying the
search the option of declining it.

| tuple size | k chosen | label bits/val | bits/val | ratio | ratio at k=256 forced |
|---|---|---|---|---|---|
| 1 | 1 | 0.000 | 13.70 | 2.336x | - |
| 2 | 1 | 0.000 | 13.70 | 2.335x | 2.191x |
| 4 | 1 | 0.000 | 13.70 | 2.335x | - |
| 8 | 1 | 0.000 | 13.70 | 2.335x | - |
| 16 | 1/64 | 0.152 | 13.69 | 2.337x | 2.338x |
| 32 | 64 | 0.153 | 13.66 | 2.342x | - |
| 64 | 64 | 0.075 | 13.66 | 2.343x | 2.343x |

### Error bound

| bound | k | bits/val | ratio | max error | escapes |
|---|---|---|---|---|---|
| 1e-03 | 1 | 11.25 | 2.84x | 9.96e-04 | 4190208 / 67,108,864 |
| 1e-04 | 1 | 13.70 | 2.34x | 9.59e-05 | 4190208 / 67,108,864 |
| 1e-05 | 1 | 17.11 | 1.87x | 1.00e-05 | 4190578 / 67,108,864 |
| 1e-06 | 1 | 20.85 | 1.53x | 1.00e-06 | 6607170 / 67,108,864 |

The escape counts are the point of the cost-based code width: with a
fixed 16-bit code, 1e-5 escaped 4.2M values and 1e-6 was hopeless.

### Window parallelism

gpt2-medium (12 windows, 1.45 GiB), varying `--max-workers`:

| max-workers | wall |
|---|---|
| 1 | 52.6s |
| 2 | 53.2s |
| 4 | 43.2s |
| 8 | 42.8s |

1.23x from 1 to 8. One 128 MiB window already saturates this GPU, so the
gain is from overlapping the host-side entropy coding and file I/O with
GPU work, not from more clustering throughput. The memory budget is what
keeps that concurrency safe: the estimate must cover a window's real peak
(~1.1 GiB here), or the run oversubscribes the device and spends its time
in the allocator's free-and-retry path instead -- which cost 10x when the
estimate was 2.5x low.

### The literal 'double k until max error fits' rule does not terminate

`--k-criterion vq` on gpt2, `--max-k 16384`:

| k | max VQ error | mean abs VQ error | label bits/tuple | residual bits/val |
|---|---|---|---|---|
| 64 | 10.882 | 0.13114 | 5.37 | 10.74 |
| 128 | 8.653 | 0.09377 | 6.24 | 10.35 |
| 256 | 5.938 | 0.06746 | 7.13 | 9.96 |
| 512 | 6.424 | 0.04824 | 8.03 | 9.55 |
| 1024 | 6.149 | 0.03484 | 8.85 | 9.15 |
| 2048 | 4.116 | 0.02538 | 9.71 | 8.76 |
| 4096 | 6.014 | 0.01811 | 10.55 | 8.33 |
| 8192 | 4.548 | 0.01279 | 11.40 | 7.91 |
| 16384 | 3.919 | 0.00909 | 12.24 | 7.48 |

Over a 256x increase in k the *mean* error falls
14x, while the *max* error stays around 3.9
-- it does not converge toward the bound at all.
Lloyd's minimises squared error, so extra centroids chase the bulk of the
distribution; the max is set by a handful of tail weights that keep
sharing a cluster with everything else. Even if it did converge, matching
1e-4 by quantization alone needs ~10^4 levels per dimension, so ~10^8
centroids at tuple size 2 -- more centroids than there are tuples in a
window. The search runs to `max_k` and the residual stage does the work.

`--mode vq` (labels only, no residuals) at k=256 shows what that
buys and what it costs: 3.26 bits/value at
9.8x, but a max error of 2331282.0 with
28,299,913 of 33,554,432 values (84%) outside the bound.

## Format

```
MAGIC | version:u32 | reserved:u32 | <chunk payloads...> | header_json | header_len:u64 | MAGIC
```

The header is written last and located by seeking to the end, so writing is a
single forward pass. Each payload is a centroid table followed by the zstd blobs
in a fixed order; the header records every length, the tensor manifest (name,
dtype, shape, offset), and the per-chunk `k`.

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

* f64 and wide-integer tensors are cast to fp32 on read, which is itself lossy;
  the bound is enforced relative to the fp32 stream.
* An error bound below the float32 ulp of the data forces every value to escape,
  giving a ratio near 1. That is correct but not useful.
* Windows are compressed independently, so there is no cross-window modelling.
