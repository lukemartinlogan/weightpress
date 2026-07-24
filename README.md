# weightpress

Learned error-bounded lossy compression for model weights: GPU k-means as the
predictor, quantized residuals as the error-bound enforcer, zstd as the lossless
integer back end.

Every stored value is guaranteed to reconstruct within an absolute error bound
(default `1e-4`). The guarantee is structural, not statistical — see
[How the bound is guaranteed](#how-the-bound-is-guaranteed).

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
| `-e, --error-bound` | `1e-4` | absolute error bound, per weight |
| `-w, --window-size` | `128MB` | window size, in bytes of the fp32 working stream |
| `-t, --tuple-size` | `2` | weights per k-means vector |
| `-g, --max-gpu-memory` | 80% of free | budget that sets how many windows run at once |
| `-o, --output-dir` | `.` | where the container and k-means tables are written |

Output for `compress model.safetensors -o out/`:

* `out/model.wp` — self-contained container (centroids, labels, residuals)
* `out/model.kmeans/chunk_NNNNNN.npz` — the k-means table for each window,
  standalone and independently inspectable, as the design calls for

## Algorithm

Per 128 MB window, independently and in parallel:

1. **Tuple** the window into `n/T` vectors of `T` adjacent weights (`T=2`).
2. **Fit** k-means on the GPU. Lloyd iterations run on a random subsample so the
   cost scales with `k` rather than with the window; assignment then sweeps every
   tuple. Empty clusters are reseeded onto the currently worst-fit points.
3. **Search k** over powers of two starting at 64 (below).
4. **Predict** each weight with its centroid and **quantize the residual** onto a
   grid of width `2·eb`.
5. **Code losslessly**: labels at the narrowest integer width that fits `k`;
   residuals zigzagged, split into low/high byte planes, and zstd'd.

### How the bound is guaranteed

The residual stage is what enforces the bound, not k-means:

```
q     = round((x - centroid) / step)          step = 2·eb·(1 - 1e-4)
x_hat = centroid + q·step                =>   |x - x_hat| ≤ eb
```

Three details make this hold in practice rather than only on paper:

* **The grid is 0.01% finer than `2·eb`.** A value landing exactly on the bound
  would otherwise be pushed past it by float32 rounding in `centroid + q·step`.
* **The encoder reconstructs with the decoder's exact arithmetic** and checks
  every value, rather than checking against a more precise calculation than the
  one that actually runs.
* **Anything that still misses escapes**, stored verbatim as float32 and
  reconstructed exactly. Code `0` is reserved as the escape marker. This covers
  residuals too large for the 16-bit code word, NaN/Inf, and error bounds below
  the float32 ulp of the data. On real checkpoints escapes are ~50 values in 137
  million.

`weightpress verify` re-reads the source and the container side by side and
reports the true max error and violation count.

### The k search

Two stopping rules, `--k-criterion`:

* **`size`** (default) — double while it still shrinks the estimated payload by
  `--min-k-gain` (2%), with `--k-patience` doublings of slack. This is the rule
  that matters once residual coding is enforcing the bound.
* **`vq`** — the literal rule: double until the max error of *pure* vector
  quantization meets the bound. On real weights this never terminates and the
  search runs to `--max-k`; see the findings below.

`k=1` is legal and is the honest baseline: one centroid means residuals are coded
against the window mean, i.e. plain scalar quantization with no learned
predictor.

## Results on real checkpoints

Hardware: RTX 5080 (16 GiB), torch 2.11 + CUDA 12.8. Defaults throughout
(`eb=1e-4`, 128 MiB windows, `tuple_size=2`, k search by size) unless noted.
Every row was verified by decoding the container and comparing against the
source value by value.

### Whole checkpoints

| checkpoint | source | windows | k | stored | bits/val | vs fp32 | vs source | max error | violations |
|---|---|---|---|---|---|---|---|---|---|
| gpt2 | 523 MiB fp32 | 5 | 1 | 178 MiB | 10.92 | 2.93x | 2.93x | 9.999e-05 | 0 |
| gpt2-medium | 1450 MiB fp32 | 12 | 1 | 499 MiB | 11.02 | 2.90x | 2.90x | 9.999e-05 | 0 |
| tinyllama | 2098 MiB bf16 | 33 | 1 | 1219 MiB | 9.30 | 3.44x | 1.72x | 9.999e-05 | 0 |

`vs source` is the number that matters for a bf16 checkpoint: the fp32 column
credits the codec for undoing a widening it performed itself.

For scale, lossless compression of the same bytes (256 MiB, zstd-3):

| checkpoint | zstd | zstd + byte-plane split | weightpress @ 1e-4 |
|---|---|---|---|
| gpt2 | 1.23x | 1.33x | 2.93x |
| gpt2-medium | 1.17x | 1.26x | 2.90x |
| tinyllama | 1.28x | 1.41x | 1.72x |

### Does the k-means predictor earn its labels?

gpt2, first 256 MiB, tuple size 2. `k=1` means a single centroid, i.e.
residuals coded against the window mean -- no learned predictor at all.

| setting | k | labels | residuals | bits/val | ratio |
|---|---|---|---|---|---|
| k1-no-kmeans | 1 | 0.0 MiB | 84.2 MiB | 10.53 | **3.04x** |
| k64-fixed | 64 | 19.1 MiB | 68.1 MiB | 10.90 | **2.93x** |
| k1024-fixed | 1024 | 41.4 MiB | 50.7 MiB | 11.51 | **2.78x** |
| k-search | 1 | 0.0 MiB | 84.2 MiB | 10.53 | **3.04x** |

**The codebook is a net loss.** At k=64 the labels cost 2.66 bits/value
and buy back only 2.09 bits of residual entropy. Adjacent weights in a
flattened transformer tensor are close to uncorrelated, so a 2-D
codebook has almost no structure to exploit -- and the label has to be
paid for regardless. This is not an artefact of the fit: an explicit
label can only beat scalar quantization by the space-filling advantage
of the lattice, which in 2-D is bounded by about 0.17 bits per tuple.

The search finds this on its own -- `k-search` lands on the same 3.04x as
the hand-set `k=1` baseline, on every window of all three checkpoints.

### Tuple size: when does a codebook start to pay?

A label costs `log2(k)/T` bits per value, so wider tuples amortise it.
The `k=256 forced` column prices the codebook directly by denying the
search the option of declining it.

| tuple size | k chosen | label bits/val | bits/val | ratio | ratio at k=256 forced |
|---|---|---|---|---|---|
| 1 | 1 | 0.000 | 10.53 | 3.039x | - |
| 2 | 1 | 0.000 | 10.53 | 3.039x | 2.970x |
| 4 | 1 | 0.000 | 10.53 | 3.039x | - |
| 8 | 1 | 0.000 | 10.53 | 3.039x | - |
| 16 | 1/128 | 0.177 | 10.55 | 3.034x | 3.016x |
| 32 | 1/64 | 0.075 | 10.51 | 3.045x | - |
| 64 | 1/64 | 0.037 | 10.49 | 3.050x | 3.049x |

### Error bound

| bound | k | bits/val | ratio | max error | escapes |
|---|---|---|---|---|---|
| 1e-03 | 1 | 7.42 | 4.31x | 1.00e-03 | 0 |
| 1e-04 | 1 | 10.53 | 3.04x | 1.00e-04 | 136 |
| 1e-05 | 1 | 13.05 | 2.45x | 1.00e-05 | 4247646 |

### The literal 'double k until max error fits' rule does not terminate

`--k-criterion vq` on gpt2, `--max-k 16384`:

| k | max VQ error | mean abs VQ error | label bits/tuple | residual bits/val |
|---|---|---|---|---|
| 64 | 16.419 | 0.01764 | 5.32 | 8.47 |
| 128 | 15.908 | 0.01234 | 6.26 | 7.95 |
| 256 | 14.783 | 0.00861 | 7.23 | 7.41 |
| 512 | 15.440 | 0.00606 | 8.16 | 6.97 |
| 1024 | 15.394 | 0.00449 | 9.01 | 6.59 |
| 2048 | 9.163 | 0.00320 | 9.86 | 6.15 |
| 4096 | 15.120 | 0.00232 | 10.71 | 5.74 |
| 8192 | 15.224 | 0.00163 | 11.55 | 5.28 |
| 16384 | 12.915 | 0.00116 | 12.39 | 4.83 |

Over a 256x increase in k the *mean* error falls
15x, while the *max* error goes from 16.4
to 12.9 -- it does not converge toward 1e-4 at all.
Lloyd's minimises squared error, so extra centroids chase the bulk of the
distribution; the max is set by a handful of tail weights that keep
sharing a cluster with everything else. Even if it did converge, matching
1e-4 by quantization alone needs ~10^4 levels per dimension, so ~10^8
centroids at tuple size 2 -- more centroids than there are tuples in a
window. The search runs to `max_k` and the residual stage does the work.

`--mode vq` (store labels only, no residuals) at k=256 shows what
that costs: 3.25 bits/value and a fine
9.8x ratio, but max error
16.0 and 28,133,234 of
33,554,432 values outside the bound.

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
