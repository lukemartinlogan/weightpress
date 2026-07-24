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

<!--RESULTS-->

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
