# weightpress — study resume notes

Snapshot for picking this up on another (less GPU-contended) machine. Last
updated 2026-07-25. Repo: `git@github.com:lukemartinlogan/weightpress.git`,
branch `main`.

## What this is

Error-bounded lossy compression for model weights (issue #1). A **k-means / VQ
codebook is the learned regressor** and the per-value **cluster labels are the
losslessly compressed integer stream**. Default `mode=cluster`:

- **k is the number of clusters.** The search doubles k from 64 until the max
  **percentage** (relative) error meets the bound, and reports it.
- Clustering happens in the **log domain** (`u = log|x|`), so an absolute cell
  width `2·ln(1+eb)` bounds relative error at `eb`. Sign stored separately;
  `x_hat = sign·exp(centroid)`. Zeros / non-finite escape (stored verbatim).
- The clusters are **uniform-in-log, not Lloyd's k-means**: Lloyd's minimises
  mean-squared error and can't meet a hard max bound at any practical k (its max
  is a tail statistic). See README "Why the clusters are uniform-in-log".
- The reconstruct-and-check is done **on the CPU with the decoder's exact numpy
  `exp`**, so what's verified is bit-for-bit what decode produces. Anything still
  over the bound escapes.

Other modes kept for comparison: `--mode residual` (small k-means predictor +
entropy-coded residual; smaller, but then k is the predictor size), `--mode vq`
(pure VQ — massively violates the bound at feasible k, illustrative only).

## Current status — DONE

- Cluster / residual / vq modes, relative + absolute error modes.
- CLI (`weightpress compress|decompress|verify`), container format v2 (codebook,
  labels, signs, escapes), safetensors/npy/raw readers.
- GPU k-means (residual/vq); cluster mode uses tiled GPU log/floor.
- 89 tests pass (`.venv/bin/python -m pytest tests/ -q`), lint clean.
- Memory: cluster GPU peak ~122 MiB/window (was 1.1 GiB); default GPU budget 50%
  of free (`-g` / `Config.gpu_budget_fraction` to change); reader streams large
  tensors in 64 MiB slices; host-RAM-aware concurrency cap.

## Results so far (cluster mode, eb=1e-4 relative, 0 violations everywhere)

| model | dtype | k (clusters) | occupied | bits/val | ratio vs source | max %err |
|---|---|---|---|---|---|---|
| gpt2 | fp32 | 131072 (2^17) | 69,594 | 17.49 | 1.83x | 8.9e-05 |
| gpt2-medium | fp32 | 131072 (2^17) | 106,426 | 17.25 | 1.86x | 9.5e-05 |
| tinyllama-1.1B | bf16 | 131072 (2^17) | 4,483 | 12.60 | 1.27x | 9.6e-05 |
| **gemma-4-E2B** | bf16 | mostly 2^17 (some 2^18) | 1.8k–7.9k/window | 12.17 | **~1.31x** | 1.0e-04 |

- gemma-4-E2B: 5.12B params, all BF16, 10.25 GB source → 19.09 GiB fp32 working
  stream, 153 windows, stored 7.26 GiB, **2.63x vs fp32 / ~1.31x vs bf16
  source**, 888 s (~15 min) on the contended RTX 5080. Report:
  `/home/iowarp/wp-gemma/gemma.json` (NOT in the repo — see below).
  **Independent `verify` PASSED**: all 5,123,178,979 values re-read from source,
  max %err 9.997e-05, 0 violations.
- Bound sweep on gpt2: k doubles per decade — 2^11 (1e-2) → 2^14 → 2^17 →
  2^21 → 2^24 (1e-6); ratio 3.16x → 0.89x. All 0 violations.
- Lossless zstd baseline on the same bytes: 1.17–1.33x (vs our ~1.8x fp32).
- **Key finding:** the k-means codebook doesn't earn its labels — clustering +
  lossless label coding *is* the compression; ~16 bits/value at 1e-4 is what
  pinning 0.01% relative precision on every weight genuinely costs.

## Environment / setup on a new machine

Hardware used here: RTX 5080 (16 GiB, sm_120), **only ~10 GiB free / shared with
a desktop** — that contention is the reason to move. Host had only 11 GiB RAM,
which was the real bottleneck (see caveats). A box with more free VRAM **and**
32 GB+ RAM will be much smoother.

```bash
git clone git@github.com:lukemartinlogan/weightpress.git && cd weightpress
python -m venv .venv
.venv/bin/pip install -e .                                   # numpy, zstandard
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install safetensors huggingface_hub pytest    # models + tests
.venv/bin/python -m pytest tests/ -q                         # sanity (needs CUDA)
```

torch 2.11 + CUDA 12.8 is what worked for sm_120 (RTX 5080). Adjust the wheel
index for your GPU's compute capability.

### Get the models (none are committed; all are re-downloadable)

`google/gemma-4-*` is **public** (`gated=False`) — no HF token needed. Gemma 3 is
gated (`gated=manual`).

```python
from huggingface_hub import hf_hub_download
hf_hub_download('google/gemma-4-E2B', 'model.safetensors',
                local_dir='models/gemma-4-E2B')          # 10.25 GB, bf16, 5.12B
# smaller comparisons already studied:
hf_hub_download('openai-community/gpt2', 'model.safetensors', local_dir='models/gpt2')
hf_hub_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'model.safetensors',
                local_dir='models/tinyllama')
```

## Reproduce / continue

```bash
# one model, cluster mode (default), full report + roundtrip verify
.venv/bin/python -m weightpress.cli compress models/gemma-4-E2B/model.safetensors \
    -o out --json out/gemma.json
.venv/bin/python -m weightpress.cli verify out/model.wp     # independent re-check

# experiment matrix (edit the models dict / --models-dir in the script)
.venv/bin/python scripts/experiments.py --models-dir models --out results --only models
.venv/bin/python scripts/experiments.py --models-dir models --out sweeps --only bound,compare
.venv/bin/python scripts/baseline_lossless.py --models-dir models --out results/lossless.json
.venv/bin/python scripts/make_report.py --results results --sweeps sweeps \
    --lossless results/lossless.json --readme README.md   # splices into <!--RESULTS-->
```

Handy flags: `--limit-bytes 256MB` (quick runs), `-e 1e-5` (tighter bound),
`--mode residual|vq`, `--error-mode absolute`, `-g 4GB` (cap GPU), `-w 64MB`
(smaller windows → less RAM per worker), `--max-workers N`.

## Caveats / gotchas to carry over

- **Host RAM, not GPU, was the binding constraint.** Cluster/residual modes
  finalize each window on the CPU; concurrency is capped by `available_host_memory
  × 0.5 / (~24 bytes·values_per_window)`. On 11 GiB this allowed ~4 windows. More
  RAM → more parallelism → faster. gemma's 9.4 GiB fp32-widened embedding forced
  the reader to stream in slices (already fixed).
- **Cluster mode is slow** (~22 MiB/s of fp32 stream; gemma ~15 min) because the
  label stream is huge and zstd-bound; the GPU is nearly idle. Not optimized —
  perf was explicitly deprioritized. A less contended GPU won't speed cluster
  mode much; more CPU cores / RAM (more parallel windows, higher zstd level
  tradeoffs) would.
- **Ratios are modest by design.** A relative 1e-4 bound needs ~16 bits/value on
  fp32; bf16 sources cap near 1x–1.4x because they carry only ~0.4% relative
  precision to begin with. This is inherent, not a codec bug.
- The `out/*.wp` containers and `models/` are gitignored — regenerate them.

## Open questions / next steps for the study

1. **Add gemma-4-E2B to the README results** (verified: ~1.31x vs source, 0
   violations; the `experiments.py` models dict already includes it).
2. **Is bf16 worth widening to fp32 at all?** For a relative bound, quantizing
   bf16 in its native precision (8-bit mantissa) may need far fewer clusters.
   Consider a bf16-native path / measuring vs-source honestly everywhere.
3. **Larger gemma-4 variants** (12B, 31B) once RAM allows — do bigger models
   have lower label entropy (fewer occupied clusters per window)?
4. **Cheaper labels:** the occupied-cluster codebook is tiny (1.8k–7.9k for
   gemma) yet labels are ~91% of bytes. Try delta/context modelling of adjacent
   labels, or per-tensor (not per-window) clustering, to cut label entropy.
5. **Absolute vs relative comparison** on the same models (absolute mode exists).
6. Optionally revisit **residual mode** at tight bounds — it was smaller than
   cluster at 1e-4 (2.34x vs 1.90x on gpt2) while holding the bound.

## Repo map

- `weightpress/config.py` — `Config` (defaults; `mode`, `error_mode`,
  `gpu_budget_fraction`, `max_k`, …)
- `weightpress/pipeline.py` — `compress`/`decompress`, `_cluster_window`
  (default), `_quantize_gpu_relative` + `_finalize_relative` (residual),
  concurrency + host/GPU budgeting
- `weightpress/codec.py` — quantization, log-domain helpers, cluster
  reconstruct, plane packing, `EncodedChunk`, zstd
- `weightpress/kmeans.py` — GPU k-means + power-of-two k search (residual/vq)
- `weightpress/reader.py` — safetensors (chunked)/npy/raw, bf16→fp32 widening
- `weightpress/container.py` — container v2 read/write + per-window `.npz` tables
- `weightpress/cli.py` — `compress|decompress|verify`
- `scripts/experiments.py`, `scripts/baseline_lossless.py`,
  `scripts/make_report.py` — evaluation + README results generation
- `tests/` — `test_codec.py`, `test_container.py` (numpy only);
  `test_pipeline.py` (torch, CPU+CUDA)
