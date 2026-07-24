"""GPU k-means over weight tuples, plus the power-of-two k search.

The window is viewed as ``n`` tuples of ``tuple_size`` adjacent weights.  Fitting
runs Lloyd iterations on a random subsample (so cost scales with k, not with the
window), while *assignment* always sweeps every tuple -- the reported errors and
entropies therefore describe the whole window, never a sample of it.

Two stopping rules for the k search are available:

``vq``
    The literal rule from the design: start at k=64 and double until the max
    elementwise error of pure vector quantization drops below the error bound.
    On real weights this is unreachable at any practical k (see README), so the
    search stops at ``max_k``.

``size``
    Double while the extra centroids actually pay for themselves -- i.e. while
    the estimated coded size (label entropy + residual entropy + codebook)
    shrinks by at least ``min_k_gain``.  This is the rule that matters once the
    residual stage is doing the error bounding.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from .codec import quant_step
from .config import Config
from .gpu import torch

#: Bytes for the (points x centroids) distance tile that dominates assignment.
DIST_BUFFER_BYTES = 256 << 20
#: Residual passes only need an (m x tuple_size) tile, so they can use far
#: bigger strides than assignment -- fewer kernel launches, fewer stalls.
RESID_TILE_VALUES = 1 << 26
#: Residual codes are histogrammed over +/- this many bins; the tails beyond it
#: are lumped in.  This only feeds the k decision, never the stored size.
HIST_HALF = 1 << 15


@dataclasses.dataclass
class Evaluation:
    """What one candidate k achieves on the full window."""

    k: int
    vq_max_error: float
    vq_mean_abs_error: float
    label_entropy_bits: float
    code_entropy_bits: float
    codebook_bytes: int
    est_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class KMeansResult:
    centroids: Any  # (k, T) fp32 tensor
    labels: Any  # (n,) int64 tensor
    evaluation: Evaluation
    trials: list[Evaluation]


def _tile_rows(k: int, n: int) -> int:
    """Rows of the distance tile that keep it inside ``DIST_BUFFER_BYTES``."""
    elems = DIST_BUFFER_BYTES // 4
    return int(max(256, min(n, elems // max(1, k))))


def assign(x, centroids, *, want_dist: bool = False):
    """Nearest-centroid assignment over every row of ``x``.

    Returns ``(labels, dist2)`` where ``dist2`` is ``None`` unless ``want_dist``.
    The ``||x||^2`` term is dropped because it is constant within a row and so
    cannot change the argmin.
    """
    t = torch()
    n = x.shape[0]
    k = centroids.shape[0]
    labels = t.empty(n, dtype=t.int64, device=x.device)
    dist2 = t.empty(n, dtype=t.float32, device=x.device) if want_dist else None
    c_sq = (centroids * centroids).sum(1)
    ct = centroids.t().contiguous()
    step = _tile_rows(k, n)
    for lo in range(0, n, step):
        hi = min(n, lo + step)
        chunk = x[lo:hi]
        scores = t.addmm(c_sq.unsqueeze(0), chunk, ct, beta=1.0, alpha=-2.0)
        if want_dist:
            best, idx = scores.min(1)
            labels[lo:hi] = idx
            # Re-add the dropped ||x||^2 so the value is a true squared distance.
            dist2[lo:hi] = best + (chunk * chunk).sum(1)
        else:
            labels[lo:hi] = scores.argmin(1)
        del scores
    return labels, dist2


def fit(x, k: int, cfg: Config, *, generator=None):
    """Lloyd's algorithm on a subsample of ``x``; returns ``(k, tuple_size)``.

    Empty clusters are reseeded onto the points that are currently worst-fit,
    which is what pushes the *max* error down rather than just the mean.
    """
    t = torch()
    n = x.shape[0]
    target = min(
        max(cfg.fit_sample_per_k * k, cfg.fit_sample_min), cfg.fit_sample_max, n
    )
    if target < n:
        idx = t.randint(0, n, (target,), device=x.device, generator=generator)
        sample = x[idx].contiguous()
    else:
        sample = x

    m = sample.shape[0]
    k = min(k, m)
    init = t.randperm(m, device=x.device, generator=generator)[:k]
    centroids = sample[init].clone().float()

    prev = None
    for _ in range(cfg.kmeans_iters):
        labels, dist2 = assign(sample, centroids, want_dist=True)
        totals = t.zeros_like(centroids)
        totals.index_add_(0, labels, sample)
        counts = t.zeros(k, dtype=t.float32, device=x.device)
        counts.index_add_(0, labels, t.ones(m, dtype=t.float32, device=x.device))
        centroids = totals / counts.clamp(min=1.0).unsqueeze(1)

        empty = t.nonzero(counts == 0, as_tuple=False).flatten()
        if empty.numel() > 0:
            worst = t.topk(dist2, k=int(empty.numel())).indices
            centroids[empty] = sample[worst]

        if prev is not None and t.equal(prev, labels):
            break
        prev = labels
    return centroids


def _entropy_bits(counts) -> float:
    t = torch()
    counts = counts[counts > 0].to(t.float64)
    if counts.numel() <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * t.log2(p)).sum().item())


def _resid_tile(n: int, tuple_size: int) -> int:
    return int(max(1, min(n, RESID_TILE_VALUES // max(1, tuple_size))))


def evaluate(x, centroids, labels, cfg: Config, k: int) -> Evaluation:
    """Error and entropy figures for one candidate codebook.

    Errors are exact over the whole window -- they are reported to the user and
    drive the ``vq`` stopping rule.  The residual entropy only ranks candidate
    k values against each other, so it is estimated from a strided subsample;
    the size that actually gets reported always comes from zstd.
    """
    t = torch()
    n, tsize = x.shape
    n_values = n * tsize
    step = quant_step(cfg.error_bound)

    max_err = t.zeros((), dtype=t.float32, device=x.device)
    abs_sum = t.zeros((), dtype=t.float64, device=x.device)
    tile = _resid_tile(n, tsize)
    for lo in range(0, n, tile):
        hi = min(n, lo + tile)
        a = (x[lo:hi] - centroids[labels[lo:hi]]).abs()
        max_err = t.maximum(max_err, a.max())
        abs_sum += a.sum(dtype=t.float64)

    stride = max(1, n // max(1, cfg.entropy_sample_tuples))
    xs = x[::stride]
    q = t.round((xs - centroids[labels[::stride]]) / step)
    q = q.nan_to_num_(0.0).clamp_(-HIST_HALF, HIST_HALF).to(t.int64)
    code_counts = t.bincount((q + HIST_HALF).flatten(), minlength=2 * HIST_HALF + 1)

    h_label = _entropy_bits(t.bincount(labels, minlength=k))
    h_code = _entropy_bits(code_counts)
    codebook_bytes = k * tsize * 4
    est_bits = n * h_label + n_values * h_code
    return Evaluation(
        k=k,
        vq_max_error=float(max_err.item()),
        vq_mean_abs_error=float(abs_sum.item() / max(1, n_values)),
        label_entropy_bits=h_label,
        code_entropy_bits=h_code,
        codebook_bytes=codebook_bytes,
        est_bytes=int(est_bits / 8) + codebook_bytes,
    )


def search_k(x, cfg: Config, *, generator=None) -> KMeansResult:
    """Find the smallest power-of-two k that satisfies the configured rule.

    ``k=1`` is legal and is the useful baseline: one centroid means the residual
    is coded against the window mean, i.e. plain scalar quantization with no
    learned predictor at all.
    """
    n = x.shape[0]
    k = max(1, 1 << int(math.log2(max(1, cfg.k_start))))
    trials: list[Evaluation] = []

    best_centroids = None
    best_labels = None
    best_eval: Evaluation | None = None
    stale = 0

    while True:
        centroids = fit(x, k, cfg, generator=generator)
        labels, _ = assign(x, centroids)
        ev = evaluate(x, centroids, labels, cfg, centroids.shape[0])
        trials.append(ev)

        if cfg.k_criterion == "vq":
            # Keep the newest: larger k always has lower or equal VQ error.
            take, stop = True, ev.vq_max_error <= cfg.error_bound
        else:
            take = (
                best_eval is None
                or ev.est_bytes < best_eval.est_bytes * (1.0 - cfg.min_k_gain)
            )
            # Payload-vs-k is not perfectly monotone, so give the search a
            # couple of doublings of patience before concluding it has peaked.
            stale = 0 if take else stale + 1
            stop = stale > cfg.k_patience

        if take:
            best_centroids, best_labels, best_eval = centroids, labels, ev

        if stop or k * 2 > cfg.max_k or k >= n:
            break
        k *= 2

    assert best_eval is not None and best_centroids is not None
    return KMeansResult(
        centroids=best_centroids,
        labels=best_labels,
        evaluation=best_eval,
        trials=trials,
    )
