"""Configuration for a weightpress run."""

from __future__ import annotations

import dataclasses
import os
from typing import Literal

MAGIC = b"WPRS"
FORMAT_VERSION = 2

#: Residual codes are zigzagged and split into this many byte planes at most.
#: The count is chosen per window from the residual range actually observed: a
#: fixed 16-bit code word looks generous until the bound is tight, and then it
#: falls off a cliff.  At eb=1e-5 it caps residuals at +/-0.65, which sends every
#: LayerNorm weight and attention mask in gpt2 (all near 1.0) to the escape list
#: -- 4.2M values, a third of the output.  Unused high planes are constant and
#: cost almost nothing after zstd, so widening is close to free when unneeded.
MAX_CODE_PLANES = 4

#: Bound on |code| so that the zigzag of the widest code still fits in int32.
CODE_MAX = (1 << 30) - 1

KCriterion = Literal["size", "vq"]
#: ``cluster`` -- the design's algorithm: k-means/VQ clustering where k is the
#: number of clusters, grown until the max percentage error meets the bound; each
#: value is stored as its cluster label, losslessly compressed.  ``residual`` and
#: ``vq`` are the earlier variants kept for comparison.
Mode = Literal["cluster", "residual", "vq"]
#: ``relative`` enforces |x - x_hat| / |x| <= eb (max percentage error) by coding
#: in the log domain; ``absolute`` enforces |x - x_hat| <= eb on a linear grid.
ErrorMode = Literal["relative", "absolute"]


@dataclasses.dataclass
class Config:
    """Knobs for a compression run.

    The five documented inputs are :attr:`error_bound`, :attr:`window_size`,
    :attr:`tuple_size`, :attr:`max_gpu_memory` and :attr:`output_dir`; the rest
    are tuning parameters with sensible defaults.
    """

    # --- the five documented inputs -------------------------------------
    error_bound: float = 1e-4
    #: Whether ``error_bound`` is a relative (max percentage) or absolute bound.
    #: The design's bound is relative, so that is the default.
    error_mode: ErrorMode = "relative"
    window_size: int = 128 << 20
    tuple_size: int = 2
    #: Explicit GPU byte budget; None -> ``gpu_budget_fraction`` of free memory.
    max_gpu_memory: int | None = None
    #: Fraction of free GPU memory to claim when ``max_gpu_memory`` is unset.
    #: The issue suggested 80%; kept conservative at 50% so weightpress leaves
    #: room for other GPU work and does not oversubscribe the device.
    gpu_budget_fraction: float = 0.5
    output_dir: str = "."

    # --- k search -------------------------------------------------------
    k_start: int = 64
    #: Upper bound on the cluster count / codebook resolution.  Cluster mode at a
    #: tight bound needs a high ceiling (1e-4 lands near 2^17, 1e-6 near 2^24).
    max_k: int = 1 << 24
    k_criterion: KCriterion = "size"
    #: For ``k_criterion="size"``: doubling k must shrink the estimated payload
    #: by at least this fraction to be worth the wider labels.
    min_k_gain: float = 0.02
    #: How many non-improving doublings to try before concluding k has peaked.
    k_patience: int = 2

    # --- k-means fitting ------------------------------------------------
    kmeans_iters: int = 25
    #: Centroids are fit on a subsample; assignment always touches every tuple.
    fit_sample_per_k: int = 64
    fit_sample_min: int = 1 << 18
    fit_sample_max: int = 1 << 22
    #: Tuples sampled when estimating residual entropy during the k search.
    entropy_sample_tuples: int = 1 << 21
    seed: int = 0

    # --- output ---------------------------------------------------------
    mode: Mode = "cluster"
    zstd_level: int = 3
    #: Upper bound on windows in flight; the GPU memory budget may lower it.
    max_workers: int = 8
    device: str = "cuda"
    verify: bool = True

    def resolved_output_dir(self) -> str:
        return os.path.abspath(self.output_dir)

    def values_per_window(self, itemsize: int) -> int:
        return max(self.tuple_size, (self.window_size // itemsize))
