"""Configuration for a weightpress run."""

from __future__ import annotations

import dataclasses
import os
from typing import Literal

MAGIC = b"WPRS"
FORMAT_VERSION = 1

#: Codes are stored zigzagged in this many bytes; anything wider escapes to the
#: raw-value list.  16 bits covers residuals up to 32767 * step.
CODE_BITS = 16
CODE_MAX = (1 << (CODE_BITS - 1)) - 1

KCriterion = Literal["size", "vq"]
Mode = Literal["residual", "vq"]


@dataclasses.dataclass
class Config:
    """Knobs for a compression run.

    The five documented inputs are :attr:`error_bound`, :attr:`window_size`,
    :attr:`tuple_size`, :attr:`max_gpu_memory` and :attr:`output_dir`; the rest
    are tuning parameters with sensible defaults.
    """

    # --- the five documented inputs -------------------------------------
    error_bound: float = 1e-4
    window_size: int = 128 << 20
    tuple_size: int = 2
    max_gpu_memory: int | None = None  # None -> 80% of free at startup
    output_dir: str = "."

    # --- k search -------------------------------------------------------
    k_start: int = 64
    max_k: int = 1 << 16
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
    mode: Mode = "residual"
    zstd_level: int = 3
    #: Upper bound on windows in flight; the GPU memory budget may lower it.
    max_workers: int = 8
    device: str = "cuda"
    verify: bool = True

    def resolved_output_dir(self) -> str:
        return os.path.abspath(self.output_dir)

    def values_per_window(self, itemsize: int) -> int:
        return max(self.tuple_size, (self.window_size // itemsize))
