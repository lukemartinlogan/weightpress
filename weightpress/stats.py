"""Per-chunk and per-run reporting structures."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class ChunkStats:
    """What the k search decided for one window, and what it cost."""

    index: int
    n_values: int
    raw_bytes: int
    #: In cluster mode, the number of clusters (codebook resolution) the search
    #: settled on -- the design's k.  In residual/vq mode, the k-means k.
    k: int
    #: Distinct clusters actually used (<= k); the stored codebook size.
    occupied_clusters: int = 0
    k_trials: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    #: Max |x - centroid| over the window, i.e. the error of pure vector
    #: quantization before residual correction.
    vq_max_error: float = float("nan")
    #: Max |x - reconstruction| of the stored bitstream.  In residual mode this
    #: is guaranteed <= error_bound.
    max_error: float = float("nan")
    mean_abs_error: float = float("nan")
    n_outliers: int = 0
    codebook_bytes: int = 0
    label_bytes: int = 0
    code_bytes: int = 0
    outlier_bytes: int = 0
    seconds: float = 0.0

    @property
    def stored_bytes(self) -> int:
        return (
            self.codebook_bytes + self.label_bytes + self.code_bytes + self.outlier_bytes
        )

    @property
    def ratio(self) -> float:
        return self.raw_bytes / max(1, self.stored_bytes)

    @property
    def bits_per_value(self) -> float:
        return 8.0 * self.stored_bytes / max(1, self.n_values)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["stored_bytes"] = self.stored_bytes
        d["ratio"] = self.ratio
        d["bits_per_value"] = self.bits_per_value
        return d


@dataclasses.dataclass
class RunStats:
    """Aggregate over every window of an input."""

    source: str
    error_bound: float
    tuple_size: int
    window_size: int
    mode: str
    chunks: list[ChunkStats] = dataclasses.field(default_factory=list)
    seconds: float = 0.0
    concurrency: int = 1
    gpu_budget_bytes: int = 0

    @property
    def raw_bytes(self) -> int:
        return sum(c.raw_bytes for c in self.chunks)

    @property
    def stored_bytes(self) -> int:
        return sum(c.stored_bytes for c in self.chunks)

    @property
    def ratio(self) -> float:
        return self.raw_bytes / max(1, self.stored_bytes)

    @property
    def max_error(self) -> float:
        return max((c.max_error for c in self.chunks), default=0.0)

    @property
    def bits_per_value(self) -> float:
        n = sum(c.n_values for c in self.chunks)
        return 8.0 * self.stored_bytes / max(1, n)

    def k_histogram(self) -> dict[int, int]:
        hist: dict[int, int] = {}
        for c in self.chunks:
            hist[c.k] = hist.get(c.k, 0) + 1
        return dict(sorted(hist.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "error_bound": self.error_bound,
            "tuple_size": self.tuple_size,
            "window_size": self.window_size,
            "mode": self.mode,
            "seconds": self.seconds,
            "concurrency": self.concurrency,
            "gpu_budget_bytes": self.gpu_budget_bytes,
            "raw_bytes": self.raw_bytes,
            "stored_bytes": self.stored_bytes,
            "ratio": self.ratio,
            "bits_per_value": self.bits_per_value,
            "max_error": self.max_error,
            "k_histogram": self.k_histogram(),
            "chunks": [c.to_dict() for c in self.chunks],
        }
