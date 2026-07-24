"""weightpress -- learned error-bounded lossy compression for model weights.

The pipeline splits a weight stream into fixed-size windows, fits a k-means
codebook over ``tuple_size``-dimensional tuples of adjacent weights (the learned
regressor), then codes the prediction residual as integers under a hard absolute
error bound and compresses those integers losslessly.
"""

from .config import Config
from .stats import ChunkStats, RunStats

__version__ = "0.1.0"
__all__ = ["ChunkStats", "Config", "RunStats", "__version__"]
