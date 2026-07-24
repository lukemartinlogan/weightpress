"""Device selection and GPU memory budgeting.

Torch is imported lazily so that the codec and container layers stay importable
(and testable) on machines without CUDA.
"""

from __future__ import annotations

import functools
import os

_TORCH = None


def torch():
    """Import torch on first use and cache it."""
    global _TORCH
    if _TORCH is None:
        import torch as _t

        _TORCH = _t
    return _TORCH


def have_cuda() -> bool:
    try:
        return torch().cuda.is_available()
    except Exception:
        return False


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not have_cuda():
        return "cpu"
    return requested


@functools.cache
def device_name(device: str) -> str:
    if device.startswith("cuda"):
        return torch().cuda.get_device_name(device)
    return "cpu"


def free_memory(device: str) -> int:
    """Bytes currently free on ``device`` (whole system, not just this process)."""
    if not device.startswith("cuda"):
        # Fall back to something conservative so CPU runs still parallelise.
        return available_host_memory()
    free, _total = torch().cuda.mem_get_info(device)
    return int(free)


def default_budget(device: str, fraction: float = 0.8) -> int:
    """80% of whatever is left on the device, per the documented default."""
    return int(free_memory(device) * fraction)


def available_host_memory() -> int:
    """Bytes of free system RAM, for capping host-side concurrency."""
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 4 << 30


def plan_concurrency(
    budget_bytes: int, per_window_bytes: int, max_workers: int
) -> int:
    """How many windows can be in flight at once without blowing the budget."""
    if per_window_bytes <= 0:
        return 1
    fits = int(budget_bytes // per_window_bytes)
    return max(1, min(max_workers, fits))


def estimate_window_bytes(
    n_values: int, tuple_size: int, dist_buffer: int, resid_tile_values: int
) -> int:
    """Peak device memory for one window.

    Counts the fp32 window and its padded copy, the int64 label vector, the two
    output byte planes plus the escape mask, the distance tile that dominates
    the k-means inner loop, and the residual-pass scratch.

    The last term is the one that is easy to get wrong: a residual tile carries
    roughly a dozen live temporaries, several of them float64, so it costs far
    more than the tile itself.  Underestimating it lets ``plan_concurrency``
    admit more workers than fit, and the run ends up in the allocator's
    free-and-retry path instead of doing work.
    """
    data = n_values * 4
    labels = (n_values // max(1, tuple_size)) * 8
    planes = n_values * 3  # low plane + high plane + escape mask
    scratch = resid_tile_values * 12 * 4  # ~12 temporaries, f64 counted twice
    return int(data * 2 + labels + planes + dist_buffer + scratch)
