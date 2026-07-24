"""End-to-end pipeline tests.  These need torch; CUDA is used when present."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from weightpress import kmeans as km  # noqa: E402
from weightpress.codec import unpack_chunk  # noqa: E402
from weightpress.config import Config  # noqa: E402
from weightpress.container import ContainerReader  # noqa: E402
from weightpress.gpu import plan_concurrency, resolve_device  # noqa: E402
from weightpress.pipeline import compress, compress_window, decompress  # noqa: E402

DEVICE = resolve_device("cuda")
DEVICES = ["cpu"] + (["cuda"] if DEVICE == "cuda" else [])


def _weights(n, seed=0):
    """Gaussian weights with a few heavy outliers, like a real tensor."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.02, size=n).astype(np.float32)
    x[rng.choice(n, size=max(1, n // 10_000), replace=False)] *= 500
    return x


@pytest.mark.parametrize("device", DEVICES)
def test_window_roundtrip_holds_the_bound(device):
    cfg = Config(error_bound=1e-4, window_size=1 << 20, max_k=256, device=device)
    x = _weights(1 << 18)
    chunk, stats = compress_window(0, x, cfg, device)
    back = unpack_chunk(chunk, cfg.error_bound)
    assert back.size == x.size
    assert np.abs(back - x).max() <= cfg.error_bound
    assert stats.max_error <= cfg.error_bound


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tuple_size", [1, 2, 4])
def test_tuple_sizes_roundtrip(device, tuple_size):
    cfg = Config(error_bound=1e-4, max_k=128, tuple_size=tuple_size, device=device)
    x = _weights(100_003, seed=tuple_size)  # prime length exercises padding
    chunk, _ = compress_window(0, x, cfg, device)
    back = unpack_chunk(chunk, cfg.error_bound)
    assert back.size == x.size
    assert np.abs(back - x).max() <= cfg.error_bound


@pytest.mark.parametrize("device", DEVICES)
def test_k_search_vq_criterion_doubles_up_to_max_k(device):
    cfg = Config(error_bound=1e-4, k_criterion="vq", max_k=1024, device=device)
    x2d = torch.from_numpy(_weights(1 << 16).reshape(-1, 2)).to(device)
    res = km.search_k(x2d, cfg)
    ks = [t.k for t in res.trials]
    assert ks == [64, 128, 256, 512, 1024], "must double from k_start up to max_k"
    # It runs to max_k because the bound is never met -- see the next test.
    assert res.evaluation.vq_max_error > cfg.error_bound


@pytest.mark.parametrize("device", DEVICES)
def test_more_centroids_cut_mean_error_but_not_max_error(device):
    """The finding that makes the literal 'double until max error fits' rule
    unusable: Lloyd's minimises squared error, so extra centroids track the bulk
    of the distribution.  The *max* error is set by a handful of tail weights
    that stay in a cluster with everything else, so it barely moves with k."""
    cfg = Config(error_bound=1e-4, k_criterion="vq", max_k=1024, device=device)
    x2d = torch.from_numpy(_weights(1 << 16).reshape(-1, 2)).to(device)
    res = km.search_k(x2d, cfg)
    first, last = res.trials[0], res.trials[-1]
    assert last.vq_mean_abs_error < first.vq_mean_abs_error / 2, "mean must improve"
    assert last.vq_max_error > first.vq_max_error * 0.5, "max barely moves"


@pytest.mark.parametrize("device", DEVICES)
def test_k_search_size_criterion_stops_early(device):
    cfg = Config(error_bound=1e-4, k_criterion="size", max_k=4096, device=device)
    x2d = torch.from_numpy(_weights(1 << 16).reshape(-1, 2)).to(device)
    res = km.search_k(x2d, cfg)
    assert res.evaluation.k <= 4096
    best = min(t.est_bytes for t in res.trials)
    assert res.evaluation.est_bytes == best, "must keep the cheapest trial"


@pytest.mark.parametrize("device", DEVICES)
def test_k_equals_one_is_the_no_predictor_baseline(device):
    cfg = Config(k_start=1, max_k=1, device=device)
    x2d = torch.from_numpy(_weights(1 << 16).reshape(-1, 2)).to(device)
    res = km.search_k(x2d, cfg)
    assert res.centroids.shape[0] == 1
    assert res.evaluation.label_entropy_bits == 0.0


def test_full_compress_decompress_roundtrip(tmp_path):
    x = _weights(1 << 20, seed=7)
    src = str(tmp_path / "w.npy")
    np.save(src, x)
    cfg = Config(error_bound=1e-4, window_size=1 << 20, max_k=256,
                 output_dir=str(tmp_path), device=DEVICE)
    path, stats = compress(src, cfg)

    assert len(stats.chunks) == 4, "1 MiB windows over a 4 MiB stream"
    assert stats.max_error <= cfg.error_bound
    back, header = decompress(path)
    assert back.size == x.size
    assert np.abs(back - x).max() <= cfg.error_bound
    assert header["tuple_size"] == 2

    tables = tmp_path / "w.kmeans"
    assert len(list(tables.iterdir())) == len(stats.chunks)


def test_vq_mode_is_lossy_beyond_the_bound_but_much_smaller(tmp_path):
    x = _weights(1 << 19, seed=11)
    src = str(tmp_path / "w.npy")
    np.save(src, x)
    common = dict(window_size=1 << 21, k_start=256, max_k=256,
                  output_dir=str(tmp_path), device=DEVICE)
    _, res_stats = compress(src, Config(mode="residual", **common))
    os.remove(str(tmp_path / "w.wp"))
    _, vq_stats = compress(src, Config(mode="vq", **common))

    # k=256 over 2-tuples is 8 bits per tuple = 4 bits per value, before zstd.
    assert vq_stats.bits_per_value <= 4.5
    assert vq_stats.stored_bytes < res_stats.stored_bytes / 2
    assert vq_stats.max_error > 1e-4, "pure VQ cannot reach the bound at k=256"


def test_concurrency_is_capped_by_the_memory_budget():
    assert plan_concurrency(budget_bytes=1 << 30, per_window_bytes=1 << 28,
                            max_workers=8) == 4
    assert plan_concurrency(budget_bytes=1 << 30, per_window_bytes=1 << 31,
                            max_workers=8) == 1, "always make progress"
    assert plan_concurrency(budget_bytes=1 << 40, per_window_bytes=1 << 20,
                            max_workers=8) == 8, "respect max_workers"
