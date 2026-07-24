"""End-to-end pipeline tests.  These need torch; CUDA is used when present."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from weightpress import kmeans as km
from weightpress.codec import unpack_chunk
from weightpress.config import Config
from weightpress.gpu import estimate_window_bytes, plan_concurrency, resolve_device
from weightpress.pipeline import compress, compress_window, decompress

DEVICE = resolve_device("cuda")
DEVICES = ["cpu"] + (["cuda"] if DEVICE == "cuda" else [])


def _weights(n, seed=0):
    """Gaussian weights with a few heavy outliers, like a real tensor."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.02, size=n).astype(np.float32)
    x[rng.choice(n, size=max(1, n // 10_000), replace=False)] *= 500
    return x


def _gen(device):
    """A seeded generator so k-means (and thus the k search) is reproducible."""
    return torch.Generator(device=device).manual_seed(0)


def _max_error(got, x, mode):
    """Absolute or relative max error, matching how the bound is enforced."""
    a = np.abs(got.astype(np.float64) - x.astype(np.float64))
    if mode == "relative":
        denom = np.abs(x.astype(np.float64))
        a = np.divide(a, denom, out=np.zeros_like(a), where=denom != 0)
    return float(a.max())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["relative", "absolute"])
def test_window_roundtrip_holds_the_bound(device, mode):
    cfg = Config(error_bound=1e-4, error_mode=mode, window_size=1 << 20,
                 max_k=256, device=device)
    x = _weights(1 << 18)
    chunk, stats = compress_window(0, x, cfg, device)
    back = unpack_chunk(chunk)
    assert back.size == x.size
    assert _max_error(back, x, mode) <= cfg.error_bound
    assert stats.max_error <= cfg.error_bound


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["relative", "absolute"])
@pytest.mark.parametrize("tuple_size", [1, 2, 4])
def test_tuple_sizes_roundtrip(device, mode, tuple_size):
    cfg = Config(error_bound=1e-4, error_mode=mode, max_k=128,
                 tuple_size=tuple_size, device=device)
    x = _weights(100_003, seed=tuple_size)  # prime length exercises padding
    chunk, _ = compress_window(0, x, cfg, device)
    back = unpack_chunk(chunk)
    assert back.size == x.size
    assert _max_error(back, x, mode) <= cfg.error_bound


@pytest.mark.parametrize("device", DEVICES)
def test_relative_bound_is_scale_invariant(device):
    """The whole point of a relative bound: the same tensor scaled by 1e6 must
    reconstruct to the same relative error and the same bitstream size."""
    cfg = Config(error_bound=1e-4, error_mode="relative", window_size=1 << 20,
                 max_k=64, device=device)
    x = _weights(1 << 18, seed=3)
    c_small, s_small = compress_window(0, x, cfg, device)
    c_big, s_big = compress_window(0, (x * 1e6).astype(np.float32), cfg, device)
    assert _max_error(unpack_chunk(c_small), x, "relative") <= 1e-4
    assert _max_error(unpack_chunk(c_big), x * 1e6, "relative") <= 1e-4
    # Log-domain coding makes the two costs essentially identical.
    assert abs(s_small.stored_bytes - s_big.stored_bytes) < 0.02 * s_small.stored_bytes


@pytest.mark.parametrize("device", DEVICES)
def test_relative_signs_and_zeros_survive(device):
    cfg = Config(error_bound=1e-4, error_mode="relative", window_size=1 << 20,
                 max_k=64, device=device)
    rng = np.random.default_rng(9)
    x = rng.normal(0, 0.02, size=40_000).astype(np.float32)
    x[::7] *= -1                       # plenty of negatives
    x[rng.choice(x.size, 200, replace=False)] = 0.0  # exact zeros escape
    chunk, _ = compress_window(0, x, cfg, device)
    back = unpack_chunk(chunk)
    assert np.array_equal(np.sign(back), np.sign(x))
    assert np.array_equal(back == 0, x == 0)
    assert _max_error(back, x, "relative") <= 1e-4


@pytest.mark.parametrize("device", DEVICES)
def test_k_search_vq_criterion_doubles_up_to_max_k(device):
    cfg = Config(error_bound=1e-4, k_criterion="vq", max_k=1024, device=device)
    x2d = torch.from_numpy(_weights(1 << 16).reshape(-1, 2)).to(device)
    res = km.search_k(x2d, cfg, generator=_gen(device))
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
    res = km.search_k(x2d, cfg, generator=_gen(device))
    first, last = res.trials[0], res.trials[-1]
    mean_gain = first.vq_mean_abs_error / last.vq_mean_abs_error
    max_gain = first.vq_max_error / last.vq_max_error
    assert mean_gain > 2.0, "16x more centroids must cut the mean error"
    assert max_gain < 2.0, "...while the max error barely moves"
    # Even at max_k the max error is orders of magnitude above any usable bound.
    assert last.vq_max_error > 1000 * cfg.error_bound


@pytest.mark.parametrize("device", DEVICES)
def test_k_search_size_criterion_stops_early(device):
    cfg = Config(error_bound=1e-4, k_criterion="size", max_k=4096, device=device)
    x2d = torch.from_numpy(_weights(1 << 16).reshape(-1, 2)).to(device)
    res = km.search_k(x2d, cfg, generator=_gen(device))
    assert res.evaluation.k <= 4096
    best = min(t.est_bytes for t in res.trials)
    assert res.evaluation.est_bytes == best, "must keep the cheapest trial"


@pytest.mark.parametrize("device", DEVICES)
def test_size_criterion_prices_k1_against_the_doubling_sequence(device):
    """Adjacent weights are near-uncorrelated, so a label costs more than the
    sharper prediction saves and k=1 wins.  The search must be able to see it."""
    cfg = Config(error_bound=1e-4, k_criterion="size", max_k=256, device=device)
    x2d = torch.from_numpy(_weights(1 << 16).reshape(-1, 2)).to(device)
    res = km.search_k(x2d, cfg, generator=_gen(device))
    assert 1 in [t.k for t in res.trials], "k=1 must be among the candidates"
    assert res.evaluation.k == 1
    assert res.evaluation.label_entropy_bits == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_pinning_k_start_to_max_k_disables_the_k1_probe(device):
    cfg = Config(k_start=64, max_k=64, k_criterion="size", device=device)
    x2d = torch.from_numpy(_weights(1 << 16).reshape(-1, 2)).to(device)
    res = km.search_k(x2d, cfg, generator=_gen(device))
    assert [t.k for t in res.trials] == [64], "a pinned k must be used verbatim"
    assert res.evaluation.k == 64


@pytest.mark.parametrize("device", DEVICES)
def test_k_equals_one_is_the_no_predictor_baseline(device):
    cfg = Config(k_start=1, max_k=1, device=device)
    x2d = torch.from_numpy(_weights(1 << 16).reshape(-1, 2)).to(device)
    res = km.search_k(x2d, cfg, generator=_gen(device))
    assert res.centroids.shape[0] == 1
    assert res.evaluation.label_entropy_bits == 0.0


@pytest.mark.parametrize("mode", ["relative", "absolute"])
def test_full_compress_decompress_roundtrip(tmp_path, mode):
    x = _weights(1 << 20, seed=7)
    src = str(tmp_path / "w.npy")
    np.save(src, x)
    cfg = Config(error_bound=1e-4, error_mode=mode, window_size=1 << 20,
                 max_k=256, output_dir=str(tmp_path), device=DEVICE)
    path, stats = compress(src, cfg)

    assert len(stats.chunks) == 4, "1 MiB windows over a 4 MiB stream"
    assert stats.max_error <= cfg.error_bound
    back, header = decompress(path)
    assert back.size == x.size
    assert _max_error(back, x, mode) <= cfg.error_bound
    assert header["tuple_size"] == 2
    assert header["error_mode"] == mode

    tables = tmp_path / "w.kmeans"
    assert len(list(tables.iterdir())) == len(stats.chunks)


def test_vq_mode_is_lossy_beyond_the_bound_but_much_smaller(tmp_path):
    x = _weights(1 << 19, seed=11)
    src = str(tmp_path / "w.npy")
    np.save(src, x)
    common = dict(window_size=1 << 21, k_start=256, max_k=256,
                  error_mode="absolute", output_dir=str(tmp_path), device=DEVICE)
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


def test_window_estimate_covers_the_observed_footprint():
    """A 128 MiB fp32 window measured ~1.5 GiB peak per worker on the RTX 5080.
    The estimate must not come in under that, or plan_concurrency oversubscribes
    the device and the run stalls in the allocator."""
    est = estimate_window_bytes(
        n_values=(128 << 20) // 4, tuple_size=2,
        dist_buffer=km.DIST_BUFFER_BYTES, resid_tile_values=km.RESID_TILE_VALUES,
    )
    assert est >= 1 << 30, f"estimate {est/2**30:.2f} GiB is too optimistic"
