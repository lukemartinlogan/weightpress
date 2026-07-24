"""Codec-level tests: these run without a GPU."""

import numpy as np
import pytest

from weightpress.codec import (
    dequantize,
    join_planes,
    labels_dtype,
    pack_chunk,
    planes_needed,
    quantize,
    split_planes,
    unpack_chunk,
    zigzag_decode,
    zigzag_encode,
)
from weightpress.config import CODE_MAX


def test_zigzag_roundtrip():
    codes = np.array([0, 1, -1, 2, -2, CODE_MAX, -CODE_MAX], dtype=np.int32)
    assert np.array_equal(zigzag_decode(zigzag_encode(codes)), codes)


def test_zigzag_keeps_small_residuals_in_the_low_byte():
    small = np.arange(-100, 100, dtype=np.int32)
    z = zigzag_encode(small)
    assert (z >> 8).max() == 0, "small residuals must not touch the high plane"


@pytest.mark.parametrize("eb", [1e-2, 1e-4, 1e-6])
def test_quantize_respects_the_error_bound(eb):
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.02, size=100_000).astype(np.float32)
    pred = (rng.normal(0, 0.02, size=100_000)).astype(np.float32)
    z, oidx, ovals, step = quantize(x, pred, eb)
    back = dequantize(z, pred, step, oidx, ovals)
    assert np.abs(back - x).max() <= eb


@pytest.mark.parametrize("max_z,expect", [
    (0, 1), (255, 1), (256, 2), (65535, 2), (65536, 3),
    (1 << 24, 4), (1 << 31, 4),
])
def test_planes_needed_picks_the_narrowest_width(max_z, expect):
    assert planes_needed(max_z) == expect


def test_plane_split_join_roundtrip():
    z = np.array([0, 1, 255, 256, 70000, (1 << 24) + 5], dtype=np.uint32)
    for n in (planes_needed(int(z.max())), 4):
        assert np.array_equal(join_planes(split_planes(z, n)), z)


def _mixed_magnitude_weights():
    """Gaussian weights, LayerNorm-like weights near 1.0, and 0/1 mask values --
    the magnitude spread that broke the fixed-width code."""
    rng = np.random.default_rng(0)
    return np.concatenate([
        rng.normal(0, 0.02, size=50_000),
        rng.normal(1.0, 0.01, size=50_000),
        rng.integers(0, 2, size=50_000),
    ]).astype(np.float32)


@pytest.mark.parametrize("eb,want_planes", [(1e-4, 2), (1e-5, 2), (1e-6, 3)])
def test_wide_residuals_widen_the_code_instead_of_escaping(eb, want_planes):
    """The regression that motivated adaptive widths: with a fixed 16-bit code,
    eb=1e-5 caps residuals at +/-0.65, so gpt2's LayerNorm weights and attention
    masks (all near 1.0) escaped -- 4.2M values, a third of the output.

    The tight bounds here also exercise the magnitude-scaled step: float32
    rounding at x~1.0 is ~6e-8, so a margin proportional to eb alone would send
    these to the escape list instead."""
    x = _mixed_magnitude_weights()
    centroid = np.float32(x.mean())
    pred = np.full(x.size, centroid, dtype=np.float32)

    z, oidx, ovals, step = quantize(x, pred, eb)
    assert oidx.size == 0, f"{oidx.size} values escaped; the code should widen"
    assert planes_needed(int(z.max())) == want_planes

    chunk = pack_chunk(0, np.array([[centroid]], np.float32),
                       np.zeros(x.size, np.int64), z, oidx, ovals,
                       n_values=x.size, tuple_size=1, mode="residual", step=step)
    assert chunk.n_planes == want_planes
    assert np.abs(unpack_chunk(chunk) - x).max() <= eb


def test_bound_below_the_float32_ulp_escapes_rather_than_violates():
    """At x~1.0 a float32 ulp is ~6e-8, so a 1e-8 bound cannot be met by any
    arithmetic in this pipeline.  The contract is that such values go to the
    escape list and reconstruct exactly -- never that the bound is quietly
    broken."""
    eb = 1e-8
    x = _mixed_magnitude_weights()
    pred = np.full(x.size, np.float32(x.mean()), dtype=np.float32)
    z, oidx, ovals, step = quantize(x, pred, eb)
    assert oidx.size > 0, "sub-ulp bounds must escape"
    back = dequantize(z, pred, step, oidx, ovals)
    assert np.abs(back - x).max() <= eb


def test_narrow_residuals_still_use_two_planes():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.02, size=50_000).astype(np.float32)
    pred = np.zeros(x.size, dtype=np.float32)
    z, _, _, _ = quantize(x, pred, 1e-4)
    assert planes_needed(int(z.max())) == 2, "typical weights must not widen"


def test_outliers_reconstruct_exactly():
    eb = 1e-4
    x = np.array([0.0, 1e9, -1e9, 0.5], dtype=np.float32)
    pred = np.zeros(4, dtype=np.float32)
    z, oidx, ovals, step = quantize(x, pred, eb)
    # 1e9 / 2e-4 blows past the 16-bit code, so those two must escape.
    assert set(oidx.tolist()) == {1, 2}
    back = dequantize(z, pred, step, oidx, ovals)
    assert back[1] == x[1] and back[2] == x[2]
    assert np.abs(back - x).max() <= eb


def test_nan_and_inf_escape_rather_than_corrupt():
    eb = 1e-4
    x = np.array([np.nan, np.inf, -np.inf, 0.25], dtype=np.float32)
    pred = np.zeros(4, dtype=np.float32)
    z, oidx, ovals, step = quantize(x, pred, eb)
    assert set(oidx.tolist()) == {0, 1, 2}
    back = dequantize(z, pred, step, oidx, ovals)
    assert np.isnan(back[0]) and back[1] == np.inf and back[2] == -np.inf
    assert abs(back[3] - x[3]) <= eb


@pytest.mark.parametrize("k,expect", [(64, 1), (256, 1), (257, 2), (65536, 2), (65537, 4)])
def test_labels_dtype_is_the_narrowest_that_fits(k, expect):
    assert labels_dtype(k).itemsize == expect


def _roundtrip(x, k, tuple_size, eb, mode="residual"):
    rng = np.random.default_rng(1)
    n_tuples = -(-x.size // tuple_size)
    padded = np.concatenate([x, np.repeat(x[-1], n_tuples * tuple_size - x.size)])
    centroids = rng.normal(0, 0.02, size=(k, tuple_size)).astype(np.float32)
    labels = rng.integers(0, k, size=n_tuples).astype(np.int64)
    pred = centroids[labels].reshape(-1)
    if mode == "residual":
        z, oidx, ovals, step = quantize(padded, pred, eb)
    else:
        z, oidx, ovals, step = None, np.empty(0, np.uint32), np.empty(0, np.float32), 0.0
    chunk = pack_chunk(
        0, centroids, labels, z, oidx, ovals,
        n_values=x.size, tuple_size=tuple_size, mode=mode, step=step,
    )
    return chunk, unpack_chunk(chunk)


def test_pack_unpack_roundtrip_holds_the_bound():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.02, size=50_000).astype(np.float32)
    chunk, back = _roundtrip(x, 64, 2, 1e-4)
    assert back.size == x.size
    assert np.abs(back - x).max() <= 1e-4


def test_odd_length_window_is_padded_and_trimmed():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.02, size=4097).astype(np.float32)  # not a multiple of 2
    chunk, back = _roundtrip(x, 64, 2, 1e-4)
    assert back.size == 4097
    assert np.abs(back - x).max() <= 1e-4


def test_tuple_size_three_roundtrips():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.02, size=10_001).astype(np.float32)
    chunk, back = _roundtrip(x, 128, 3, 1e-4)
    assert back.size == 10_001
    assert np.abs(back - x).max() <= 1e-4


def test_vq_mode_stores_no_residuals():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.02, size=10_000).astype(np.float32)
    chunk, back = _roundtrip(x, 64, 2, 1e-4, mode="vq")
    assert chunk.code_bytes == 0
    assert back.size == x.size
