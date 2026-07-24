"""Codec-level tests: these run without a GPU."""

import numpy as np
import pytest

from weightpress.codec import (
    dequantize,
    labels_dtype,
    pack_chunk,
    quantize,
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
    z, oidx, ovals = quantize(x, pred, eb)
    back = dequantize(z, pred, eb, oidx, ovals)
    assert np.abs(back - x).max() <= eb


def test_outliers_reconstruct_exactly():
    eb = 1e-4
    x = np.array([0.0, 1e9, -1e9, 0.5], dtype=np.float32)
    pred = np.zeros(4, dtype=np.float32)
    z, oidx, ovals = quantize(x, pred, eb)
    # 1e9 / 2e-4 blows past the 16-bit code, so those two must escape.
    assert set(oidx.tolist()) == {1, 2}
    back = dequantize(z, pred, eb, oidx, ovals)
    assert back[1] == x[1] and back[2] == x[2]
    assert np.abs(back - x).max() <= eb


def test_nan_and_inf_escape_rather_than_corrupt():
    eb = 1e-4
    x = np.array([np.nan, np.inf, -np.inf, 0.25], dtype=np.float32)
    pred = np.zeros(4, dtype=np.float32)
    z, oidx, ovals = quantize(x, pred, eb)
    assert set(oidx.tolist()) == {0, 1, 2}
    back = dequantize(z, pred, eb, oidx, ovals)
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
        z, oidx, ovals = quantize(padded, pred, eb)
    else:
        z, oidx, ovals = None, np.empty(0, np.uint32), np.empty(0, np.float32)
    chunk = pack_chunk(
        0, centroids, labels, z, oidx, ovals,
        n_values=x.size, tuple_size=tuple_size, mode=mode,
    )
    return chunk, unpack_chunk(chunk, eb)


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
