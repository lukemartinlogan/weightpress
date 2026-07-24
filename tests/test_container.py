"""Container format and reader tests (no GPU required)."""

import os

import numpy as np
import pytest

from weightpress.codec import pack_chunk, quantize, unpack_chunk
from weightpress.container import ContainerReader, ContainerWriter, write_codebook_sidecar
from weightpress.reader import WeightStream, restore_tensors


def _make_chunk(index, n_values=1000, k=64, tuple_size=2, eb=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    n_tuples = -(-n_values // tuple_size)
    x = rng.normal(0, 0.02, size=n_tuples * tuple_size).astype(np.float32)
    centroids = rng.normal(0, 0.02, size=(k, tuple_size)).astype(np.float32)
    labels = rng.integers(0, k, size=n_tuples).astype(np.int64)
    z, oidx, ovals = quantize(x, centroids[labels].reshape(-1), eb)
    chunk = pack_chunk(index, centroids, labels, z, oidx, ovals,
                       n_values=n_values, tuple_size=tuple_size, mode="residual")
    return chunk, x[:n_values]


def test_container_roundtrip(tmp_path):
    path = str(tmp_path / "t.wp")
    originals = []
    with ContainerWriter(path, {"error_bound": 1e-4}) as w:
        for i in range(5):
            chunk, x = _make_chunk(i, seed=i)
            w.add(chunk)
            originals.append(x)

    with ContainerReader(path) as r:
        assert r.header["error_bound"] == 1e-4
        assert len(r.chunks) == 5
        for i, x in enumerate(originals):
            back = unpack_chunk(r.read_chunk(i), 1e-4)
            assert np.abs(back - x).max() <= 1e-4


def test_container_rejects_garbage(tmp_path):
    path = str(tmp_path / "bad.wp")
    with open(path, "wb") as fh:
        fh.write(b"NOPE" + b"\x00" * 64)
    with pytest.raises(ValueError):
        ContainerReader(path)


def test_container_detects_truncation(tmp_path):
    path = str(tmp_path / "t.wp")
    with ContainerWriter(path, {"error_bound": 1e-4}) as w:
        w.add(_make_chunk(0)[0])
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - 4)
    with pytest.raises(ValueError):
        ContainerReader(path)


def test_codebook_sidecars_are_written_per_chunk(tmp_path):
    cent = np.arange(128, dtype=np.float32).reshape(64, 2)
    for i in range(3):
        write_codebook_sidecar(str(tmp_path), "model", i, cent, {"k": 64})
    d = tmp_path / "model.kmeans"
    files = sorted(p.name for p in d.iterdir())
    assert files == ["chunk_000000.npz", "chunk_000001.npz", "chunk_000002.npz"]
    loaded = np.load(d / "chunk_000001.npz")
    assert np.array_equal(loaded["centroids"], cent)


def test_npy_stream_windows_cover_every_value(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.normal(size=10_000).astype(np.float32)
    p = str(tmp_path / "a.npy")
    np.save(p, arr)
    s = WeightStream.open(p)
    got = np.concatenate([w for _, w in s.windows(3000)])
    assert np.allclose(got, arr)
    assert s.source_bytes == arr.nbytes


def test_raw_stream_respects_limit_bytes(tmp_path):
    arr = np.arange(10_000, dtype=np.float32)
    p = str(tmp_path / "a.bin")
    arr.tofile(p)
    s = WeightStream.open(p, dtype="float32", limit_bytes=4 * 1000)
    got = np.concatenate([w for _, w in s.windows(256)])
    assert got.size == 1000
    assert np.allclose(got, arr[:1000])


def test_windows_are_exactly_sized_except_the_last(tmp_path):
    arr = np.arange(2500, dtype=np.float32)
    p = str(tmp_path / "a.npy")
    np.save(p, arr)
    sizes = [w.size for _, w in WeightStream.open(p).windows(1000)]
    assert sizes == [1000, 1000, 500]


def test_restore_tensors_slices_the_stream_back():
    manifest = {"tensors": [
        {"name": "a", "offset": 0, "numel": 6, "shape": [2, 3]},
        {"name": "b", "offset": 6, "numel": 4, "shape": [4]},
    ]}
    vals = np.arange(10, dtype=np.float32)
    out = restore_tensors(vals, manifest)
    assert out["a"].shape == (2, 3) and out["b"].shape == (4,)
    assert np.array_equal(out["b"], np.arange(6, 10, dtype=np.float32))
