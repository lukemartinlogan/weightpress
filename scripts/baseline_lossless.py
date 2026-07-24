#!/usr/bin/env python3
"""What plain lossless compression gets on the same checkpoints.

Context for the weightpress ratios: if zstd alone already gets most of the way,
the lossy pipeline is not earning its complexity.  Reported against the source
bytes (fp16/bf16 checkpoints are not widened here).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weightpress.reader import read_safetensors_header

MB = 1 << 20


def zstd_ratio(path: str, limit: int, level: int) -> dict:
    """zstd over the raw tensor bytes, and over a byte-plane-split version."""
    _header, data_start = read_safetensors_header(path)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    raw = bytes(mm[data_start : data_start + limit])
    cctx = zstd.ZstdCompressor(level=level)

    t0 = time.time()
    plain = len(cctx.compress(raw))
    t_plain = time.time() - t0

    # Split fp32 into 4 byte planes: sign/exponent bytes are far more regular
    # than the mantissa, and separating them is the standard trick.
    arr = np.frombuffer(raw, dtype=np.uint8)
    n = (arr.size // 4) * 4
    planes = arr[:n].reshape(-1, 4)
    split = sum(len(cctx.compress(np.ascontiguousarray(planes[:, i]).tobytes()))
                for i in range(4))
    return {
        "source_bytes": len(raw),
        "zstd_bytes": plain,
        "zstd_ratio": len(raw) / plain,
        "zstd_split_bytes": split,
        "zstd_split_ratio": len(raw) / split,
        "seconds": t_plain,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="/home/iowarp/wp-models")
    ap.add_argument("--limit", type=int, default=256 * MB)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--out", default="/home/iowarp/wp-results/lossless.json")
    args = ap.parse_args()

    rows = {}
    for name in ("gpt2", "gpt2-medium", "tinyllama"):
        p = os.path.join(args.models_dir, f"{name}.safetensors")
        if not os.path.exists(p):
            continue
        r = zstd_ratio(p, args.limit, args.level)
        rows[name] = r
        print(f"  {name:14s} zstd {r['zstd_ratio']:.3f}x   "
              f"byte-split zstd {r['zstd_split_ratio']:.3f}x   "
              f"({r['source_bytes']/MB:.0f} MiB in {r['seconds']:.1f}s)")
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
