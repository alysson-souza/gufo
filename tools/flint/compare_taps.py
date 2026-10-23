#!/usr/bin/env python3
# /// script
# dependencies = ["numpy"]
# ///
"""Locate the first diverging Flint graph stage between two tap dumps.

Reads the record files written by `flint_reference_probe --taps` and
`llama_logits --taps` (u32 name length, name, i32 layer, u32 position,
u32 count, float32 values). For every (position, layer, name) present in both
with equal width it prints the relative L2 error and max |delta|, in graph
order of the candidate; llama.cpp's last record for a key is its final value.

  uv run tools/flint/compare_taps.py --teacher llama.taps --candidate ref.taps \
      [--position P] [--names attn_norm,l_out]
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def read(path: Path):
    records: dict[tuple[int, int, str], np.ndarray] = {}
    order: list[tuple[int, int, str]] = []
    data = path.read_bytes()
    offset = 0
    while offset < len(data):
        (size,) = struct.unpack_from("<I", data, offset)
        offset += 4
        name = data[offset:offset + size].decode()
        offset += size
        layer, position, count = struct.unpack_from("<iII", data, offset)
        offset += 12
        values = np.frombuffer(data, dtype=np.float32, count=count, offset=offset)
        offset += 4 * count
        key = (position, layer, name)
        if key not in records:
            order.append(key)
        records[key] = values
    return records, order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--position", type=int)
    parser.add_argument("--names", help="comma-separated stage names to show")
    args = parser.parse_args()

    teacher, _ = read(args.teacher)
    candidate, order = read(args.candidate)
    names = set(args.names.split(",")) if args.names else None
    print(f"{'pos':>5} {'layer':>5} {'name':<12} {'rel_l2':>10} {'max_abs':>10} "
          f"{'|cand|max':>10}")
    for key in order:
        position, layer, name = key
        if args.position is not None and position != args.position:
            continue
        if names is not None and name not in names:
            continue
        t = teacher.get(key)
        c = candidate[key]
        if t is None or t.size != c.size:
            continue
        t = t.astype(np.float64)
        c = c.astype(np.float64)
        rel = np.linalg.norm(t - c) / max(np.linalg.norm(c), 1e-30)
        print(f"{position:>5} {layer:>5} {name:<12} {rel:>10.3e} "
              f"{np.abs(t - c).max():>10.3e} {np.abs(c).max():>10.3e}")


if __name__ == "__main__":
    main()
