#!/usr/bin/env python3
"""The thread-per-sequence path must be identical to the block-per-sequence one.

test_validation.py checks gpu_contrafold against the original CONTRAfold binary,
which most machines will not have. This file checks the claim that actually
matters for the _tps kernels -- that they reproduce the already-validated
mea_gpu/bpp_gpu exactly -- and needs nothing but a GPU.

The claim is stronger than "close": post_kernel_tps runs the same serial
recurrence per thread as post_kernel, with no atomics and no cross-thread
reduction, so the posterior should be bitwise equal, not merely within
tolerance. A nonzero difference here means the lane mapping or the
sequence-last layout introduced a race.

Usage:
  python tests/test_tps.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_contrafold as gc
from gpu_contrafold import gpu_mea as gm


def rand_seq(rng, L):
    return "".join(rng.choice(list("ACGU"), L))


def main():
    rng = np.random.default_rng(0)
    P = gc.load()
    ok = True

    for L, n in [(60, 64), (120, 64), (200, 32)]:
        seq = rand_seq(rng, L)
        masks = [(rng.random(L) < 0.05).astype(np.int8) for _ in range(n)]
        a = gm.mea_gpu([seq] * n, P, gamma=6.0, forced_list=masks)
        b = gm.mea_gpu_tps([seq] * n, P, gamma=6.0, forced_list=masks)
        same = sum(x == y for x, y in zip(a, b))
        ok &= same == n
        print(f"[1] MEA L={L:3d} n={n:3d}: {same}/{n} identical  "
              f"{'PASS' if same == n else 'FAIL'}")

    seq = rand_seq(rng, 90)
    masks = [(rng.random(90) < 0.05).astype(np.int8) for _ in range(32)]
    A = gm.bpp_gpu([seq] * 32, P, forced_list=masks)
    B = gm.bpp_gpu_tps([seq] * 32, P, forced_list=masks)
    d = max(float(np.abs(x - y).max()) for x, y in zip(A, B))
    ok &= d == 0.0
    print(f"[2] posterior max |diff| = {d:.3e}  "
          f"{'PASS (bitwise)' if d == 0.0 else 'FAIL (expected bitwise equality)'}")

    seq = rand_seq(rng, 100)
    ms = [(rng.random(100) < 0.15).astype(np.int8) for _ in range(32)]
    viol = sum(1
               for db, m in zip(gm.mea_gpu_tps([seq] * 32, P, gamma=6.0,
                                               forced_list=ms), ms)
               for i, c in enumerate(db) if m[i] == 1 and c != ".")
    ok &= viol == 0
    print(f"[3] forced-unpaired violations: {viol}  {'PASS' if not viol else 'FAIL'}")

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
