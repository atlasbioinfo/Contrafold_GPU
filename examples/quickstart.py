#!/usr/bin/env python3
"""gpu-contrafold quickstart: logZ, sampling, and DMS-style hard constraints."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_contrafold import load, logZ_batch, sample_batch, cpu

P = load()

seqs = ["GGGGAAAACCCC", "GCGCGCAAAAGCGCGCAAAAGCGC"]

print("=== log partition function (GPU vs CPU) ===")
for s, z in zip(seqs, logZ_batch(seqs, P)):
    print(f"  {s:26s} GPU={z:.5f}  CPU={cpu.logZ(s, P):.5f}")

print("\n=== Boltzmann samples (5 structures of seq[0]) ===")
for db in sample_batch([seqs[0]], P, 5, seed=0)[0]:
    print("  ", db)

print("\n=== with a hard constraint (force position 3 unpaired) ===")
mask = np.zeros(len(seqs[0]), np.int8)
mask[3] = 1
for db in sample_batch([seqs[0]], P, 5, forced_list=[mask], seed=0)[0]:
    flag = "OK" if db[3] == "." else "VIOLATION"
    print(f"   {db}   pos3={db[3]} {flag}")
