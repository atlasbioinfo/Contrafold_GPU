#!/usr/bin/env python3
"""Generate random RNA sequences for benchmarking gpu_contrafold.

Uniform i.i.d. ACGU. Deterministic given --seed, so the same FASTA can be
regenerated anywhere instead of committing a large data file.

    python benchmarks/generate_sequences.py -n 10000 -l 200 -o benchmarks/data/sim_10k_200nt.fa
"""
import argparse
import os
import numpy as np

BASES = np.array(list("ACGU"))


def generate(n, length, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=(n, length), dtype=np.int8)
    return ["".join(BASES[row]) for row in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num", type=int, default=10000, help="number of sequences")
    ap.add_argument("-l", "--length", type=int, default=200, help="length (nt) of each sequence")
    ap.add_argument("-s", "--seed", type=int, default=0, help="RNG seed (for reproducibility)")
    ap.add_argument("-o", "--out", default="benchmarks/data/sim_10k_200nt.fa", help="output FASTA path")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    seqs = generate(args.num, args.length, args.seed)
    with open(args.out, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">sim_{i}\n{s}\n")
    print(f"wrote {args.num} x {args.length} nt sequences (seed={args.seed}) -> {args.out}")


if __name__ == "__main__":
    main()
