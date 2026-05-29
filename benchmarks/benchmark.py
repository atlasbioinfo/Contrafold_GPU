#!/usr/bin/env python3
"""Benchmark gpu_contrafold throughput on a FASTA of RNA sequences.

Measures partition-function (logZ) folds/s on the GPU, chunked to bound memory,
and optionally Boltzmann sampling. Validates a small sample against the CPU
reference (the recurrence is identical; this guards against fp/length bugs).

    python benchmarks/benchmark.py benchmarks/data/sim_10k_200nt.fa
    python benchmarks/benchmark.py <fa> --chunk 4096 --limit 2000 --sample
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_contrafold import load, logZ_batch, sample_batch, cpu  # noqa: E402


def read_fasta(path, limit=None):
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line[0] == ">":
                if cur:
                    seqs.append("".join(cur))
                    cur = []
                if limit and len(seqs) >= limit:
                    return seqs
            else:
                cur.append(line)
    if cur and (not limit or len(seqs) < limit):
        seqs.append("".join(cur))
    return seqs


def fmt_mem():
    try:
        from numba import cuda
        free, total = cuda.current_context().get_memory_info()
        return f"{(total - free) / 1e9:.2f}/{total / 1e9:.2f} GB used"
    except Exception:
        return "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fasta", help="input FASTA")
    ap.add_argument("--chunk", type=int, default=4096, help="sequences per GPU launch")
    ap.add_argument("--limit", type=int, default=None, help="only use first N sequences")
    ap.add_argument("--threads", type=int, default=128, help="threads per block")
    ap.add_argument("--sample", action="store_true", help="also benchmark Boltzmann sampling (1/seq)")
    ap.add_argument("--check", type=int, default=8, help="validate this many seqs vs CPU logZ")
    args = ap.parse_args()

    seqs = read_fasta(args.fasta, args.limit)
    if not seqs:
        sys.exit("no sequences read")
    n = len(seqs)
    lens = [len(s) for s in seqs]
    print(f"loaded {n} sequences, length {min(lens)}-{max(lens)} nt (mean {sum(lens)/n:.0f})")

    P = load()

    # warmup / JIT compile (excluded from timing)
    print("compiling kernels (warmup)...", flush=True)
    _ = logZ_batch(seqs[:2], P, threads=args.threads)

    # ---- logZ throughput ----
    t0 = time.perf_counter()
    done = 0
    for c0 in range(0, n, args.chunk):
        sub = seqs[c0:c0 + args.chunk]
        logZ_batch(sub, P, threads=args.threads)
        done += len(sub)
    dt = time.perf_counter() - t0
    print(f"\n[logZ]  {n} folds in {dt:.2f}s  ->  {n/dt:,.0f} folds/s  ({dt/n*1e3:.2f} ms/fold)")
    print(f"        peak GPU mem: {fmt_mem()}  (chunk={args.chunk})")

    # ---- correctness vs CPU ----
    if args.check:
        k = min(args.check, n)
        gpu = logZ_batch(seqs[:k], P, threads=args.threads)
        maxd = max(abs(g - cpu.logZ(s, P)) for g, s in zip(gpu, seqs[:k]))
        print(f"\n[check] CPU vs GPU logZ over {k} seqs: max |diff| = {maxd:.2e}  "
              f"{'PASS' if maxd < 1e-2 else 'FAIL'}")

    # ---- sampling throughput (optional) ----
    if args.sample:
        t0 = time.perf_counter()
        for c0 in range(0, n, args.chunk):
            sub = seqs[c0:c0 + args.chunk]
            sample_batch(sub, P, 1, threads=args.threads, seed=c0)
        dt = time.perf_counter() - t0
        print(f"\n[sample] {n} folds+1 sample in {dt:.2f}s  ->  {n/dt:,.0f} folds/s")


if __name__ == "__main__":
    main()
