#!/usr/bin/env python3
"""Benchmark study for gpu_contrafold: writes Benchmark.md.

Three parts:
  1. Batch-size sweep  - GPU logZ throughput vs --chunk (sequences per launch),
     to find the batch size that best saturates the GPU.
  2. Length sweep      - GPU logZ throughput vs sequence length (fixed batch).
  3. Device comparison - wall time to fold the whole dataset on GPU, single-core
     CPU, and multi-core CPU.

Every timed configuration is repeated (--repeat) and reported as mean +/- std.

    python benchmarks/compare.py benchmarks/data/sim_10k_200nt.fa --repeat 3 --out Benchmark.md
"""
import argparse
import os
import platform
import statistics
import sys
import time
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gpu_contrafold import load, logZ_batch, cpu          # noqa: E402
from benchmarks.generate_sequences import generate         # noqa: E402


def read_fasta(path, limit=None):
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line[0] == ">":
                if cur:
                    seqs.append("".join(cur)); cur = []
                if limit and len(seqs) >= limit:
                    return seqs
            else:
                cur.append(line)
    if cur and (not limit or len(seqs) < limit):
        seqs.append("".join(cur))
    return seqs


def mean_std(xs):
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, sd


def timed(fn, repeat):
    """Run fn() `repeat` times, return list of wall-clock seconds."""
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


# ---------------- GPU ----------------
def gpu_fold_all(seqs, P, chunk, threads=128):
    for c0 in range(0, len(seqs), chunk):
        logZ_batch(seqs[c0:c0 + chunk], P, threads=threads)   # syncs internally


# ---------------- CPU single-core ----------------
def cpu1_fold_all(seqs, P):
    for s in seqs:
        cpu.logZ(s, P)


# ---------------- CPU multi-core (spawn-safe, module-level) ----------------
_WP = None


def _winit():
    global _WP
    _WP = cpu.load()
    cpu.logZ("GGGGAAAACCCC", _WP)   # warm JIT (loads numba disk cache)


def _wfold(chunk):
    for s in chunk:
        cpu.logZ(s, _WP)
    return len(chunk)


def split(lst, nparts):
    k = (len(lst) + nparts - 1) // nparts
    return [lst[i:i + k] for i in range(0, len(lst), k)]


def gpu_name():
    try:
        from numba import cuda
        return cuda.get_current_device().name.decode()
    except Exception:
        return "unknown GPU"


def cpu_name():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown CPU"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fasta", help="input FASTA")
    ap.add_argument("--out", default="Benchmark.md", help="markdown report output")
    ap.add_argument("--repeat", type=int, default=3, help="repeats per configuration")
    ap.add_argument("--limit", type=int, default=None, help="use only first N sequences")
    ap.add_argument("--threads", type=int, default=128, help="GPU threads/block")
    ap.add_argument("--chunks", default="256,512,1024,2048,4096,8192,10000",
                    help="batch sizes for the sweep")
    ap.add_argument("--lengths", default="50,100,200,300,400,500",
                    help="lengths for the length sweep")
    ap.add_argument("--len-n", type=int, default=2048, help="sequences per length-sweep point")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="multi-core CPU workers")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-length", action="store_true")
    ap.add_argument("--skip-compare", action="store_true")
    args = ap.parse_args()

    seqs = read_fasta(args.fasta, args.limit)
    n = len(seqs)
    L = len(seqs[0])
    print(f"loaded {n} sequences x {L} nt | GPU repeat={args.repeat} workers={args.workers}")
    P = load()
    logZ_batch(seqs[:4], P, threads=args.threads)   # GPU warmup/compile
    gname, cname = gpu_name(), cpu_name()

    report = []
    report.append("# Benchmark\n")
    report.append(f"`gpu_contrafold` partition-function (logZ) throughput. "
                  f"Each configuration is the wall-clock time to fold the dataset, "
                  f"reported as **mean ± std over {args.repeat} runs**.\n")
    report.append("## Environment\n")
    report.append(f"- GPU: {gname}")
    report.append(f"- CPU: {cname} ({os.cpu_count()} logical cores)")
    report.append(f"- Dataset: {n} random RNA sequences x {L} nt (uniform ACGU)")
    report.append(f"- numba {__import__('numba').__version__}, numpy {__import__('numpy').__version__}, "
                  f"Python {platform.python_version()}\n")

    best_chunk = 4096

    # ---------- Phase 1: batch-size sweep ----------
    if not args.skip_sweep:
        chunks = [int(c) for c in args.chunks.split(",")]
        print("\n== batch-size sweep (GPU, logZ) ==")
        rows, best_fps = [], -1.0
        for ch in chunks:
            ts = timed(lambda ch=ch: gpu_fold_all(seqs, P, ch, args.threads), args.repeat)
            m, sd = mean_std(ts)
            fps = n / m
            mem = 3 * (L + 2) ** 2 * 4 * ch / 1e9
            rows.append((ch, m, sd, fps, mem))
            if fps > best_fps:
                best_fps, best_chunk = fps, ch
            print(f"  chunk={ch:6d}  {m:6.3f}±{sd:.3f}s  {fps:9,.0f} folds/s  ~{mem:.1f} GB")
        knee = min((c for c, _, _, f, _ in rows if f >= 0.9 * best_fps), default=best_chunk)
        knee_mem = 3 * (L + 2) ** 2 * 4 * knee / 1e9
        report.append(f"## 1. Batch-size sweep (GPU, {L} nt, {n} seqs)\n")
        report.append("How many sequences per GPU launch (`--chunk`) best saturates the GPU. "
                      "Throughput rises with batch size until the GPU is full, then plateaus; "
                      "memory grows linearly (`~3·(L+2)²·4 B` per sequence).\n")
        report.append("| batch (chunk) | time (s) | folds/s | ~GPU mem |")
        report.append("|---:|---:|---:|---:|")
        for ch, m, sd, fps, mem in rows:
            mark = "  **(peak)**" if ch == best_chunk else ""
            report.append(f"| {ch:,}{mark} | {m:.3f} ± {sd:.3f} | {fps:,.0f} | {mem:.1f} GB |")
        report.append("")
        report.append(f"Peak ≈ **{best_fps:,.0f} folds/s at batch {best_chunk:,}** ({L} nt). "
                      f"Throughput reaches ~90% of peak by batch **{knee:,}** (~{knee_mem:.1f} GB); "
                      f"larger batches add only a few % for proportionally more memory. "
                      f"**Recommended batch: {knee:,}–{best_chunk:,}** — use the lower end on "
                      f"memory-limited GPUs, the higher end when VRAM is free. Scale inversely with "
                      f"length, since memory per sequence grows as `(L+2)²`.\n")
        print(f"  -> peak chunk = {best_chunk}, knee (90% peak) = {knee}")

    # ---------- Phase 2: length sweep ----------
    if not args.skip_length:
        lengths = [int(x) for x in args.lengths.split(",")]
        print("\n== length sweep (GPU, logZ) ==")
        rows = []
        for Lx in lengths:
            sx = generate(args.len_n, Lx, seed=1)
            ch = min(best_chunk, args.len_n)
            logZ_batch(sx[:4], P, threads=args.threads)   # warmup for this length
            ts = timed(lambda sx=sx, ch=ch: gpu_fold_all(sx, P, ch, args.threads), args.repeat)
            m, sd = mean_std(ts)
            fps = args.len_n / m
            rows.append((Lx, m, sd, fps))
            print(f"  L={Lx:4d}  {m:6.3f}±{sd:.3f}s  {fps:9,.0f} folds/s")
        report.append(f"## 2. Length sweep (GPU, batch={min(best_chunk, args.len_n):,}, "
                      f"{args.len_n} seqs each)\n")
        report.append("Throughput vs sequence length. The partition function is O(L³) time / "
                      "O(L²) memory, so folds/s falls steeply as sequences get longer.\n")
        report.append("| length (nt) | time (s) | folds/s |")
        report.append("|---:|---:|---:|")
        for Lx, m, sd, fps in rows:
            report.append(f"| {Lx} | {m:.3f} ± {sd:.3f} | {fps:,.0f} |")
        report.append("")

    # ---------- Phase 3: device comparison ----------
    if not args.skip_compare:
        print("\n== device comparison (fold all, logZ) ==")
        # GPU
        gpu_ts = timed(lambda: gpu_fold_all(seqs, P, best_chunk, args.threads), args.repeat)
        gm, gsd = mean_std(gpu_ts)
        print(f"  GPU (batch {best_chunk}):   {gm:7.3f}±{gsd:.3f}s  {n/gm:,.0f} folds/s")

        # CPU single-core
        cpu.logZ(seqs[0], P)   # warmup
        c1_ts = timed(lambda: cpu1_fold_all(seqs, P), args.repeat)
        c1m, c1sd = mean_std(c1_ts)
        print(f"  CPU 1-core:          {c1m:7.3f}±{c1sd:.3f}s  {n/c1m:,.0f} folds/s")

        # CPU multi-core
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(args.workers, initializer=_winit)
        parts = split(seqs, args.workers * 4)
        pool.map(_wfold, split(seqs[:max(args.workers, 8)], args.workers))  # warm all workers
        cN_ts = timed(lambda: pool.map(_wfold, parts), args.repeat)
        pool.close(); pool.join()
        cNm, cNsd = mean_std(cN_ts)
        print(f"  CPU {args.workers}-core:         {cNm:7.3f}±{cNsd:.3f}s  {n/cNm:,.0f} folds/s")

        report.append(f"## 3. GPU vs CPU ({n} sequences x {L} nt)\n")
        report.append(f"Wall-clock time to compute logZ for all {n} sequences.\n")
        report.append(f"| device | time (s) | folds/s | speedup vs 1-core |")
        report.append("|:--|---:|---:|---:|")
        report.append(f"| {gname} (GPU, batch {best_chunk:,}) | "
                      f"{gm:.3f} ± {gsd:.3f} | {n/gm:,.0f} | {c1m/gm:,.1f}× |")
        report.append(f"| CPU 1 core | {c1m:.3f} ± {c1sd:.3f} | {n/c1m:,.0f} | 1× (baseline) |")
        report.append(f"| CPU {args.workers} cores | {cNm:.3f} ± {cNsd:.3f} | {n/cNm:,.0f} | "
                      f"{c1m/cNm:,.1f}× |")
        report.append("")
        report.append(f"GPU is **{c1m/gm:,.1f}× a single CPU core** and "
                      f"**{cNm/gm:,.1f}× the full {args.workers}-thread CPU** on this workload.\n")
        report.append(f"> **Baseline note:** the CPU figures are this package's own NumPy/Numba "
                      f"`cpu.logZ` reference (~{c1m / n * 1e3:.0f} ms/fold single-core), *not* the "
                      f"optimized CONTRAfold C++ binary, which is considerably faster. These "
                      f"multiples are GPU vs. this reference implementation; the speedup over the "
                      f"C++ binary would be smaller. Multi-core scaling is sublinear "
                      f"({c1m / cNm:,.1f}× on {args.workers} threads) because the i9-14900F mixes "
                      f"performance and efficiency cores with hyper-threading.\n")

    report.append("---\n")
    report.append("Reproduce:\n")
    report.append("```bash")
    report.append(f"python benchmarks/generate_sequences.py -n {n} -l {L} "
                  f"-o benchmarks/data/sim_{n//1000}k_{L}nt.fa")
    report.append(f"python benchmarks/compare.py benchmarks/data/sim_{n//1000}k_{L}nt.fa "
                  f"--repeat {args.repeat}")
    report.append("```")

    with open(args.out, "w") as fh:
        fh.write("\n".join(report) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
