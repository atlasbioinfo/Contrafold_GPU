#!/usr/bin/env python
"""
Benchmark THIS package's pure-Python/Numba CPU engine (gpu_contrafold.cpu)
against the original CONTRAfold 2.02 C++ binary, single core, no GPU.

Question: for one CPU core, is our reimplementation faster or slower than the
original contrafold, and by how much, across task and sequence length?

Matched tasks:
    logz     : cpu.logZ   vs  contrafold predict --partition
    mea      : cpu.mea    vs  contrafold predict            (default = MEA decode)
    viterbi  : cpu.mfe    vs  contrafold predict --viterbi

The CONTRAfold binary folds one independent sequence per process invocation
(a multi-record FASTA is treated as a single alignment, so it cannot batch
independent sequences). We therefore time it per sequence and report:
  * bin_raw  : full per-sequence wall time incl. process spawn + file IO
               (this is the real cost of folding many independent RNAs with it)
  * bin_comp : compute-only, with the measured spawn+IO floor subtracted
               (the binary's pure folding time, a lower bound)

Our engine is timed per sequence after JIT warm-up (the numba kernels are
@njit(cache=True), so a deployed run pays the compile cost only once).

Usage:
    python benchmarks/bench_cpu_vs_binary.py --lengths 50,100,150,200,300 -n 20
"""
import argparse, os, statistics, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_contrafold import cpu

BIN = os.environ.get(
    "GPU_CONTRAFOLD_BIN",
    "/usr/users/JIC_c1/hyu/miniforge3/envs/RNAFOLD/bin/contrafold",
)
BASES = np.array(list("ACGU"))


def gen_seqs(length, n, seed):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(BASES, size=length)) for _ in range(n)]


def tcall(fn, *a):
    t0 = time.perf_counter(); fn(*a); return time.perf_counter() - t0


# ---- our CPU engine (single core) ----
def ours_logz(seqs, P):  return [tcall(cpu.logZ, s, P) for s in seqs]
def ours_mea(seqs, P):   return [tcall(cpu.mea, s, P) for s in seqs]
def ours_vit(seqs, P):   return [tcall(cpu.mfe, s, P) for s in seqs]


# ---- original binary (one process per sequence) ----
def _runbin(args, fp):
    subprocess.run([BIN, "predict", *args, fp],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def bin_task(seqs, args, tmp):
    out = []
    for i, s in enumerate(seqs):
        fp = os.path.join(tmp, f"s{i}.fa")
        with open(fp, "w") as fh:
            fh.write(f">s{i}\n{s}\n")
        out.append(tcall(_runbin, args, fp))
    return out

def spawn_floor(tmp, reps=21):
    fp = os.path.join(tmp, "tiny.fa")
    with open(fp, "w") as fh:
        fh.write(">t\nGGGGAAAACCCC\n")   # 12 nt, trivially fast to fold
    return statistics.median(tcall(_runbin, ["--partition"], fp) for _ in range(reps))


def stats(ts):
    a = np.array(ts)
    return a.mean() * 1e3, np.median(a) * 1e3   # mean_ms, median_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="50,100,150,200,300")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tasks", default="logz,mea,viterbi")
    ap.add_argument("--tmpdir", default="/tmp/cf_bench")
    args = ap.parse_args()

    os.makedirs(args.tmpdir, exist_ok=True)
    lengths = [int(x) for x in args.lengths.split(",")]
    tasks = args.tasks.split(",")
    P = cpu.load()

    # warm all three JIT kernels
    for L0 in (40,):
        s0 = gen_seqs(L0, 1, 1)[0]
        cpu.logZ(s0, P); cpu.mea(s0, P); cpu.mfe(s0, P)

    floor = spawn_floor(args.tmpdir)
    print(f"# binary       : {BIN}")
    print(f"# spawn+IO floor (median, 12-nt --partition): {floor*1e3:.2f} ms/call")
    print(f"# n per length = {args.n}, seed = {args.seed}, single CPU core")
    print(f"# ours_*  = this package's numba CPU engine (post warm-up), ms/seq")
    print(f"# bin_raw = contrafold per-seq incl. spawn ; bin_comp = spawn subtracted")
    print()

    tmap = {"logz": (ours_logz, ["--partition"]),
            "mea":  (ours_mea,  []),
            "viterbi": (ours_vit, ["--viterbi"])}

    hdr = (f"{'task':<8}{'len':>5}{'ours_ms':>10}{'bin_raw':>10}{'bin_comp':>10}"
           f"{'x_raw':>9}{'x_comp':>9}")
    print(hdr); print("-" * len(hdr))
    for task in tasks:
        fn, bargs = tmap[task]
        for L in lengths:
            seqs = gen_seqs(L, args.n, args.seed + L)
            o_mean, o_med = stats(fn(seqs, P))
            b_mean, b_med = stats(bin_task(seqs, bargs, args.tmpdir))
            b_comp = max(b_mean - floor * 1e3, 1e-6)
            print(f"{task:<8}{L:>5}{o_mean:>10.2f}{b_mean:>10.2f}{b_comp:>10.2f}"
                  f"{b_mean/o_mean:>8.2f}x{b_comp/o_mean:>8.2f}x")
        print()
    print("# x_raw  > 1  => ours faster than contrafold's real per-seq cost")
    print("# x_comp > 1  => ours faster than contrafold's compute alone")


if __name__ == "__main__":
    main()
