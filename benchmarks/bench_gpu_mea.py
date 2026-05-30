"""Benchmark GPU MEA (gpu_contrafold.gpu_mea) vs CPU MEA (cpu.mea).

Measures, after warm-up (JIT + CUDA compile excluded):
  1) Speed: per-sequence latency for CPU mea and single-seq GPU mea, plus
     large-batch GPU throughput (folds/s), across lengths.
  2) Accuracy: dot-bracket exact-match rate GPU vs CPU; BPP max/mean abs diff;
     a spot-check of GPU MEA vs the CONTRAfold binary `predict` default.

Run with the conda-base python that has Numba CUDA:
    /usr/users/JIC_c1/hyu/miniforge3/bin/python benchmarks/bench_gpu_mea.py
"""
import os
import sys
import time
import tempfile
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_contrafold import cpu, gpu_mea

CONTRAFOLD_BIN = "/usr/users/JIC_c1/hyu/miniforge3/envs/RNAFOLD/bin/contrafold"
LENGTHS = [50, 100, 150, 200, 300, 400]
RNG = np.random.default_rng(12345)


def rand_seq(L):
    return "".join(RNG.choice(list("ACGU"), size=L))


def time_call(fn, repeat):
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat


def main():
    P = cpu.load()

    # ---- warm-up: compile CPU njit + GPU kernel (excluded from timing) ----
    _ = cpu.mea("GGGGAAAACCCC", P)
    _ = cpu.bpp("GGGGAAAACCCC", P)
    _ = gpu_mea.mea_gpu(["GGGGAAAACCCC", "GCGCGCAAAAGCGC"], P)
    _ = gpu_mea.bpp_gpu(["GGGGAAAACCCC"], P)

    print("=" * 72)
    print("SPEED  (per-sequence ms after warm-up)")
    print("=" * 72)
    print(f"{'len':>5} {'CPU ms':>10} {'GPU 1seq ms':>12} "
          f"{'GPU batch ms/seq':>17} {'batch folds/s':>14} "
          f"{'spdup(1)':>9} {'spdup(batch)':>13}")

    speed_rows = []
    for L in LENGTHS:
        # one representative sequence for latency, plus a batch
        seq = rand_seq(L)
        batch_n = 512 if L <= 200 else 256
        batch = [rand_seq(L) for _ in range(batch_n)]

        rep = 5 if L <= 200 else 3
        cpu_ms = time_call(lambda: cpu.mea(seq, P), rep) * 1e3
        gpu1_ms = time_call(lambda: gpu_mea.mea_gpu([seq], P), rep) * 1e3

        # batch throughput (single GPU launch over batch_n seqs)
        t0 = time.perf_counter()
        gpu_mea.mea_gpu(batch, P)
        bt = time.perf_counter() - t0
        gpu_batch_ms = bt / batch_n * 1e3
        folds_s = batch_n / bt

        speed_rows.append((L, cpu_ms, gpu1_ms, gpu_batch_ms, folds_s))
        print(f"{L:>5} {cpu_ms:>10.3f} {gpu1_ms:>12.3f} "
              f"{gpu_batch_ms:>17.4f} {folds_s:>14.1f} "
              f"{cpu_ms/gpu1_ms:>9.2f} {cpu_ms/gpu_batch_ms:>13.1f}")

    # ---- batch-decode breakdown: posterior(GPU) vs decode(CPU) share ----
    print("\n" + "=" * 72)
    print("BATCH TIME BREAKDOWN  (posterior on GPU vs MEA decode on CPU)")
    print("=" * 72)
    for L in [100, 200, 400]:
        batch_n = 256
        batch = [rand_seq(L) for _ in range(batch_n)]
        t0 = time.perf_counter()
        POSTh, S, Larr, FO = gpu_mea._run_post_kernel(batch, P, None, 128)
        t_post = time.perf_counter() - t0
        g = np.float32(6.0)
        t0 = time.perf_counter()
        for bi in range(batch_n):
            Lb = int(Larr[bi]); nn = Lb + 2
            POST = np.ascontiguousarray(POSTh[bi, :nn, :nn])
            s = np.ascontiguousarray(S[bi, :nn].astype(np.int64))
            fo = np.ascontiguousarray(FO[bi, :nn].astype(np.int64))
            cpu.mea_decode(POST, Lb, fo, cpu._CANON, s, g)
        t_dec = time.perf_counter() - t0
        print(f"len {L:>4}  n={batch_n}  GPU posterior {t_post*1e3:8.1f} ms total "
              f"({t_post/batch_n*1e3:.3f} ms/seq)   "
              f"CPU decode {t_dec*1e3:8.1f} ms total ({t_dec/batch_n*1e3:.3f} ms/seq)")

    # ---- accuracy: dot-bracket exact match + BPP diff ----
    print("\n" + "=" * 72)
    print("ACCURACY  (GPU vs CPU, 100 random seqs per length)")
    print("=" * 72)
    print(f"{'len':>5} {'mea match':>12} {'pos diff/seqs':>14} "
          f"{'bpp max|d|':>12} {'bpp mean|d|':>12}")
    for L in LENGTHS:
        nacc = 100 if L <= 200 else 40
        seqs = [rand_seq(L) for _ in range(nacc)]
        gpu_db = gpu_mea.mea_gpu(seqs, P)
        cpu_db = [cpu.mea(s, P) for s in seqs]
        match = sum(1 for a, b in zip(gpu_db, cpu_db) if a == b)
        # base-pair position differences across all mismatched seqs
        pos_diff = 0
        for a, b in zip(gpu_db, cpu_db):
            if a != b:
                pos_diff += sum(1 for ca, cb in zip(a, b) if ca != cb)
        # BPP diff over a sub-sample (20 seqs) for cost
        sub = seqs[:20]
        gbpp = gpu_mea.bpp_gpu(sub, P)
        maxd = 0.0
        sumd = 0.0
        cnt = 0
        for s, gb in zip(sub, gbpp):
            cb = cpu.bpp(s, P)
            dif = np.abs(gb - cb)
            maxd = max(maxd, float(dif.max()))
            sumd += float(dif.sum())
            cnt += dif.size
        meand = sumd / cnt
        print(f"{L:>5} {match:>5}/{nacc:<4}  {pos_diff:>14} {maxd:>12.2e} {meand:>12.2e}", flush=True)

    # ---- anchor to gold binary: GPU MEA vs contrafold predict default ----
    print("\n" + "=" * 72)
    print("GOLD ANCHOR  (GPU MEA vs CONTRAfold binary `predict`, 20 seqs)")
    print("=" * 72)
    if not os.path.exists(CONTRAFOLD_BIN):
        print("contrafold binary not found; skipping")
    else:
        bseqs = [rand_seq(RNG.integers(40, 160)) for _ in range(20)]
        gpu_db = gpu_mea.mea_gpu(bseqs, P)
        bmatch = 0
        for sq, gdb in zip(bseqs, gpu_db):
            with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as fh:
                fh.write(">s\n" + sq + "\n")
                fa = fh.name
            try:
                out = subprocess.run([CONTRAFOLD_BIN, "predict", fa],
                                     capture_output=True, text=True, timeout=120)
                lines = [l for l in out.stdout.splitlines()
                         if set(l.strip()) <= set("().") and l.strip()]
                bin_db = lines[-1].strip() if lines else ""
            finally:
                os.unlink(fa)
            if bin_db == gdb:
                bmatch += 1
        print(f"GPU MEA matches binary on {bmatch}/20 sequences")


if __name__ == "__main__":
    main()
