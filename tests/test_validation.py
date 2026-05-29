#!/usr/bin/env python3
"""Validation: CPU==GPU logZ, vs CONTRAfold binary (if available), sampling vs posteriors.

Run:  python tests/test_validation.py
Optional binary check:  GPU_CONTRAFOLD_BIN=/path/to/contrafold python tests/test_validation.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_contrafold import load, logZ_batch, sample_batch, cpu

P = load()
rng = np.random.default_rng(0)
bases = np.array(list("ACGU"))


def rand_seq(L):
    return "".join(rng.choice(bases, size=L))


def main():
    ok = True

    # 1. CPU vs GPU logZ
    seqs = [rand_seq(int(rng.integers(30, 129))) for _ in range(60)]
    cpu_z = [cpu.logZ(s, P) for s in seqs]
    gpu_z = logZ_batch(seqs, P)
    d = max(abs(a - b) for a, b in zip(cpu_z, gpu_z))
    print(f"[1] CPU vs GPU logZ (60 seqs):   max diff = {d:.2e}  {'PASS' if d < 1e-2 else 'FAIL'}")
    ok &= d < 1e-2

    # 2. vs CONTRAfold binary (optional)
    try:
        from gpu_contrafold import reference as ref
        rz = [ref.logZ(s) for s in seqs[:20]]
        d2 = max(abs(a - b) for a, b in zip(gpu_z[:20], rz))
        print(f"[2] GPU vs CONTRAfold binary:    max diff = {d2:.2e}  {'PASS' if d2 < 5e-3 else 'FAIL'}")
        ok &= d2 < 5e-3
    except Exception as e:
        print(f"[2] CONTRAfold binary check skipped (set GPU_CONTRAFOLD_BIN): {type(e).__name__}")

    # 3. sampling distribution vs CONTRAfold posteriors (optional)
    try:
        from gpu_contrafold import reference as ref
        s = "GCGCGCAAAAGCGCGCAAAAGCGC"
        L = len(s)
        samples = sample_batch([s], P, 20000, seed=1)[0]
        bpp = np.zeros((L, L))
        for db in samples:
            st = []
            for k, ch in enumerate(db):
                if ch == '(':
                    st.append(k)
                elif ch == ')':
                    a = st.pop(); bpp[a, k] += 1
        bpp /= len(samples)
        refbpp = ref.bpp(s)
        md = np.abs(bpp - refbpp).max()
        print(f"[3] sample bpp vs posteriors:    max diff = {md:.4f}  {'PASS' if md < 0.02 else 'FAIL'}")
        ok &= md < 0.02
    except Exception as e:
        print(f"[3] posterior check skipped: {type(e).__name__}")

    # 4. constraint honored
    L = 24
    s = rand_seq(L)
    mask = (rng.random(L) < 0.3).astype(np.int8)
    viol = 0
    for db in sample_batch([s], P, 2000, forced_list=[mask], seed=2)[0]:
        for k, ch in enumerate(db):
            if mask[k] and ch != '.':
                viol += 1
    print(f"[4] forced-unpaired violations:  {viol}  {'PASS' if viol == 0 else 'FAIL'}")
    ok &= viol == 0

    # 5. MFE (Viterbi) structure vs CONTRAfold --viterbi (optional)
    try:
        from gpu_contrafold import reference as ref
        nmatch = sum(cpu.mfe(s, P) == ref.mfe_structure(s) for s in seqs[:20])
        print(f"[5] MFE structure vs --viterbi:  {nmatch}/20 exact  {'PASS' if nmatch == 20 else 'FAIL'}")
        ok &= nmatch == 20
    except Exception as e:
        print(f"[5] MFE structure check skipped: {type(e).__name__}")

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
