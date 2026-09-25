#!/usr/bin/env python3
"""Long-sequence structure parity vs the original CONTRAfold binary.

test_validation.py samples 30-128 nt, where the GPU and the binary agree exactly.
Above ~150 nt the float32 posterior noise (~3e-4) occasionally exceeds the gap
between two near-degenerate MEA optima, so the GPU returns a structure differing
by a single base pair. Viterbi decoding has no such tie and stays exact.

Run:  GPU_CONTRAFOLD_BIN=/path/to/contrafold python tests/test_binary_long.py
Env:  GPU_CONTRAFOLD_NSEQ (default 100), GPU_CONTRAFOLD_SEED (default 20260831)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_contrafold import load, mea_gpu, cpu

N = int(os.environ.get("GPU_CONTRAFOLD_NSEQ", 100))
SEED = int(os.environ.get("GPU_CONTRAFOLD_SEED", 20260831))
MIN_EXACT_FRAC = 0.95      # >=95% of sequences must match the binary exactly
MAX_BP_DIFF = 2            # a non-matching structure may differ by <=2 base pairs

P = load()
rng = np.random.default_rng(SEED)
bases = np.array(list("ACGU"))


def rand_seq(L):
    return "".join(rng.choice(bases, size=L))


def pairs(db):
    st, out = [], set()
    for i, ch in enumerate(db):
        if ch == '(':
            st.append(i)
        elif ch == ')':
            out.add((st.pop(), i))
    return out


def report(tag, got, ref, exact_required):
    n = len(ref)
    bad = [i for i in range(n) if got[i] != ref[i]]
    worst = max((len(pairs(got[i]) ^ pairs(ref[i])) for i in bad), default=0)
    frac = (n - len(bad)) / n
    if exact_required:
        good = not bad
    else:
        good = frac >= MIN_EXACT_FRAC and worst <= MAX_BP_DIFF
    print(f"{tag:34s} {n - len(bad):3d}/{n} exact, worst diff {worst} bp"
          f"  {'PASS' if good else 'FAIL'}")
    return good


def main():
    try:
        from gpu_contrafold import reference as ref
        ref.logZ("GGGGAAAACCCC")            # fail fast if the binary is absent
    except Exception as e:
        print(f"CONTRAfold binary unavailable (set GPU_CONTRAFOLD_BIN): {type(e).__name__}")
        sys.exit(0)

    seqs = [rand_seq(int(L)) for L in rng.integers(150, 301, N)]
    print(f"{N} random sequences, {min(map(len, seqs))}-{max(map(len, seqs))} nt, seed {SEED}\n")
    ok = True

    # Viterbi/MAP: no near-ties, must be exact
    ok &= report("[1] Viterbi vs --viterbi",
                 [cpu.mfe(s, P) for s in seqs],
                 [ref.mfe_structure(s) for s in seqs], exact_required=True)

    # MEA on the GPU: allow rare 1-bp differences from float32 posterior noise
    for gamma in (6.0, 2.0):
        ok &= report(f"[2] GPU MEA gamma={gamma:g} vs predict",
                     mea_gpu(seqs, P, gamma=gamma),
                     [ref.mea_structure(s, gamma=gamma) for s in seqs],
                     exact_required=False)

    # CPU (double) engine must reproduce the binary exactly, including on the
    # sequences where the GPU diverges
    sub = seqs[:20]
    ok &= report("[3] CPU MEA gamma=6 vs predict",
                 [cpu.mea(s, P) for s in sub],
                 [ref.mea_structure(s) for s in sub], exact_required=True)

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
