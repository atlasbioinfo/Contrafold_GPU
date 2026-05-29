"""Optional gold-standard reference: wraps the original CONTRAfold binary.

Requires the CONTRAfold binary (build it from the original source, BSD-licensed,
Do/Woods/Batzoglou 2006). Set its path via env GPU_CONTRAFOLD_BIN or pass
binary=... . Only needed to validate against the original; the GPU/CPU engines
do not require it.
"""
import os
import subprocess
import tempfile
import numpy as np

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRAFOLD = os.environ.get("GPU_CONTRAFOLD_BIN", "contrafold")
PARAMS = os.path.join(_PKG, "data", "contrafold.params.complementary")


def _write_fasta(seq, path):
    with open(path, "w") as f:
        f.write(f">s\n{seq}\n")


def logZ(seq, params=PARAMS):
    """Log partition coefficient from CONTRAfold."""
    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as f:
        f.write(f">s\n{seq}\n")
        fa = f.name
    try:
        r = subprocess.run([CONTRAFOLD, "predict", "--partition",
                            "--params", params, fa],
                           capture_output=True, text=True)
        # "Log partition coefficient for "...": 3.05664"
        for tok in r.stdout.split():
            try:
                return float(tok)
            except ValueError:
                continue
        raise RuntimeError(f"parse fail: {r.stdout} {r.stderr}")
    finally:
        os.remove(fa)


def viterbi_score(seq, params=PARAMS):
    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as f:
        f.write(f">s\n{seq}\n")
        fa = f.name
    try:
        r = subprocess.run([CONTRAFOLD, "predict", "--viterbi", "--partition",
                            "--params", params, fa],
                           capture_output=True, text=True)
        for tok in r.stdout.split():
            try:
                return float(tok)
            except ValueError:
                continue
        raise RuntimeError(f"parse fail: {r.stdout} {r.stderr}")
    finally:
        os.remove(fa)


def mfe_structure(seq, params=PARAMS):
    """Viterbi (MFE) dot-bracket structure via --viterbi --parens."""
    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as f:
        f.write(f">s\n{seq}\n")
        fa = f.name
    out = fa + ".parens"
    try:
        subprocess.run([CONTRAFOLD, "predict", "--viterbi", "--parens", out,
                        "--params", params, fa], capture_output=True, text=True)
        with open(out) as fh:
            lines = [l.rstrip("\n") for l in fh]
        # format: >s / seq / structure
        db = lines[-1].strip()
        return db
    finally:
        os.remove(fa)
        if os.path.exists(out):
            os.remove(out)


def bpp(seq, cutoff=1e-5, params=PARAMS):
    """Base-pair probability matrix (n x n, upper triangle) from --posteriors."""
    n = len(seq)
    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as f:
        f.write(f">s\n{seq}\n")
        fa = f.name
    out = fa + ".post"
    try:
        subprocess.run([CONTRAFOLD, "predict", "--posteriors", str(cutoff), out,
                        "--params", params, fa], capture_output=True, text=True)
        mat = np.zeros((n, n), dtype=np.float64)
        with open(out) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                i = int(parts[0]) - 1            # 1-based
                for tok in parts[2:]:
                    j_s, p_s = tok.split(":")
                    j = int(j_s) - 1
                    p = float(p_s)
                    if j > i:
                        mat[i, j] = p
                    else:
                        mat[j, i] = p
        return mat
    finally:
        os.remove(fa)
        if os.path.exists(out):
            os.remove(out)


if __name__ == "__main__":
    for s in ["GGGGAAAACCCC", "GCGCGCAAAAGCGCGCAAAAGCGC",
              "GGGCUAUUAGCUCAGUUGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCCCA"]:
        print(f"{s}")
        print(f"  logZ={logZ(s):.5f}  viterbi={viterbi_score(s):.5f}")
        print(f"  MFE struct: {mfe_structure(s)}")
        b = bpp(s)
        print(f"  bpp nonzero: {(b>0.01).sum()}  max={b.max():.4f}\n")
