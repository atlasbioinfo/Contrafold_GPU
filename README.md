# Contrafold_GPU

CONTRAfold on the GPU — **~460× faster than a single CPU core** and **~38× a full
32-thread CPU** at folding 10,000 × 200 nt RNAs on an RTX 5090 (vs this package's
own CPU reference; see [Benchmark.md](Benchmark.md)), reproducing the original
CONTRAfold's output.

GPU-accelerated **CONTRAfold** RNA secondary-structure model: partition function,
posterior base-pair probabilities, and Boltzmann structure sampling — with optional
per-base hard constraints (e.g. force DMS-reactive positions unpaired).

It is a faithful reimplementation of CONTRAfold's CRF scoring + inside recurrence
(complementary model) that **reproduces the original CONTRAfold binary's output**,
running on the GPU via [Numba CUDA].

## Why

This started from a practical need: I had to fold large batches of RNA with
[CONTRAfold](http://contra.stanford.edu/contrafold/), but it is a CPU-only C++
tool and was simply too slow for that volume. So I reimplemented its model on the
GPU. Folding millions of short windows (e.g. for DMS-MaPseq / SHAPE
structure-probing pipelines) now runs the same model on the GPU, batching
thousands of sequences at once.

- **Accurate**: the GPU engine reproduces CONTRAfold's **float-precision arithmetic**
  — the same 8-segment polynomial `Fast_LogExpPlusOne` used by the `RealT=float`
  binary — so log-partition matches the binary to ~5e-4 (≈1e-4 relative). The small
  residual is float32 summation order on the GPU, not the algorithm; bit-identical
  output isn't achievable on a parallel device (non-associative float adds + FMA).
  The bundled CPU engine instead uses exact `log1p/exp` (double) as a higher-precision
  reference, mirroring a `RealT=double` CONTRAfold build. Boltzmann sample base-pair
  frequencies match CONTRAfold's posterior probabilities within sampling noise.
- **Fast**: ~6,000–19,000 folds/s for 128-nt windows on an RTX 5090
  (≈2–6× a 32-core CPU running the CONTRAfold binary; ≈65–200× single-core).
- **Constraints**: any position can be forced unpaired (hard constraint), e.g. to
  encode DMS/SHAPE reactivity.

## Install

```bash
pip install gpu-contrafold
```

Requires an NVIDIA GPU + CUDA toolkit (tested CUDA 12.8, sm_120); `numpy` and `numba`
are pulled in automatically. The trained parameters
(`gpu_contrafold/data/contrafold.params.complementary`) are bundled. The CPU engine
works without a GPU. Install from source with `pip install .`.

## Command-line tool

`pip install` exposes a `gpu-contrafold` command (also `gpu_contrafold`, or
`python -m gpu_contrafold`) — pass the sequence or file directly:

```bash
gpu-contrafold GGGGAAAACCCC                 # one sequence -> MFE structure (dot-bracket)
gpu-contrafold GGGGAAAACCCC --sample 10     # 10 Boltzmann samples
gpu-contrafold GGGGAAAACCCC --logz          # partition function (logZ) instead
gpu-contrafold seqs.jsonl  -o out.jsonl     # batch: JSONL in -> JSONL out
gpu-contrafold seqs.fasta  -o out.jsonl --sample 100
```

The default is the **maximum-probability (Viterbi/MAP) structure** — identical to
`contrafold --viterbi` (verified: exact dot-bracket match over 40 random sequences).
`--sample N` draws `N` Boltzmann samples; `--logz` gives the partition function.

The argument is a literal RNA sequence if it is not an existing file; otherwise it is
read as **JSONL** (lines starting with `{`), **FASTA** (`>`), or one sequence per line.

JSONL input — one object per line:

```json
{"id": "rna1", "seq": "GGGGAAAACCCC"}
{"id": "rna2", "seq": "GGGGAAAACCCC", "constrain": [0,0,0,1,0,0,0,0,0,0,0,0]}
```

- `seq` (required); non-ACGU is treated as `N` (cannot pair).
- `id` (optional), echoed to output; defaults to the 0-based index.
- `constrain` (optional): per-base **0/1 hard-constraint mask**, length `== len(seq)`,
  `1` forces that position unpaired (e.g. a DMS-reactive base). A list `[0,0,1,...]`
  or string `"001..."`; omit for none.

Output: a single literal sequence prints structures to stdout (one per line); file
input (or any `-o`) writes JSONL — `{"id","structure"}` by default, `{"id","samples":[...]}`
with `--sample N`, or `{"id","logZ"}` with `--logz` — input order preserved. Flags:
`--sample N`, `--logz`, `--chunk`, `--seed`, `--threads`.

## Usage

```python
from gpu_contrafold import load, logZ_batch, sample_batch, mfe

P = load()                                  # bundled CONTRAfold complementary params

# Maximum-probability (Viterbi/MAP) structure, dot-bracket (matches contrafold --viterbi)
db = mfe("GGGGAAAACCCC", P)                  # -> "((((....))))"

# Log partition function (one per sequence), batched on GPU
z = logZ_batch(["GGGGAAAACCCC", "GCGCGCAAAAGCGCGC"], P)

# Boltzmann-sample structures (dot-bracket), N per sequence
structs = sample_batch(["GGGGAAAACCCC"], P, n_samples=10)   # -> [[db, db, ...]]

# Hard constraints: force positions unpaired (1 = forced unpaired)
import numpy as np
mask = np.zeros(12, np.int8); mask[3] = 1                   # position 3 cannot pair
structs = sample_batch(["GGGGAAAACCCC"], P, n_samples=10, forced_list=[mask])

# Bulk pipeline primitive: fold many (seq, mask) tasks, 1 sample each, chunked
from gpu_contrafold import fold_tasks_gpu
dbs = fold_tasks_gpu(seqs, masks, P)        # list of dot-bracket strings
```

CPU reference (matches the binary, no GPU needed):

```python
from gpu_contrafold import cpu
P = cpu.load()
print(cpu.logZ("GGGGAAAACCCC", P))
```

## Benchmark

Full results are in [Benchmark.md](Benchmark.md). Reproduce:

```bash
# deterministic random sequences (reproducible from the seed)
python benchmarks/generate_sequences.py -n 10000 -l 200 -o benchmarks/data/sim_10k_200nt.fa

# batch-size sweep, length scaling, constraint overhead, sampling, GPU-vs-CPU
python benchmarks/benchmark.py benchmarks/data/sim_10k_200nt.fa --repeat 3
# add --skip-compare for a fast GPU-only run (no slow single-core CPU pass)
```

Headline on an RTX 5090 (10k × 200 nt): logZ peaks at ~**9,800 folds/s** (~460× a
single CPU core, ~38× the full 32-thread CPU); hard constraints are effectively free;
Boltzmann sampling runs at ~5,600 structures/s (100 seqs × 100 samples). GPU memory
scales as `~3 × (L+2)² × 4 B` per sequence in a batch (≈0.5 MB/seq at 200 nt,
≈3 MB/seq at 500 nt), so use a smaller batch for longer sequences or smaller GPUs.

## Model

Complementary CONTRAfold model, all scoring terms enabled: `base_pair`,
`helix_stacking`, `terminal_mismatch`, `helix_closing`, `dangle_left/right`,
`hairpin_length`, `bulge_length`, `internal_length`, `internal_symmetric`,
`internal_asymmetry`, `internal_explicit` (≤4×4), `internal_1x1`, `bulge_0x1`,
affine multi-branch and external loops. Recurrence is CONTRAfold's unambiguous
`FC/FM/FM1/FM2/F5` (simple-FC variant), 1-based `s[1..L]`, natural-log
(`log_base = 1.0`). Non-ACGU characters (`N`) map to a 5th code that cannot pair
and carries zero scores — exactly as the CONTRAfold binary handles them.

## Validation

`tests/test_validation.py` checks:
1. CPU (exact, double) vs GPU (CONTRAfold float arithmetic) `logZ` agree to ~1e-3, and
2. GPU `logZ` matches the original CONTRAfold binary (`--partition`) to ~5e-4 if available, and
3. GPU sample base-pair frequencies == CONTRAfold posteriors (`--posteriors`).

Verified against CONTRAfold 2.02 (max |Δ logZ| 5e-4 over 20 random sequences;
sample BPP within 0.006 of `--posteriors`).

To validate against the original binary, build CONTRAfold and point to it:
```bash
export GPU_CONTRAFOLD_BIN=/path/to/contrafold
python tests/test_validation.py
```

## Attribution & License

This is a GPU rewrite of **CONTRAfold** (http://contra.stanford.edu/contrafold/)
and ships its trained parameter file:

> Do CB, Woods DA, Batzoglou S. *CONTRAfold: RNA secondary structure prediction
> without physics-based models.* Bioinformatics. 2006;22(14):e90-8.
> Project page: http://contra.stanford.edu/contrafold/

This GPU reimplementation is released under the **MIT license** (see `LICENSE`).
It incorporates the CONTRAfold model and bundles its trained parameter file, which
derive from the **BSD-licensed** CONTRAfold source (see `NOTICE.md`); those components
remain under their original terms. Please retain that attribution and cite the paper.

[Numba CUDA]: https://numba.readthedocs.io/en/stable/cuda/index.html
