# Contrafold_GPU

CONTRAfold on the GPU — **~430× faster than a single CPU core** and **~37× a full
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

- **Accurate**: log-partition matches the CONTRAfold binary to ~1e-4 relative error
  (the residual is CONTRAfold's own `Fast_LogPlusEquals` lookup-table approximation;
  this implementation uses exact `log1p/exp`). Boltzmann sample base-pair frequencies
  match CONTRAfold's posterior probabilities within sampling noise.
- **Fast**: ~6,000–19,000 folds/s for 128-nt windows on an RTX 5090
  (≈2–6× a 32-core CPU running the CONTRAfold binary; ≈65–200× single-core).
- **Constraints**: any position can be forced unpaired (hard constraint), e.g. to
  encode DMS/SHAPE reactivity.

## Install

```bash
pip install numba numpy        # requires an NVIDIA GPU + CUDA toolkit (tested CUDA 12.8, sm_120)
```

The trained parameters (`data/contrafold.params.complementary`) are bundled.

## Usage

```python
from gpu_contrafold import load, logZ_batch, sample_batch

P = load()                                  # bundled CONTRAfold complementary params

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

## Batch CLI (JSONL in / JSONL out)

Fold a whole file of sequences in batches on the GPU:

```bash
python -m gpu_contrafold fold in.jsonl -o out.jsonl              # partition function (logZ)
python -m gpu_contrafold fold in.jsonl -o out.jsonl --sample 100 # 100 Boltzmann samples each
```

**Input** — one JSON object per line:

```json
{"id": "rna1", "seq": "GGGGAAAACCCC"}
{"id": "rna2", "seq": "GGGGAAAACCCC", "constrain": [0,0,0,1,0,0,0,0,0,0,0,0]}
```

- `seq` (required); non-ACGU is treated as `N` (cannot pair).
- `id` (optional) is echoed to the output; defaults to the 0-based line index.
- `constrain` (optional) is a per-base **0/1 hard-constraint mask**, length `== len(seq)`,
  where `1` forces that position unpaired (e.g. a DMS-reactive base). Accepts a list
  `[0,0,0,1,...]` or a string `"000100..."`; omit it (or `[]`) for no constraint.

**Output** — one JSON object per line, input order preserved:

```json
{"id": "rna1", "logZ": 3.05661}                                  # default
{"id": "rna1", "samples": ["((((....))))", "............"]}      # with --sample N
```

Add `--logz` to also include `logZ` in sample mode. Other flags: `--chunk` (sequences
per GPU launch, default 4096), `--seed`, `--threads`. Sequences are grouped by length
internally for efficiency.

## Benchmark

Full results are in [Benchmark.md](Benchmark.md). Reproduce:

```bash
# deterministic random sequences (reproducible from the seed)
python benchmarks/generate_sequences.py -n 10000 -l 200 -o benchmarks/data/sim_10k_200nt.fa

# batch-size sweep, length scaling, constraint overhead, sampling, GPU-vs-CPU
python benchmarks/benchmark.py benchmarks/data/sim_10k_200nt.fa --repeat 3
# add --skip-compare for a fast GPU-only run (no slow single-core CPU pass)
```

Headline on an RTX 5090 (10k × 200 nt): logZ peaks at ~**9,800 folds/s** (~430× a
single CPU core, ~37× the full 32-thread CPU); hard constraints add no GPU compute;
Boltzmann sampling runs at ~6,000 structures/s (100 seqs × 100 samples). GPU memory
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
1. CPU `logZ` == GPU `logZ` (batched), and
2. both == the original CONTRAfold binary (`--partition`) if available, and
3. GPU sample base-pair frequencies == CONTRAfold posteriors (`--posteriors`).

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
