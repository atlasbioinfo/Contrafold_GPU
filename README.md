# Contrafold_GPU

CONTRAfold deployed on the GPU — up to ~64× faster than a single CPU core
(≈2–6× a 32-core CPU), reproducing the original CONTRAfold's output.

GPU-accelerated **CONTRAfold** RNA secondary-structure model: partition function,
posterior base-pair probabilities, and Boltzmann structure sampling — with optional
per-base hard constraints (e.g. force DMS-reactive positions unpaired).

It is a faithful reimplementation of CONTRAfold's CRF scoring + inside recurrence
(complementary model) that **reproduces the original CONTRAfold binary's output**,
running on the GPU via [Numba CUDA].

## Why

CONTRAfold is a CPU-only C++ tool. Folding millions of short windows (e.g. for
DMS-MaPseq / SHAPE structure-probing pipelines) is slow. This package runs the
same model on the GPU, batching thousands of sequences at once.

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

This reimplements the model of **CONTRAfold** and ships its trained parameter file:

> Do CB, Woods DA, Batzoglou S. *CONTRAfold: RNA secondary structure prediction
> without physics-based models.* Bioinformatics. 2006;22(14):e90-8.

CONTRAfold is distributed under the BSD license. The bundled parameter file and the
scoring/recurrence logic derive from the CONTRAfold source; please retain this
attribution and cite the paper. This GPU reimplementation is provided for research use.

[Numba CUDA]: https://numba.readthedocs.io/en/stable/cuda/index.html
