"""gpu-contrafold: GPU-accelerated CONTRAfold partition function, sampling, and decoding.

A faithful reimplementation of CONTRAfold's CRF secondary-structure model on the
GPU (Numba CUDA). Reproduces the original CONTRAfold binary's log-partition,
posterior base-pair distribution, Viterbi/MFE structure, and maximum-expected-
accuracy (MEA) structure, with optional per-base hard constraints (e.g.
DMS-reactive positions forced unpaired).

Quick start:
    from gpu_contrafold import load, logZ_batch, sample_batch, mfe, mea, bpp
    P = load()                              # bundled CONTRAfold complementary params
    z = logZ_batch(["GGGGAAAACCCC"], P)     # log partition function(s)
    structs = sample_batch(["GGGGAAAACCCC"], P, n_samples=10)   # Boltzmann samples
    db = mea("GGGGAAAACCCC", P, gamma=6)    # MEA structure (== contrafold default predict)
    prob = bpp("GGGGAAAACCCC", P)           # exact base-pair probabilities
"""
from . import cpu
from . import gpu

load = cpu.load
cpu_logZ = cpu.logZ
mfe = cpu.mfe
mea = cpu.mea
bpp = cpu.bpp
logZ_batch = gpu.logZ_batch
sample_batch = gpu.sample_batch
fold_tasks_gpu = gpu.fold_tasks_gpu

__all__ = ["load", "cpu_logZ", "mfe", "mea", "bpp", "logZ_batch", "sample_batch",
           "fold_tasks_gpu", "cpu", "gpu"]
__version__ = "0.1.0"
