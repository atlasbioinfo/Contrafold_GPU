"""gpu-contrafold: GPU-accelerated CONTRAfold partition function + Boltzmann sampling.

A faithful reimplementation of CONTRAfold's CRF secondary-structure model on the
GPU (Numba CUDA). Reproduces the original CONTRAfold binary's log-partition and
posterior base-pair distribution, with optional per-base hard constraints
(e.g. DMS-reactive positions forced unpaired).

Quick start:
    from gpu_contrafold import load, logZ_batch, sample_batch
    P = load()                              # bundled CONTRAfold complementary params
    z = logZ_batch(["GGGGAAAACCCC"], P)     # log partition function(s)
    structs = sample_batch(["GGGGAAAACCCC"], P, n_samples=10)   # Boltzmann samples
"""
from . import cpu
from . import gpu

load = cpu.load
cpu_logZ = cpu.logZ
logZ_batch = gpu.logZ_batch
sample_batch = gpu.sample_batch
fold_tasks_gpu = gpu.fold_tasks_gpu

__all__ = ["load", "cpu_logZ", "logZ_batch", "sample_batch", "fold_tasks_gpu", "cpu", "gpu"]
__version__ = "0.1.0"
