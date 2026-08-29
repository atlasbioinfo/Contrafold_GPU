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

Folding many reads of one transcript (probing data) is a different shape of
problem: thousands of short sequences rather than one long one. Use the
thread-per-sequence path, which folds one sequence per CUDA thread instead of
one per block:

    from gpu_contrafold import mea_gpu_tps
    dbs = mea_gpu_tps([seq] * len(masks), P, gamma=6, forced_list=masks)

``mea_gpu_tps`` produces bit-identical posteriors to ``mea_gpu``; only the lane
mapping and matrix layout differ.
"""
from . import cpu
from . import gpu
from . import gpu_mea

load = cpu.load
cpu_logZ = cpu.logZ
mfe = cpu.mfe
mea = cpu.mea
bpp = cpu.bpp
logZ_batch = gpu.logZ_batch
sample_batch = gpu.sample_batch
fold_tasks_gpu = gpu.fold_tasks_gpu
# GPU posterior + MEA. The *_tps variants fold one sequence per CUDA thread and
# are the fast path for a batch of many short sequences.
mea_gpu = gpu_mea.mea_gpu
bpp_gpu = gpu_mea.bpp_gpu
mea_gpu_tps = gpu_mea.mea_gpu_tps
bpp_gpu_tps = gpu_mea.bpp_gpu_tps

__all__ = ["load", "cpu_logZ", "mfe", "mea", "bpp", "logZ_batch", "sample_batch",
           "fold_tasks_gpu", "mea_gpu", "bpp_gpu", "mea_gpu_tps", "bpp_gpu_tps",
           "cpu", "gpu", "gpu_mea"]
__version__ = "0.2.0"
