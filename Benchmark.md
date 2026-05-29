# Benchmark

GPU throughput of `gpu_contrafold` for the CONTRAfold partition function (logZ) and Boltzmann sampling. Each figure is the wall-clock time to process the whole dataset, reported as **mean ± std over 3 runs**.

## Summary

| workload | result |
|:--|:--|
| logZ throughput, 10k × 200 nt | **9,830 folds/s** (batch 10,000) |
| logZ + DMS-like constraints, 10k × 200 nt | **9,640 folds/s** |
| sampling, 100 × 200 nt × 100 | **5,627 samples/s** |
| GPU vs CPU, 10k × 200 nt | **464× one core**, 38× all 32 threads |

## Environment

- GPU: NVIDIA GeForce RTX 5090
- CPU: Intel(R) Core(TM) i9-14900F (32 logical cores)
- Dataset: 10,000 random RNA sequences × 200 nt (uniform ACGU, seed 0)
- numba 0.62.1, numpy 1.26.4, Python 3.12.8
- CPU/GPU logZ agree to max|diff| 1.7e-04 over 8 sequences.

## 1. Batch-size sweep — GPU logZ, 200 nt, 10,000 seqs

How many sequences per GPU launch (`--chunk`) best saturate the device. Throughput climbs with batch size until the GPU is full, then plateaus; memory grows linearly (`~3·(L+2)²·4 B` per sequence).

| batch (chunk) | time (s) | folds/s | ~GPU mem |
|---:|---:|---:|---:|
| 256 | 3.258 ± 0.011 | 3,069 | 0.1 GB |
| 512 | 1.852 ± 0.006 | 5,399 | 0.3 GB |
| 1,024 | 1.271 ± 0.006 | 7,870 | 0.5 GB |
| 2,048 | 1.193 ± 0.004 | 8,381 | 1.0 GB |
| 4,096 | 1.148 ± 0.007 | 8,707 | 2.0 GB |
| 8,192 | 1.077 ± 0.009 | 9,282 | 4.0 GB |
| 10,000 **(peak)** | 1.017 ± 0.002 | 9,830 | 4.9 GB |

Peak ≈ **9,830 folds/s at batch 10,000**. Throughput reaches ~90 % of peak by batch **8,192** (~4.0 GB); larger batches add only a few percent for proportionally more memory. **Recommended batch: 8,192–10,000** at 200 nt — use the lower end on memory-limited GPUs. Scale the batch down for longer sequences, since per-sequence memory grows as `(L+2)²`.

## 2. Length scaling — GPU logZ, batch 2,048, 2,048 seqs each

The partition function is O(L³) time / O(L²) memory, so throughput falls steeply as sequences get longer.

| length (nt) | time (s) | folds/s |
|---:|---:|---:|
| 50 | 0.021 ± 0.001 | 95,263 |
| 100 | 0.068 ± 0.000 | 30,219 |
| 200 | 0.240 ± 0.001 | 8,537 |
| 300 | 0.580 ± 0.003 | 3,531 |
| 400 | 1.131 ± 0.006 | 1,811 |
| 500 | 1.940 ± 0.007 | 1,056 |

## 3. Hard-constraint overhead — GPU logZ, 10,000 × 200 nt, batch 10,000

Hard constraints are effectively free: forbidden pairs are pruned from the recurrence (no extra GPU work) and the per-base mask arrays are built with vectorised host code. Masks here force a random 2–10 positions per 100 nt unpaired (avg **12.0** forced positions per sequence).

| configuration | time (s) | folds/s |
|:--|---:|---:|
| no constraint | 1.031 ± 0.019 | 9,698 |
| random DMS-like constraint | 1.037 ± 0.016 | 9,640 |

Constrained vs unconstrained throughput differs by only -0.6 % (within run-to-run noise) — applying constraints costs nothing.

## 4. Boltzmann sampling — 100 × 200 nt, 100 samples each

Fold the inside matrices once per sequence, then draw 100 stochastic tracebacks each (10,000 structures total).

| workload | time (s) | samples/s | seqs/s |
|:--|---:|---:|---:|
| 100 seqs × 100 samples | 1.777 ± 0.065 | 5,627 | 56 |

## 5. GPU vs CPU — 10,000 × 200 nt

Wall-clock time to compute logZ for all 10,000 sequences.

| device | time (s) | folds/s | speedup vs 1-core |
|:--|---:|---:|---:|
| NVIDIA GeForce RTX 5090 (GPU, batch 10,000) | 1.038 ± 0.013 | 9,636 | 464.5× |
| CPU 1 core | 481.984 ± 5.234 | 21 | 1× (baseline) |
| CPU 32 cores | 39.949 ± 1.490 | 250 | 12.1× |

GPU is **464.5× a single CPU core** and **38.5× the full 32-thread CPU** on this workload.

> **Baseline note:** the CPU figures use this package's own NumPy/Numba `cpu.logZ` reference (~48 ms/fold single-core), *not* the optimized CONTRAfold C++ binary, which is considerably faster. These multiples are GPU vs. this reference; the speed-up over the C++ binary would be smaller. Multi-core scaling is sublinear (12.1× on 32 threads) because the Intel(R) Core(TM) i9-14900F mixes performance and efficiency cores with hyper-threading.

---

Reproduce:

```bash
python benchmarks/generate_sequences.py -n 10000 -l 200 -o benchmarks/data/sim_10k_200nt.fa
python benchmarks/benchmark.py benchmarks/data/sim_10k_200nt.fa --repeat 3
```
