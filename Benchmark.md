# Benchmark

GPU throughput of `gpu_contrafold` for the CONTRAfold partition function (logZ) and Boltzmann sampling. Each figure is the wall-clock time to process the whole dataset, reported as **mean ± std over 3 runs**.

## Summary

| workload | result |
|:--|:--|
| logZ throughput, 10k × 200 nt | **9,778 folds/s** (batch 10,000) |
| logZ + DMS-like constraints, 10k × 200 nt | **8,413 folds/s** |
| sampling, 100 × 200 nt × 100 | **6,061 samples/s** |
| GPU vs CPU, 10k × 200 nt | **433× one core**, 37× all 32 threads |

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
| 256 | 3.260 ± 0.005 | 3,068 | 0.1 GB |
| 512 | 1.856 ± 0.001 | 5,389 | 0.3 GB |
| 1,024 | 1.271 ± 0.002 | 7,870 | 0.5 GB |
| 2,048 | 1.197 ± 0.003 | 8,354 | 1.0 GB |
| 4,096 | 1.152 ± 0.005 | 8,684 | 2.0 GB |
| 8,192 | 1.085 ± 0.002 | 9,221 | 4.0 GB |
| 10,000 **(peak)** | 1.023 ± 0.004 | 9,778 | 4.9 GB |

Peak ≈ **9,778 folds/s at batch 10,000**. Throughput reaches ~90 % of peak by batch **8,192** (~4.0 GB); larger batches add only a few percent for proportionally more memory. **Recommended batch: 8,192–10,000** at 200 nt — use the lower end on memory-limited GPUs. Scale the batch down for longer sequences, since per-sequence memory grows as `(L+2)²`.

## 2. Length scaling — GPU logZ, batch 2,048, 2,048 seqs each

The partition function is O(L³) time / O(L²) memory, so throughput falls steeply as sequences get longer.

| length (nt) | time (s) | folds/s |
|---:|---:|---:|
| 50 | 0.022 ± 0.000 | 94,561 |
| 100 | 0.069 ± 0.001 | 29,757 |
| 200 | 0.241 ± 0.000 | 8,508 |
| 300 | 0.582 ± 0.003 | 3,518 |
| 400 | 1.131 ± 0.008 | 1,810 |
| 500 | 1.948 ± 0.010 | 1,051 |

## 3. Hard-constraint overhead — GPU logZ, 10,000 × 200 nt, batch 10,000

Hard constraints add no GPU compute — forbidden pairs are pruned from the recurrence. The only difference is host-side: building the per-base mask arrays. Masks here force a random 2–10 positions per 100 nt unpaired (avg **12.0** forced positions per sequence).

| configuration | time (s) | folds/s |
|:--|---:|---:|
| no constraint | 1.025 ± 0.009 | 9,753 |
| random DMS-like constraint | 1.189 ± 0.015 | 8,413 |

End-to-end throughput differs by -15.9 % here, from host-side mask marshalling (a per-sequence Python loop); the GPU kernel itself does no extra work. Vectorising that marshalling would close the gap.

## 4. Boltzmann sampling — 100 × 200 nt, 100 samples each

Fold the inside matrices once per sequence, then draw 100 stochastic tracebacks each (10,000 structures total).

| workload | time (s) | samples/s | seqs/s |
|:--|---:|---:|---:|
| 100 seqs × 100 samples | 1.650 ± 0.090 | 6,061 | 61 |

## 5. GPU vs CPU — 10,000 × 200 nt

Wall-clock time to compute logZ for all 10,000 sequences.

| device | time (s) | folds/s | speedup vs 1-core |
|:--|---:|---:|---:|
| NVIDIA GeForce RTX 5090 (GPU, batch 10,000) | 1.052 ± 0.010 | 9,504 | 433.0× |
| CPU 1 core | 455.623 ± 5.567 | 22 | 1× (baseline) |
| CPU 32 cores | 38.756 ± 0.467 | 258 | 11.8× |

GPU is **433.0× a single CPU core** and **36.8× the full 32-thread CPU** on this workload.

> **Baseline note:** the CPU figures use this package's own NumPy/Numba `cpu.logZ` reference (~46 ms/fold single-core), *not* the optimized CONTRAfold C++ binary, which is considerably faster. These multiples are GPU vs. this reference; the speed-up over the C++ binary would be smaller. Multi-core scaling is sublinear (11.8× on 32 threads) because the Intel(R) Core(TM) i9-14900F mixes performance and efficiency cores with hyper-threading.

---

Reproduce:

```bash
python benchmarks/generate_sequences.py -n 10000 -l 200 -o benchmarks/data/sim_10k_200nt.fa
python benchmarks/benchmark.py benchmarks/data/sim_10k_200nt.fa --repeat 3
```
