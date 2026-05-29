# Benchmark

`gpu_contrafold` partition-function (logZ) throughput. Each configuration is the wall-clock time to fold the dataset, reported as **mean ± std over 3 runs**.

## Environment

- GPU: NVIDIA GeForce RTX 5090
- CPU: Intel(R) Core(TM) i9-14900F (32 logical cores)
- Dataset: 10000 random RNA sequences x 200 nt (uniform ACGU)
- numba 0.62.1, numpy 1.26.4, Python 3.12.8

## 1. Batch-size sweep (GPU, 200 nt, 10000 seqs)

How many sequences per GPU launch (`--chunk`) best saturates the GPU. Throughput rises with batch size until the GPU is full, then plateaus; memory grows linearly (`~3·(L+2)²·4 B` per sequence).

| batch (chunk) | time (s) | folds/s | ~GPU mem |
|---:|---:|---:|---:|
| 256 | 4.161 ± 0.110 | 2,403 | 0.1 GB |
| 512 | 2.361 ± 0.115 | 4,236 | 0.3 GB |
| 1,024 | 1.631 ± 0.012 | 6,131 | 0.5 GB |
| 2,048 | 1.526 ± 0.084 | 6,551 | 1.0 GB |
| 4,096 | 1.449 ± 0.131 | 6,904 | 2.0 GB |
| 8,192 | 1.384 ± 0.179 | 7,224 | 4.0 GB |
| 10,000  **(peak)** | 1.366 ± 0.148 | 7,322 | 4.9 GB |

Peak ≈ **7,322 folds/s at batch 10,000** (200 nt). Throughput reaches ~90% of peak by batch **4,096** (~2.0 GB); larger batches add only a few % for proportionally more memory. **Recommended batch: 4,096–10,000** — use the lower end on memory-limited GPUs, the higher end when VRAM is free. Scale inversely with length, since memory per sequence grows as `(L+2)²`.

## 2. Length sweep (GPU, batch=2,048, 2048 seqs each)

Throughput vs sequence length. The partition function is O(L³) time / O(L²) memory, so folds/s falls steeply as sequences get longer.

| length (nt) | time (s) | folds/s |
|---:|---:|---:|
| 50 | 0.022 ± 0.000 | 91,527 |
| 100 | 0.071 ± 0.000 | 28,958 |
| 200 | 0.245 ± 0.001 | 8,355 |
| 300 | 0.828 ± 0.210 | 2,473 |
| 400 | 1.385 ± 0.150 | 1,479 |
| 500 | 2.492 ± 0.242 | 822 |

## 3. GPU vs CPU (10000 sequences x 200 nt)

Wall-clock time to compute logZ for all 10000 sequences.

| device | time (s) | folds/s | speedup vs 1-core |
|:--|---:|---:|---:|
| NVIDIA GeForce RTX 5090 (GPU, batch 10,000) | 1.291 ± 0.220 | 7,747 | 367.7× |
| CPU 1 core | 474.658 ± 28.038 | 21 | 1× (baseline) |
| CPU 32 cores | 39.407 ± 0.375 | 254 | 12.0× |

GPU is **367.7× a single CPU core** and **30.5× the full 32-thread CPU** on this workload.

> **Baseline note:** the CPU figures are this package's own NumPy/Numba `cpu.logZ` reference (~47 ms/fold single-core), *not* the optimized CONTRAfold C++ binary, which is considerably faster. These multiples are GPU vs. this reference implementation; the speedup over the C++ binary would be smaller. Multi-core scaling is sublinear (12.0× on 32 threads) because the i9-14900F mixes performance and efficiency cores with hyper-threading.

---

Reproduce:

```bash
python benchmarks/generate_sequences.py -n 10000 -l 200 -o benchmarks/data/sim_10k_200nt.fa
python benchmarks/compare.py benchmarks/data/sim_10k_200nt.fa --repeat 3
```
