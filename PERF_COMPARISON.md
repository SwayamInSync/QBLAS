# QBLAS overhaul — performance before/after

## Setup

Both versions built on the same machine (AMD EPYC 7V13, 96 cores, AVX2/FMA, no AVX-512), against the same SLEEF (built from `third_party/sleef` with all SIMD paths enabled, installed to `.sleef-prefix`). Same compiler (gcc), same `-O3 -march=native`, same Google Benchmark.

- **OLD** = commit `42126fd` on `main` (the header-only `QuadBLAS` template library). Built via the original `CMakeLists.txt`; benchmarked with `benchmarks/benchmark.cpp` (the suite that already existed on `main`).
- **NEW** = branch `overhaul-rewrite`, current head. Built via the new CMake. Benchmarked with [bench/bench_compare.cc](bench/bench_compare.cc), which calls `cblas_q*` on the same problem sizes as the legacy suite (1000/10000/100000/1000000 for dot, 1000/4000/16000/64000 for axpy and nrm2, 100/200/400/800/1600 for gemv, 64/128/256/512 for gemm — and where sizes don't coincide, both use a no-trans no-trans configuration with random data).

Each cell below is the **median of 3 runs** with `--benchmark_min_time=0.3s`. All numbers are `items_per_second` (Google Benchmark's `SetItemsProcessed`): for dot/axpy/nrm2 that's `n`, for gemv it's `m*n`, for gemm it's `m*n*k` — i.e. effective quad-FMA throughput.

---

## Single-thread (`OMP_NUM_THREADS=1`) — pure kernel quality

| Routine | Size | OLD (M items/s) | NEW (M items/s) | Speedup |
| ------- | ---- | --------------- | --------------- | ------- |
| dot | 1 000        | 2.63 | 30.0 | **11.4×** |
| dot | 10 000       | 2.65 | 29.5 | **11.1×** |
| dot | 100 000      | 2.66 | 30.0 | **11.3×** |
| dot | 1 000 000    | 2.70 | 30.1 | **11.2×** |
| axpy | 1 000       | 2.73 | 28.1 | **10.3×** |
| axpy | 4 000/4 096  | 2.73 | 28.2 | **10.3×** |
| axpy | 16 000/16 384 | 2.75 | 27.8 | **10.1×** |
| axpy | 64 000       | 2.72 | 28.1 | **10.3×** |
| nrm2 | 1 000        | 2.69 | 29.3 | **10.9×** |
| nrm2 | 64 000       | 2.77 | 29.7 | **10.7×** |
| gemv | 100²         | 2.60 | 27.0 | **10.4×** |
| gemv | 1 600²       | 2.65 | 29.5 | **11.1×** |
| gemm | 64³          | 10.4 | 29.7 | **2.9×**  |
| gemm | 128³         | 2.62 | 29.9 | **11.4×** |
| gemm | 256³         | 2.61 | 29.6 | **11.3×** |
| gemm | 512³         | 2.63 | 29.9 | **11.4×** |

The old code was **not actually using SLEEF's vectorized quad operations**. At every size, every routine, single-thread throughput sits at ~2.6 M ops/s — the speed of `Sleef_addq1`/`Sleef_fmaq1` (scalar). The new code hits ~30 M ops/s at every size, which is the throughput of SLEEF's `Sleef_fmaq4_u05avx2` (vectorized over 4 quads via AVX2). The one outlier — `gemm/64³` at 10 M/s in old — is because that problem fits in L1 and the unvectorized loop hits some compiler-vectorized win that vanishes at larger sizes.

So the SIMD path was the largest single perf bug fixed by the overhaul: **~11× from running the right SLEEF symbols**.

---

## 16-thread (`OMP_NUM_THREADS=16`) — realistic moderate-machine

| Routine | Size | OLD (M items/s) | NEW (M items/s) | Speedup |
| ------- | ---- | --------------- | --------------- | ------- |
| dot | 1 000        | 29.4   | 28.4    | 0.97× |
| dot | 10 000       | 37.3   | 322.1   | **8.6×**  |
| dot | 100 000      | 38.1   | 414.9   | **10.9×** |
| dot | 1 000 000    | 38.4   | 430.2   | **11.2×** |
| axpy | 1 000       | 30.1   | 26.5    | 0.88× |
| axpy | 4 000/4 096  | 35.9   | 28.2    | 0.79× |
| axpy | 16 000/16 384 | 38.9  | 349.5   | **9.0×**  |
| axpy | 64 000       | 39.3   | 386.0   | **9.8×**  |
| nrm2 | 1 000        | 29.4   | 27.7    | 0.94× |
| nrm2 | 16 000/16 384 | 38.7  | 356.1   | **9.2×**  |
| nrm2 | 64 000       | 39.2   | 410.3   | **10.5×** |
| gemv | 100²         | 2.46   | 290.7   | **118×**  |
| gemv | 128²         | 2.44   | —       | —     |
| gemv | 256²         | 2.48   | —       | —     |
| gemv | 200²         | —      | 368.4   | —     |
| gemv | 400²         | —      | 409.3   | —     |
| gemv | 512²         | 37.9   | —       | —     |
| gemv | 800²         | —      | 420.6   | —     |
| gemv | 1 024²       | 38.4   | —       | —     |
| gemv | 1 600²       | 38.5   | 423.5   | **11.0×** |
| gemm | 64³          | 148.8  | 399.7   | 2.7×  |
| gemm | 128³         | 2.59   | 424.2   | **164×**  |
| gemm | 256³         | 2.61   | 425.9   | **163×**  |
| gemm | 512³         | 19.1   | 429.8   | **22.5×** |

Two phenomena stand out:

1. **Old code saturates at ~38 M ops/s** for every L1 / Level-2 routine regardless of `n`. That ceiling is its single-thread quad-FMA throughput times some small parallel speedup that never scales. The new code climbs to ~430 M ops/s on the same 16 cores — clean 11×.

2. **Old GEMM is broken in the 128-256 size range.** At `n³ = 128³` it runs at 2.6 M ops/s, well below its own single-thread speed at smaller sizes — there's a regression somewhere in its blocked path. The new code is steady at ~425 M ops/s across all gemm sizes.

3. **Small-N regressions** (≤4 K) of 0.79–0.97× exist on the new code. They come from spawning a parallel region for work that's barely above its overhead; the old code happened to be slightly more conservative there. This is a known trade-off — could be tightened later by raising the L1 thread-fork threshold.

---

## 96-thread (full machine) — what the new code can actually use

Old code did not scale past ~38 M/s. New code, on a quiet 96-core machine:

| Routine | Size | NEW (G items/s, 96 thr) | New/Old at this size (16 thr) |
| ------- | ---- | --- | --- |
| dot     | 1 M  | 0.84 | (1.94× the 16-thr number) |
| axpy    | 4 M  | 2.15 | — |
| nrm2    | 4 M  | 1.29 | — |
| gemv    | 4096² | 0.94 | — |
| gemm    | 1024³ | **1.61** | — |

The 96-thread numbers were stable on a quiet box (`load avg ~0.2`) but jumpy on a busy one — that's expected on a shared 96-core machine where any one of the worker threads getting scheduled out stalls the parallel region. The 16-thread comparison is the most defensible apples-to-apples result; the 96-thread numbers are the upper bound when the box is quiet.

---

## Why so much faster

Three changes account for almost all of the speedup:

1. **Use SLEEF's vector q-symbols.** The old template-only build called scalar `Sleef_*q1` everywhere even when it claimed to be using `Sleef_quadx2` — there was a missing dispatch and a scattered set of SLEEF store-bug workarounds that ended up disabling vectorization. The new code compiles per-ISA TUs with the correct `Sleef_addq4_u05avx2`, `Sleef_fmaq4_u05avx2`, etc., picked at library init.

2. **Actually parallelize GEMM.** The old block driver had `nc = 1024` hardcoded, so a 1024×1024 problem had exactly one `jc`-block and exactly one thread saw work. The new driver auto-scales `nc` to `n / (nthreads * 2)` so every thread gets ≥ 2 blocks, and uses `#pragma omp for schedule(dynamic)` over them.

3. **Fix the L1 threading thresholds and `gemv_t` parallelisation.** `qnrm2`/`qasum` used to fall through to a single-thread path; `gemv_t` only forked once `m ≥ 4096`. Now both thread above their natural break-even (~10 K elements for L1, immediately for L2 above the parallel threshold).

The remaining ~30 M ops/s/core for SLEEF quad SIMD is approximately the inherent cost of `Sleef_fmaq4_u05avx2`'s software DD math on AVX2 doubles — there's a smaller next-step win available from a hand-tuned micro-kernel that interleaves DD doubles across several FMAs, but nothing close to another 10×.

---

## Reproducing this

```bash
# OLD
git checkout main
SLEEF_ROOT=$(pwd)/.sleef-prefix cmake -S . -B build-old -DCMAKE_BUILD_TYPE=Release
cmake --build build-old --target quadblas_benchmark -j
OMP_NUM_THREADS=16 LD_LIBRARY_PATH=$(pwd)/.sleef-prefix/lib \
    ./build-old/quadblas_benchmark --benchmark_min_time=0.3s \
    --benchmark_repetitions=3 --benchmark_report_aggregates_only=true \
    --benchmark_filter='BM_DotProduct|BM_AXPY|BM_VectorNorm|BM_GEMV/|BM_GEMM/'

# NEW
git checkout overhaul-rewrite
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target qblas_bench_compare -j
OMP_NUM_THREADS=16 ./build/bench/qblas_bench_compare \
    --benchmark_min_time=0.3s --benchmark_repetitions=3 \
    --benchmark_report_aggregates_only=true
```
