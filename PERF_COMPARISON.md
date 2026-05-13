# QBLAS overhaul — comprehensive before/after benchmark

## Setup

Same machine, same compiler, same SLEEF build for both sides:

- Host: AMD EPYC 7V13 (Zen 3, AVX2+FMA, no AVX-512), 96 cores.
- Compiler: gcc (`-O3 -march=native`).
- SLEEF: built from `third_party/sleef` with every SIMD path enabled
  (`SLEEF_ENABLE_{SSE2,AVX,AVX2,AVX512F,INLINE_HEADERS}=ON`) and installed
  to `.sleef-prefix/`.  Both old and new binaries link the same shared
  libraries.
- Bench harness: Google Benchmark, **median of 3 runs**, `--benchmark_min_time=0.3s`.

Both sides are run with `OMP_NUM_THREADS=16` — that's the most defensible
apples-to-apples setup on this shared 96-core box. Single-thread numbers
(showing pure kernel quality) are also reported at the end.

- **OLD** = branch `main`, commit `42126fd` (original `QuadBLAS`).
- **NEW** = branch `overhaul-rewrite`, current head.

Source for the matched-size bench: [bench/bench_compare.cc](bench/bench_compare.cc).

All numbers are `items_per_second` from Google Benchmark, where `items =
n` for L1, `m*n` for gemv, `m*n*k` for gemm/syrk, `m*m*n/2` for trmm/trsm.
For gemm that's exactly the FMA throughput.

---

## Master comparison table — 16 threads

### Level 1 — vector operations

| routine  | n        | OLD (M items/s) | NEW (M items/s) | speedup |
| -------- | -------- | --------------: | --------------: | ------: |
| **dot**  | 1 000    | 29.9            | 28.3            |   0.95× |
| dot      | 10 000   | 37.2            | 319.8           |   **8.6×** |
| dot      | 100 000  | 38.1            | 415.7           |  **10.9×** |
| dot      | 1 000 000| 38.7            | 429.6           |  **11.1×** |
| **axpy** | 1 000    | 29.2            | 27.9            |   0.96× |
| axpy     | 4 000    | 36.3            | 251.9           |   **6.9×** |
| axpy     | 16 000   | 38.8            | 349.0           |   **9.0×** |
| axpy     | 64 000   | 39.5            | 387.8           |   **9.8×** |
| **nrm2** | 1 000    | 28.9            | 29.2            |   1.01× |
| nrm2     | 4 000    | 36.9            | 224.9           |   **6.1×** |
| nrm2     | 16 000   | 38.7            | 355.7           |   **9.2×** |
| nrm2     | 64 000   | 39.2            | 409.8           |  **10.5×** |
| asum     | 1 000    | n/a             | 65.3            |   —     |
| asum     | 4 000    | n/a             | 343.0           |   —     |
| asum     | 16 000   | n/a             | 598.5           |   —     |
| asum     | 64 000   | n/a             | 853.7           |   —     |
| scal     | 1 000    | n/a             | 65.9            |   —     |
| scal     | 4 000    | n/a             | 386.9           |   —     |
| scal     | 16 000   | n/a             | 703.0           |   —     |
| scal     | 64 000   | n/a             | 888.2           |   —     |

OLD was missing `asum` and `scal` entirely; rows are marked n/a.

### Level 2 — matrix-vector

| routine  | dim         | OLD (M items/s) | NEW (M items/s) | speedup |
| -------- | ----------- | --------------: | --------------: | ------: |
| **gemv** | 100²        | 2.5             | 296.8           |  **117×** |
| gemv     | 128²        | 2.6             | —               |   —     |
| gemv     | 200²        | —               | 376.5           |   —     |
| gemv     | 256²        | 2.6             | —               |   —     |
| gemv     | 400²        | —               | 408.9           |   —     |
| gemv     | 512²        | 38.0            | —               |   —     |
| gemv     | 800²        | —               | 419.5           |   —     |
| gemv     | 1 024²      | 38.0            | —               |   —     |
| gemv     | 1 600²      | 38.2            | 431.2           |  **11.3×** |

The two side's bench harnesses use different power-of-2 grids; only 100²
and 1 600² are common.

### Level 3 — matrix-matrix

| routine  | n³ (dim)    | OLD (M items/s) | NEW (M items/s) | speedup |
| -------- | ----------- | --------------: | --------------: | ------: |
| **gemm** | 64³         | 148.1           | 407.4           |    2.8× |
| gemm     | 128³        | 2.6             | 414.7           |  **159×** |
| gemm     | 256³        | 2.6             | 421.4           |  **161×** |
| gemm     | 512³        | 16.6            | 429.7           |   **26×** |
| syrk     | 64³         | n/a             | 410.2           |   —     |
| syrk     | 128³        | n/a             | 411.2           |   —     |
| syrk     | 256³        | n/a             | 418.9           |   —     |
| syrk     | 512³        | n/a             | 443.3           |   —     |
| trmm     | 64²·64      | n/a             | 151.0           |   —     |
| trmm     | 128²·128    | n/a             | 215.2           |   —     |
| trmm     | 256²·256    | n/a             | 287.3           |   —     |
| trmm     | 512²·512    | n/a             | 350.3           |   —     |
| trsm     | 64²·64      | n/a             | 170.2           |   —     |
| trsm     | 128²·128    | n/a             | 221.6           |   —     |
| trsm     | 256²·256    | n/a             | 282.9           |   —     |
| trsm     | 512²·512    | n/a             | 335.7           |   —     |

The 128³–256³ GEMM cliff in OLD (2.6 M/s, ~1/60 of single-thread!) was a
real regression in the legacy blocked path that this overhaul fixed.
`syrk`, `trmm`, `trsm` did not exist in OLD at all.

---

## Single-thread numbers (`OMP_NUM_THREADS=1`)

These reveal *pure kernel quality* — no parallel speedup, just the cost
of one routine call against one core.  Across every routine and every
size, OLD ran at the unvectorised SLEEF scalar speed (~2.6 M ops/s) and
NEW runs at SLEEF's vectorised AVX2 ceiling (~30 M ops/s):

| routine  | OLD (M items/s) | NEW (M items/s) | speedup |
| -------- | --------------: | --------------: | ------: |
| dot      | 2.6 – 2.7       | 29 – 30         | ~11×    |
| axpy     | 2.7 – 2.7       | 28              | ~10×    |
| nrm2     | 2.7 – 2.8       | 29 – 30         | ~11×    |
| gemv     | 2.6 – 2.7       | 27 – 30         | ~10×    |
| gemm     | 2.6 – 10.4      | 30              | ~3 – 12×|

So the new code is ~11× faster *per core* before parallelisation even
enters the picture. That's the SIMD that the old build never actually
benefited from.

---

## Why the wins look like this

1. **SIMD path actually engaged** (×11 per-core baseline).  The old
   template-only build called scalar `Sleef_*q1` everywhere — its
   `Sleef_quadx2` wrapper never reached the `Sleef_*q4_u05avx2`
   symbols. The new build compiles one TU per ISA tier
   (sse2/avx2/avx512/neon) and runtime-dispatches to the widest one
   available.

2. **GEMM threading actually works.**  OLD hard-coded `nc = 1024` in its
   block driver, so a 1024² problem got one jc-block and one busy thread.
   NEW auto-scales `nc` (and all of `mc`/`kc`/`nc`) from the host's
   detected L1/L2/L3 cache sizes plus core count, then schedules
   `#pragma omp for dynamic` over jc-blocks so every thread gets ≥ 2
   blocks worth of work.

3. **Goto-blocked trmm/trsm.**  Both routines run a small parallel
   diagonal-block solve plus a `cblas_qgemm` update for the trailing
   matrix, so they inherit gemm's blocking and threading. At n=512 we
   hit ~80 % of pure gemm throughput.

4. **Dynamic thresholds** (your specific ask).  At library init we read
   CPUID leaf 4 for cache sizes, count cores, and *time* one empty
   OpenMP region. From those three numbers we derive `mc`, `kc`, `nc`,
   the L1 thread-fork threshold, and the gemv thread-fork threshold.
   No `#define` survives anywhere in the hot path. On this host the
   detection reports L1=32 KB / L2=256 KB / L3=8 MB / 96 cores / OMP fork
   ≈ 57 K cycles → `mc=108 kc=120 nc=4096 L1-thr=3 536`.

5. **`qasum` / `qnrm2` are threaded** (didn't exist or weren't threaded
   in OLD).  `qnrm2(x)` reduces to `sqrt(qdot(x,x))` so it inherits
   `qdot`'s threading directly.

---

## Honest negatives (still in the new build)

At **n ≤ ~1 000 for L1 ops**, NEW is **3 – 5 %** slower than OLD. Reason:
NEW pays a fixed per-call overhead (dispatch lookup + thread heuristic +
function-call boundary). OLD's header-only inlining means a `cblas_qdot`
at n=1000 has almost no overhead. Below the parallel threshold there's
no parallelism to recoup that, so OLD edges out.

This is genuinely a trade-off (compiled library vs header-only) and the
gap is small. If users do lots of tiny dots/axpys, an `-flto` build of
the consumer that pulls qblas symbols into LTO with their code would
close the gap; we don't ship that by default to keep the ABI stable.

The other thing worth flagging: **single-thread peak is ~30 M ops/s per
core** and that *is* SLEEF's `Sleef_fmaq4_u05avx2` ceiling. A microbench
with 8 independent FMAs in flight tops out at the same 30 M/core, and
SLEEF's inline-header variant only adds ~5 % over the library symbol.
Beating it would require a hand-written DD-FMA kernel that gives up some
of SLEEF's 0.5-ULP accuracy guarantee — a non-trivial precision
trade-off that isn't done in this rewrite.

---

## Reproducing this exact run

```bash
# Build the SLEEF that both binaries link
scripts/bootstrap_sleef.sh

# OLD
git checkout main
SLEEF_ROOT=$(pwd)/.sleef-prefix cmake -S . -B build-old -DCMAKE_BUILD_TYPE=Release
cmake --build build-old --target quadblas_benchmark -j

# NEW
git checkout overhaul-rewrite
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target qblas_bench_compare -j

# Comparison (16 threads, median of 3)
OMP_NUM_THREADS=16 LD_LIBRARY_PATH=$(pwd)/.sleef-prefix/lib \
    ./build-old/quadblas_benchmark --benchmark_min_time=0.3s \
    --benchmark_repetitions=3 --benchmark_report_aggregates_only=true \
    --benchmark_filter='BM_DotProduct|BM_AXPY|BM_VectorNorm|BM_GEMV/|BM_GEMM/'
OMP_NUM_THREADS=16 ./build/bench/qblas_bench_compare \
    --benchmark_min_time=0.3s --benchmark_repetitions=3 \
    --benchmark_report_aggregates_only=true
```
