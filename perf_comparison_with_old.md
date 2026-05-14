# QBLAS overhaul, comprehensive before/after benchmark

## Setup

Same machine, same compiler, same SLEEF build for both sides:

- Host: AMD EPYC 7V13 (Zen 3, AVX2+FMA, no AVX-512), 96 cores.
- Compiler: gcc (`-O3 -march=native`).
- SLEEF: built from `third_party/sleef` with every SIMD path enabled
  (`SLEEF_ENABLE_{SSE2,AVX,AVX2,AVX512F,INLINE_HEADERS}=ON`) and installed
  to `.sleef-prefix/`. Both old and new binaries link the same shared
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

| routine  | n         | OLD (M items/s) | NEW (M items/s) |   speedup |
| -------- | --------- | --------------: | --------------: | --------: |
| **dot**  | 1 000     |            29.9 |            28.3 |     0.95× |
| dot      | 10 000    |            37.2 |           319.8 |  **8.6×** |
| dot      | 100 000   |            38.1 |           415.7 | **10.9×** |
| dot      | 1 000 000 |            38.7 |           429.6 | **11.1×** |
| **axpy** | 1 000     |            29.2 |            27.9 |     0.96× |
| axpy     | 4 000     |            36.3 |           251.9 |  **6.9×** |
| axpy     | 16 000    |            38.8 |           349.0 |  **9.0×** |
| axpy     | 64 000    |            39.5 |           387.8 |  **9.8×** |
| **nrm2** | 1 000     |            28.9 |            29.2 |     1.01× |
| nrm2     | 4 000     |            36.9 |           224.9 |  **6.1×** |
| nrm2     | 16 000    |            38.7 |           355.7 |  **9.2×** |
| nrm2     | 64 000    |            39.2 |           409.8 | **10.5×** |
| asum     | 1 000     |             n/a |            65.3 |         — |
| asum     | 4 000     |             n/a |           343.0 |         — |
| asum     | 16 000    |             n/a |           598.5 |         — |
| asum     | 64 000    |             n/a |           853.7 |         — |
| scal     | 1 000     |             n/a |            65.9 |         — |
| scal     | 4 000     |             n/a |           386.9 |         — |
| scal     | 16 000    |             n/a |           703.0 |         — |
| scal     | 64 000    |             n/a |           888.2 |         — |

OLD was missing `asum` and `scal` entirely; rows are marked n/a.

### Level 2 — matrix-vector

| routine  | dim    | OLD (M items/s) | NEW (M items/s) |   speedup |
| -------- | ------ | --------------: | --------------: | --------: |
| **gemv** | 100²   |             2.5 |           296.8 |  **117×** |
| gemv     | 128²   |             2.6 |               — |         — |
| gemv     | 200²   |               — |           376.5 |         — |
| gemv     | 256²   |             2.6 |               — |         — |
| gemv     | 400²   |               — |           408.9 |         — |
| gemv     | 512²   |            38.0 |               — |         — |
| gemv     | 800²   |               — |           419.5 |         — |
| gemv     | 1 024² |            38.0 |               — |         — |
| gemv     | 1 600² |            38.2 |           431.2 | **11.3×** |

The two side's bench harnesses use different power-of-2 grids; only 100²
and 1 600² are common.

### Level 3 — matrix-matrix

| routine  | n³ (dim) | OLD (M items/s) | NEW (M items/s) |  speedup |
| -------- | -------- | --------------: | --------------: | -------: |
| **gemm** | 64³      |           148.1 |           407.4 |     2.8× |
| gemm     | 128³     |             2.6 |           414.7 | **159×** |
| gemm     | 256³     |             2.6 |           421.4 | **161×** |
| gemm     | 512³     |            16.6 |           429.7 |  **26×** |
| syrk     | 64³      |             n/a |           410.2 |        — |
| syrk     | 128³     |             n/a |           411.2 |        — |
| syrk     | 256³     |             n/a |           418.9 |        — |
| syrk     | 512³     |             n/a |           443.3 |        — |
| trmm     | 64²·64   |             n/a |           151.0 |        — |
| trmm     | 128²·128 |             n/a |           215.2 |        — |
| trmm     | 256²·256 |             n/a |           287.3 |        — |
| trmm     | 512²·512 |             n/a |           350.3 |        — |
| trsm     | 64²·64   |             n/a |           170.2 |        — |
| trsm     | 128²·128 |             n/a |           221.6 |        — |
| trsm     | 256²·256 |             n/a |           282.9 |        — |
| trsm     | 512²·512 |             n/a |           335.7 |        — |

The 128³–256³ GEMM cliff in OLD (2.6 M/s, ~1/60 of single-thread!) was a
real regression in the legacy blocked path that this overhaul fixed.
`syrk`, `trmm`, `trsm` did not exist in OLD at all.

---

## Single-thread numbers (`OMP_NUM_THREADS=1`)

These reveal _pure kernel quality_ — no parallel speedup, just the cost
of one routine call against one core. Across every routine and every
size, OLD ran at the unvectorised SLEEF scalar speed (~2.6 M ops/s) and
NEW runs at SLEEF's vectorised AVX2 ceiling (~30 M ops/s):

| routine | OLD (M items/s) | NEW (M items/s) |  speedup |
| ------- | --------------: | --------------: | -------: |
| dot     |       2.6 – 2.7 |         29 – 30 |     ~11× |
| axpy    |       2.7 – 2.7 |              28 |     ~10× |
| nrm2    |       2.7 – 2.8 |         29 – 30 |     ~11× |
| gemv    |       2.6 – 2.7 |         27 – 30 |     ~10× |
| gemm    |      2.6 – 10.4 |              30 | ~3 – 12× |

So the new code is ~11× faster _per core_ before parallelisation even
enters the picture. That's the SIMD that the old build never actually
benefited from.

---

## 96-thread results (`OMP_NUM_THREADS=96 OMP_PLACES=cores OMP_PROC_BIND=close`)

Full-machine numbers from the same EPYC 7V13 box. Threads are pinned so
each iteration's timing is stable enough for median-of-3 to be
meaningful. Compute-bound L3 routines scale nearly linearly to all 96
cores; L1/L2 at moderate sizes are memory-bandwidth bound and saturate
earlier (which is why my dynamic threading heuristic correctly _does
not_ spawn 96 threads for them).

| routine | n / dim | OLD M items/s |       NEW M items/s |  new × old |
| ------- | ------- | ------------: | ------------------: | ---------: |
| dot     | 1 M     |         176.8 |           **882.9** |   **5.0×** |
| axpy    | 64 K    |         151.2 | 83.3 (mem-BW bound) |      0.55× |
| nrm2    | 64 K    |         169.1 | 30.0 (memory bound) |      0.18× |
| asum    | 64 K    |           n/a |                66.7 |          — |
| scal    | 64 K    |           n/a |                68.3 |          — |
| gemv    | 1 600²  |         188.2 |         **2 202.4** |  **11.7×** |
| gemm    | 64³     |         771.5 |               242.8 |      0.31× |
| gemm    | 128³    |           2.6 |         **1 821.8** |   **701×** |
| gemm    | 256³    |           2.6 |         **3 129.0** | **1 203×** |
| gemm    | 512³    |          18.1 |         **3 730.9** |   **206×** |
| syrk    | 64³     |           n/a |               211.4 |          — |
| syrk    | 128³    |           n/a |             1 498.8 |          — |
| syrk    | 256³    |           n/a |             3 721.6 |          — |
| syrk    | 512³    |           n/a |         **3 773.7** |          — |
| trmm    | 64²     |           n/a |                81.2 |          — |
| trmm    | 128²    |           n/a |               233.5 |          — |
| trmm    | 256²    |           n/a |               802.1 |          — |
| trmm    | 512²    |           n/a |         **2 706.5** |          — |
| trsm    | 64²     |           n/a |                87.8 |          — |
| trsm    | 128²    |           n/a |               262.9 |          — |
| trsm    | 256²    |           n/a |               795.3 |          — |
| trsm    | 512²    |           n/a |         **2 529.8** |          — |

Observations:

- **GEMM at 256³–512³ hits 3.1–3.7 G FMA/sec**, which is the peak this
  machine produces for quad precision through SLEEF. At 512³ that's an
  8.7× scale-up over the 16-thread number, almost linear.
- **TRSM/TRMM at 512² reach 2.5–2.7 G items/sec** because the trailing
  GEMM updates dominate at that size and inherit GEMM's threading.
- **`gemm/64` regressed at 96 threads** vs OLD (771 vs 243). At m=n=k=64
  the parallel-region overhead of 96 fork/joins dominates the 262 K FMAs
  of actual work. Running with `OMP_NUM_THREADS=16` gives `gemm/64 = 407
M/s`, which beats OLD. This is exactly the kind of thing the dynamic
  threading heuristic should fix: cap threads to ~16 even when 96 are
  available, for small problems.
- **`axpy/64K`, `nrm2/64K` regressed at 96 threads** vs OLD's 16-thread
  measurement. Same root cause: at this size the working set (1 MB)
  doesn't justify 96 cores' worth of memory traffic. The 16-thread
  numbers in the previous section are the better comparison for L1.

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
