# Changelog

All notable changes to QBLAS are documented in this file. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); QBLAS adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.5.0] — 2026-05-14

Full rewrite of QBLAS from the legacy header-only C++ template
implementation (1.0) to a compiled shared library with a stable
CBLAS-style C ABI, runtime CPU dispatch, and OpenMP-backed parallelism.

### Highlights

Measured on AMD EPYC 7V13 (Zen 3, AVX2 tier), same f64 baseline
(scipy-openblas Haswell tier), same harness on both versions:

| metric                                       | 1.0      | 1.5      |
|----------------------------------------------|----------|----------|
| single-thread gemm slowdown vs OpenBLAS      | ~1 400×  | ~800×    |
| 16-thread gemm @ n=256 slowdown vs OpenBLAS  | 7 072×   | 287×     |
| single-core kernel speedup                   | baseline | 1.6-1.7× |
| threading at small N (≤256)                  | broken*  | working  |

\* Old code used a fixed `nc = NC_DEFAULT = 512`, so 128×128 gemm at
16 threads ran on a single thread. New code auto-scales `nc` so each
thread gets ≥ 2 blocks.

Full before/after tables in
[perf_comparison_with_old.md](perf_comparison_with_old.md);
reproduction harness at [bench/bench_quad_vs_numpy.py](bench/bench_quad_vs_numpy.py).

### Added

- CBLAS-style C ABI: `cblas_q{dot,nrm2,asum,axpy,scal,gemv,ger,gemm,syrk,trmm,trsm}`
  in `include/qblas/qblas.h`. Mirrors the standard CBLAS surface for the
  binary128 `Sleef_quad` type, including transpose / layout / conj flags
  (previously rejected by old `QuadBLAS::gemm`).
- Per-ISA OBJECT libraries: `qblas_kernels_{generic,sse2,avx2,avx512,neon}`,
  all generated from one parametrised template
  ([src/kernels/kernels_template.h](src/kernels/kernels_template.h)).
- Runtime CPU dispatch via CPUID at library init; override with
  `QBLAS_DISPATCH={generic,sse2,avx2,avx512,neon}`. Falls back safely if
  asked to enable a tier the host doesn't support.
- Goto/van-de-Geijn 5-loop blocked GEMM with dynamic `mc / kc / nc` sized
  from detected L1 / L2 / L3 cache.
- Blocked TRSM and TRMM with parallelised diagonal solve (5-15× over the
  reference path at all sizes).
- OpenMP threading with thresholds detected at init time (measures actual
  parallel-region overhead rather than hardcoding 8 192).
- meson.build for use as a Meson subproject by downstream packages
  (numpy-quaddtype integration verified on branch `qblas-rewrite-integration`).
- Numpy-vs-QBLAS correctness test (85 cases, rounded-to-double comparison).
- Google Benchmark suite (`qblas_bench`, `qblas_bench_compare`).
- Cross-dtype throughput harness ([bench/bench_quad_vs_numpy.py](bench/bench_quad_vs_numpy.py))
  for honest reproduction of the perf-comparison numbers.

### Changed

- **Breaking**: C++ template API removed. Consumers now link against
  `libqblas` and `#include <qblas/qblas.h>`. The old
  `QuadBLAS::gemm`/`dot`/`gemv` C++ free-function surface is gone.
- **Breaking**: function signatures take `Sleef_quad` by value (or by
  pointer in shim layers for ctypes/libffi callers — see
  `tests/python/qblas_shim.c` for the pattern).
- `qblas_set_num_threads()` no longer mutates global OpenMP state; it
  only caps QBLAS-internal scheduling. Callers that want process-wide
  thread-count changes should use `OMP_NUM_THREADS` / `omp_set_num_threads`.

### Removed

- Orphaned `third_party/OpenBLAS` submodule (was never referenced in any
  build). Drops ~200 MB from `git clone --recurse-submodules`.

### Known limitations

- **Windows / MSVC is not supported.** The kernel template uses GCC-only
  flags (`-march`, `-mavx2`), POSIX-only APIs (`clock_gettime`, `sysconf`,
  `_SC_LEVEL2_CACHE_SIZE`), and GCC built-ins for CPUID. Downstream
  packages can gate QBLAS off via their build system (numpy-quaddtype
  uses `-Ddisable_quadblas=true` automatically on Windows).
- **AVX-512 tier is built but never run on hardware in CI** — the bench
  host is Zen 3 (no AVX-512). The dispatcher selects it correctly on
  Sapphire Rapids / Zen 4, but those paths are uncovered. Tracking:
  [performance_bottlenecks.md §3](performance_bottlenecks.md).
- **Single-thread quad-FMA throughput is capped at SLEEF's TD-FMA
  ceiling** (~60 cycles per FMA on Zen 3, ~800× the f64 cost). Not a
  blocker for 1.5; the optimisation plan to bring this to ~100× lives in
  [performance_bottlenecks.md §1](performance_bottlenecks.md).
- **Level-2 `qger` / `qsymv` / `qtrsv` still drop through to a
  scalar+dispatched-axpy path** (qgemv is fully vectorised + threaded).
  [performance_bottlenecks.md §4](performance_bottlenecks.md).

### CI

Matrix: `ubuntu-latest` × {gcc, clang} × `macos-14` × `macos-15`.
Per-tier dispatch test on Linux (`QBLAS_DISPATCH={generic,sse2,avx2}`).
Numpy comparison test on every push. Benchmark artifact uploaded by the
`bench-baseline` job.

---

## [1.0.0] — pre-2026

The original header-only C++ template implementation. Compiled into the
consumer at use-site, x86-64 path hardcoded to SSE2, no transpose
support in `gemm`, no runtime dispatch.

This entry is here for reference only; the 1.0 line is no longer
maintained.
