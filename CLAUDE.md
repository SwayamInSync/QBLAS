# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

QBLAS is a BLAS Level 1/2/3 library for IEEE-754 binary128 (quad-precision, `Sleef_quad`) built on top of SLEEF. It ships a compiled shared library `libqblas.so` with a CBLAS-style C API (`cblas_qdot`, `cblas_qgemm`, …) and runtime CPU dispatch (scalar / SSE2 / AVX2 / AVX-512 / NEON).

This repo was rewritten from scratch on the `overhaul-rewrite` branch; the older header-only template implementation on `main` is obsolete.

## Build

```bash
scripts/bootstrap_sleef.sh           # builds SLEEF 3.8 with all SIMD paths
                                     # and installs into ./.sleef-prefix/
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
ctest                                # ~5 s, 5 test executables
./bench/qblas_bench --benchmark_min_time=0.1s
```

`scripts/bootstrap_sleef.sh` forces `SLEEF_ENABLE_{SSE2,SSE4,AVX,AVX2,AVX512F}=ON` so every dispatch tier the binary contains has its underlying SLEEF symbols available; SLEEF's default config only builds for the host CPU and leaves unresolved references for other tiers in `libsleefquad.so`.

The top-level CMake auto-discovers SLEEF via `SLEEF_ROOT` (env var or cmake var), then `./.sleef-prefix/`, then pkg-config. The binary RPATH is set with `--disable-new-dtags` so `libtlfloat.so.1` (an indirect dep of `libsleefquad.so`) resolves; without that, GLIBC's DT_RUNPATH semantics drop the path for transitive deps.

## Architecture

The library is a single shared object with multiple internal "kernel tiers", each compiled as an OBJECT library with its own `-march` flags. Runtime dispatch picks the best available tier at library init time.

```
include/qblas/         Public C API. Each cblas_q* declaration ties to a
                       Sleef_quad function (no template metaprogramming).
src/common/            Internal helpers (dispatch table, threading,
                       aligned alloc, branch hints).
src/cpu/               CPUID-based feature detection + dispatch init.
                       Reads QBLAS_DISPATCH=avx2|sse2|generic to override.
src/kernels/           One TU per tier (generic/sse2/avx2/avx512/neon).
                       Each defines QV_WIDTH + QV_ISA_SUFFIX and includes
                       kernels_template.h, which contains the actual SIMD
                       kernels parametrised by width and ISA suffix.
src/level1/            Front-end for vec ops (delegates to dispatch + threads).
src/level2/            Front-end for gemv/ger/symv/trsv.
src/level3/            qgemm.c (Goto-style blocking) + qblas_level3_extra.c
                       (syrk/trmm/trsm; the last two are reference, not yet
                       blocked).
tests/                 Smoke + per-level correctness suites. Each test compares
                       against a naive scalar reference across positive and
                       negative strides, row/col-major layouts, all transpose
                       combos, and edge tile sizes (m=1, k=1, prime, multi-block).
bench/                 Google Benchmark suite (BM_qdot, BM_qgemm, …).
                       items/sec ≈ quad FMA throughput.
cmake/QBLASArch.cmake  Per-tier compile flags. Edit here to add new ISAs.
```

### Kernel template ([src/kernels/kernels_template.h](src/kernels/kernels_template.h))

A single header parametrised over `QV_WIDTH` (1, 2, 4, 8) and `QV_ISA_SUFFIX` (`sse2`, `advsimd`, …). For widths 1/4/8 there is only one SLEEF symbol set so the suffix is implicit; for width 2 we have to pick between `sse2` (x86) and `advsimd` (ARM) at the call site, which is why `QV_ISA_SUFFIX` is its own macro. Each ISA TU defines those macros and `#include`s the template — no source duplication.

The template implements: `qdot`, `qaxpy`, `qscal`, `qasum`, `qiamax`, `qgemv_n`, `qgemv_t`, and the packed `qgemm_kernel`. The corresponding `qblas_register_<tier>` function publishes its function pointers into the global dispatch table.

### GEMM ([src/level3/qblas_gemm.c](src/level3/qblas_gemm.c))

Goto / van de Geijn 5-loop blocking around a register tile published by the dispatch tier (MR×NR = 4×4 currently). Key correctness trick: **column-major C is computed as row-major C^T by swapping M↔N and A↔B with original (not flipped) transpose flags** — the buffer addresses come out identical because additions commute. This eliminated a separate col-major code path and a class of off-by-something bugs.

Parallelism is over the outermost `jc` loop. `nc` is auto-scaled down so each thread gets ≥2 blocks (`target_blocks = nthreads * 2`); without this, large square GEMMs ran single-threaded because `nc=QBLAS_NC_DEFAULT (512)` swallowed the whole problem.

### Threading

OpenMP. Each Level 1 entry point checks `qblas_resolve_threads()` (a work-vs-overhead heuristic) and only forks above `QBLAS_PARALLEL_THRESHOLD_L1 = 8192`. Reduction-style ops (`qdot`, `qasum`, `qnrm2`) sum per-thread partials into a stack array (max 256 threads); `qnrm2` is implemented as `sqrt(qdot(x,x))` so it inherits the threading for free.

### Dispatch override

Set `QBLAS_DISPATCH=generic|sse2|avx2|avx512|neon` to force a tier — useful for comparing per-tier throughput in the bench and for diagnosing tier-specific bugs. The auto-selected tier is printed by `qblas_get_dispatch_tier()`.

## Common gotchas

- **Negative strides**: BLAS lets `incx < 0` walk a vector in reverse. The convention here: the caller (in `cblas_q*` front-ends) shifts `x` to its logical first element via `neg_offset(incx, n)`. The kernel iterates `ix = 0; ix += incx` from there — **never re-applies** the offset. There's a class of bugs where the offset gets applied twice (caller + kernel), so always cross-check the kernel's strided branch.

- **SLEEF naming**: ULP-bounded ops (add, fma, mul, sqrt) use `Sleef_<op>q<W>_u05<isa>` (no underscore before isa) — e.g. `Sleef_fmaq4_u05avx2`. Non-ULP ops (splat, load, store, get, fabs, neg) use `Sleef_<op>q<W>_<isa>` — e.g. `Sleef_storeq2_sse2`. The template's `QV_NAME` macro encodes this.

- **`Sleef_storeq2` / `Sleef_loadq2` (no ISA suffix) are not exported** even though declared. Always use explicit-ISA suffixes for load/store.

- **`tlfloat.so` is not installed by SLEEF's `cmake --install`** even when it builds. The bootstrap script copies it manually into `.sleef-prefix/lib/`.

## Performance baseline (AMD EPYC 7V13, 96 cores, AVX2 tier)

| Routine | Size | Throughput (items/s, multi-thread) |
| ------- | ---- | --------- |
| qdot    | 1 M  | 836 M     |
| qaxpy   | 4 M  | 2.15 G    |
| qscal   | 1 M  | 1.90 G    |
| qnrm2   | 4 M  | 1.29 G    |
| qasum   | 1 M  | 1.85 G    |
| qgemv   | 4096² | 935 M    |
| qgemv_t | 4096² | 822 M    |
| qgemm   | 1024³ | **1.61 G FMA** |
| qsyrk   | 1024² | 1.60 G FMA |

Single-thread GEMM peak is ~30 MFMA/s; the parallel speedup is ~55× across 96 cores (≈57 % efficiency). The remaining gap is mostly per-quad-FMA latency inside SLEEF's `Sleef_fmaq4_u05avx2` (software DD over AVX2 doubles), not our blocking — further gains would need a hand-written micro-kernel that interleaves the DD doubles across multiple FMAs.

## Repo layout reference

- [include/qblas/qblas.h](include/qblas/qblas.h) — umbrella public header (pulls in level1/2/3)
- [src/CMakeLists.txt](src/CMakeLists.txt) — `qblas_add_tier()` macro for per-ISA OBJECT libs
- [src/cpu/qblas_cpu.c](src/cpu/qblas_cpu.c) — CPUID detection + `__attribute__((constructor))` library init
- [src/kernels/kernels_template.h](src/kernels/kernels_template.h) — the generic SIMD kernel template
- [scripts/bootstrap_sleef.sh](scripts/bootstrap_sleef.sh) — required first-time setup
