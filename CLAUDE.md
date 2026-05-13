# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

QuadBLAS is a header-only C++17 library, but the repository builds test and benchmark executables via CMake. The build depends on **SLEEF** (with quad support) and optionally OpenMP + Google Benchmark (vendored as submodules under `third_party/`).

```bash
git submodule update --init --recursive          # needed for Google Benchmark
export SLEEF_ROOT=/path/to/sleef/installation    # points at a prefix containing include/ and lib/
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

If `SLEEF_ROOT` is unset, CMake falls back to `pkg-config` to locate SLEEF. CI installs SLEEF 3.8 from source with `-DSLEEF_BUILD_QUAD=ON -DSLEEF_BUILD_SHARED_LIBS=ON` into `/usr/local`. Release builds use `-O3 -march=native -ffast-math`; the Debug config enables ASan.

Note: `build.sh` hardcodes `SLEEF_ROOT=/Users/swayam/Desktop/numpy_dtypes/sleef/build` (the author's machine). It also runs `rm -rf build` at the start. Edit `SLEEF_ROOT` before using it locally, or skip it and run the cmake commands above.

## Test and benchmark

The build produces three executables in `build/`:

- `quadblas_test` — the full correctness suite from [test_quadblas.cpp](test_quadblas.cpp). It exits non-zero on any failure. There is no per-test selector; tests are grouped into `test_level1_blas`, `test_level2_blas`, `test_level3_blas`, `test_edge_cases` and run unconditionally from `main`. To run a single group, comment out the others in `main`.
- `quadblas_benchmark` — Google Benchmark suite from [benchmarks/benchmark.cpp](benchmarks/benchmark.cpp). Filter with `--benchmark_filter='BM_DotProduct|BM_GEMM|...'`. Available cases: `BM_DotProduct`, `BM_VectorNorm`, `BM_AXPY`, `BM_GEMV`, `BM_RowMajorGEMV`, `BM_ColMajorGEMV`, `BM_GEMM`, `BM_ThreadingScalability`, `BM_NumericalPrecision`.
- `quadblas_benchmark_legacy` — standalone benchmark using `std::chrono`, no GBench needed.

A separate ad-hoc benchmark exists at [perf_compare.cpp](perf_compare.cpp) but is not wired into CMake.

## Architecture

QuadBLAS implements Level 1/2/3 BLAS for IEEE 754 binary128 (`Sleef_quad`) on top of SLEEF's vectorized quad-precision primitives. The public header is [include/quadblas/quadblas.hpp](include/quadblas/quadblas.hpp), which transitively includes everything in dependency order.

Headers under [include/quadblas/](include/quadblas/) are layered — `core` → `memory` → `simd` → `threading` → `detail` → `algorithms` → `interface`. New code should respect this ordering (lower layers must not include higher ones).

Key cross-cutting concerns:

- **Platform abstraction** lives in [core/platform.hpp](include/quadblas/core/platform.hpp): defines `QUADBLAS_X86_64` or `QUADBLAS_AARCH64` and includes `immintrin.h` / `arm_neon.h` accordingly. The `QuadVector` SIMD wrapper in [simd/quad_vector.hpp](include/quadblas/simd/quad_vector.hpp) dispatches between `Sleef_*_sse2` and `Sleef_*_advsimd` variants per platform, with a scalar fallback. There are intentional workarounds for SLEEF store bugs (load uses `Sleef_loadq2_*`, but stores use `Sleef_getq2_*` instead of `Sleef_storeq2_*`) — preserve those.
- **GEMM blocking** in [detail/blocking.hpp](include/quadblas/detail/blocking.hpp) computes `mc/kc/nc` panel sizes from L1/L2 cache estimates. Apple Silicon overrides are hardcoded. Tile constants `GEMM_MR = GEMM_NR = 4` in [algorithms/level3.hpp](include/quadblas/algorithms/level3.hpp) must stay in sync with the `MR/NR` constants in `BlockingParams`.
- **Threading** uses OpenMP guarded by `_OPENMP`, with a `PARALLEL_THRESHOLD` (in [core/constants.hpp](include/quadblas/core/constants.hpp)) that gates when work is parallelized.
- **Two public interfaces** in `interface/`: [c_interface.hpp](include/quadblas/interface/c_interface.hpp) exposes `quadblas_q{dot,nrm2,axpy,gemv,gemm}` taking `void*` quad buffers and `double` alpha/beta — designed for Python/NumPy interop. [cpp_classes.hpp](include/quadblas/interface/cpp_classes.hpp) exposes `Vector<Layout>` / `Matrix<Layout>` templates with member methods. The C interface in `quadblas_qgemm` currently ignores `transa`/`transb` — calls with `'T'`/`'C'` silently behave like `'N'`.

## CI

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs on Ubuntu and macOS 14/15. It builds SLEEF 3.8 from source each run, then builds and runs `quadblas_test` plus a subset of benchmarks (`BM_DotProduct|BM_AXPY|BM_VectorNorm`). When making changes that affect platform-specific SIMD, expect both Ubuntu (x86-64) and macOS (ARM64) jobs to exercise different code paths in [simd/quad_vector.hpp](include/quadblas/simd/quad_vector.hpp).
