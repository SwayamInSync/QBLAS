# QBLAS

High-performance BLAS for IEEE 754 binary128 (quad-precision) floating point, built on top of [SLEEF](https://sleef.org). Compiled C library with runtime CPU dispatch (SSE2 / AVX2 / AVX-512 on x86, NEON on AArch64). Drop-in CBLAS-style C API.

This repository is the rewrite on branch `overhaul-rewrite`. The legacy header-only implementation on `main` is preserved for reference but is no longer maintained.

## What is included

Routines implemented and tested:

| Level   | Routine     | Description                                              |
| ------- | ----------- | -------------------------------------------------------- |
| 1       | `cblas_qdot`   | dot product                                           |
| 1       | `cblas_qaxpy`  | y = alpha*x + y                                       |
| 1       | `cblas_qscal`  | x = alpha*x                                           |
| 1       | `cblas_qnrm2`  | sqrt(sum(x_i^2))                                      |
| 1       | `cblas_qasum`  | sum(|x_i|)                                            |
| 1       | `cblas_iqamax` | argmax |x_i|                                          |
| 1       | `cblas_qcopy`  | y = x                                                 |
| 1       | `cblas_qswap`  | swap x and y                                          |
| 2       | `cblas_qgemv`  | y = alpha*op(A)*x + beta*y                            |
| 2       | `cblas_qger`   | A += alpha*x*y^T                                      |
| 2       | `cblas_qsymv`  | y = alpha*A*x + beta*y, A symmetric                   |
| 2       | `cblas_qtrsv`  | solve op(A)*x = b in place                            |
| 3       | `cblas_qgemm`  | C = alpha*op(A)*op(B) + beta*C                        |
| 3       | `cblas_qsyrk`  | C = alpha*op(A)*op(A)^T + beta*C                      |
| 3       | `cblas_qtrmm`  | B = alpha*op(A)*B or alpha*B*op(A)                    |
| 3       | `cblas_qtrsm`  | solve op(A)*X = alpha*B or X*op(A) = alpha*B          |

All routines accept both row-major and column-major matrices, all four `Trans` / `NoTrans` / `ConjTrans` combinations, `Upper` / `Lower`, `Unit` / `NonUnit`, and BLAS-standard positive or negative strides.

Library control:

```c
const char *qblas_get_version(void);
const char *qblas_get_dispatch_tier(void); /* "generic" | "sse2" | "avx2" | "avx512" | "neon" */
void  qblas_set_num_threads(int n);
int   qblas_get_num_threads(void);
int   qblas_get_max_threads(void);
```

Environment overrides:

- `QBLAS_DISPATCH=generic|sse2|avx2|avx512|neon` forces a specific SIMD tier. Useful to confirm cross-tier numerical equivalence.
- `OMP_NUM_THREADS=N` sets thread count (standard OpenMP).
- `OMP_PLACES=cores OMP_PROC_BIND=close` is recommended on many-core hosts to keep per-iteration timing stable.

## Dependencies

- C11 + C++17 compilers (gcc, clang, AppleClang).
- CMake 3.16 or newer.
- OpenMP (libgomp, libomp). Optional but strongly recommended.
- SLEEF 3.8+ with quad-precision support, built with all SIMD paths enabled. The `scripts/bootstrap_sleef.sh` script builds it the right way.
- Google Benchmark (vendored as a submodule; pulled in automatically by `cmake`).

## Installation

### 1. Clone with submodules

```bash
git clone --recursive https://github.com/SwayamInSync/QBLAS.git
cd QBLAS
```

If you cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### 2. Build SLEEF

This must be done once, before the first qblas build:

```bash
bash scripts/bootstrap_sleef.sh
```

The script builds SLEEF with every SIMD path turned on (`SSE2`, `AVX`, `AVX2`, `AVX512F` on x86; `AdvSIMD` on AArch64; inline headers enabled) and installs it into `.sleef-prefix/`. Without this, SLEEF only builds for the host CPU and binaries fail to link when run on a different CPU or to use a different ISA tier.

The script also copies the vendored `libtlfloat` into the prefix, which is a runtime dependency of `libsleefquad`.

### 3. Build qblas

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

This produces:

- `build/src/libqblas.so` (or `.dylib` on macOS) — the shared library.
- `build/tests/test_smoke`, `test_level1`, `test_level2`, `test_gemm`, `test_level3_extra` — correctness tests.
- `build/bench/qblas_bench` — Google Benchmark performance suite.
- `build/bench/qblas_bench_compare` — sized to match the legacy `main`-branch benchmark for direct before/after.

CMake options:

- `-DQBLAS_BUILD_SHARED=OFF` for a static library.
- `-DQBLAS_USE_OPENMP=OFF` to disable threading.
- `-DQBLAS_BUILD_TESTS=OFF` to skip test executables.
- `-DQBLAS_BUILD_BENCH=OFF` to skip benchmarks.
- `-DSLEEF_ROOT=/path/to/sleef-prefix` if your SLEEF lives somewhere other than `./.sleef-prefix/`.

### 4. Install (optional)

```bash
sudo cmake --install build
```

Installs `libqblas` to `${CMAKE_INSTALL_PREFIX}/lib` and headers to `${CMAKE_INSTALL_PREFIX}/include/qblas/`.

## Using qblas

Include the umbrella header:

```c
#include <qblas/qblas.h>
```

Compile and link:

```bash
gcc -O3 my_app.c -lqblas -lsleefquad -lsleef -o my_app
```

Minimal example:

```c
#include <qblas/qblas.h>
#include <stdio.h>

int main(void) {
    enum { N = 4 };
    Sleef_quad x[N], y[N];
    for (int i = 0; i < N; ++i) {
        x[i] = Sleef_cast_from_doubleq1((double)(i + 1));
        y[i] = Sleef_cast_from_doubleq1(2.0);
    }
    Sleef_quad d = cblas_qdot(N, x, 1, y, 1);
    printf("dot = %f\n", (double)Sleef_cast_to_doubleq1(d));  /* 20.0 */
    printf("tier = %s\n", qblas_get_dispatch_tier());
    return 0;
}
```

GEMM:

```c
#include <qblas/qblas.h>

void mul(const Sleef_quad *A, const Sleef_quad *B, Sleef_quad *C,
         int m, int n, int k) {
    Sleef_quad alpha = Sleef_cast_from_doubleq1(1.0);
    Sleef_quad beta  = Sleef_cast_from_doubleq1(0.0);
    cblas_qgemm(QblasRowMajor, QblasNoTrans, QblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
}
```

The full public API is in `include/qblas/qblas.h` (level-1/2/3 headers are included transitively).

## Correctness testing

Every routine is checked against a naive scalar reference written in plain SLEEF `Sleef_*q1` calls. Tests cover all transpose / layout / uplo / diag combinations, positive and negative strides, and sizes that exercise edge tiles (m=1, k=1, primes, sizes both above and below the GEMM block size).

Run the full suite:

```bash
ctest --test-dir build --output-on-failure
```

Run a single executable:

```bash
./build/tests/test_gemm
./build/tests/test_level3_extra
```

Tolerance is `1e-25` for vector operations (about 9 orders of magnitude tighter than double precision) and a backward-error bound for trsm.

Cross-tier validation:

```bash
QBLAS_DISPATCH=generic ctest --test-dir build
QBLAS_DISPATCH=sse2    ctest --test-dir build
QBLAS_DISPATCH=avx2    ctest --test-dir build
```

All four should pass identically. This confirms the runtime CPU dispatch produces the same result across SIMD widths. If you set `QBLAS_DISPATCH` to a tier the CPU does not support (e.g. `avx512` on a Zen 3 box), the library prints a stderr warning and falls back to the auto-detected tier instead of executing illegal instructions.

### numpy comparison test (`test_against_numpy`)

In addition to the C tests against a naive scalar reference, there is a Python test that calls every QBLAS routine via ctypes, rounds the quad output to float64, and compares against numpy's float64 implementation of the same operation. This catches any issue that survives both the naive C reference and the quad-precision pathway but would surface when a user does `Sleef_cast_to_doubleq1(...)`.

Requirements: Python 3 and `numpy >= 2.0`.

```bash
python3 -m venv .venv
.venv/bin/pip install 'numpy>=2.0'
PATH=$(pwd)/.venv/bin:$PATH ctest --test-dir build --output-on-failure -R test_against_numpy
```

The test exercises all Level 1 / 2 / 3 routines at a range of sizes with random `~U(-1, 1)` inputs. Tolerances are picked per routine to allow for the double-precision rounding step (~1e-13 for elementwise, ~1e-10 for matrix-vector, scales with k for matrix-matrix).

## Benchmarking

There are two benchmark binaries. Both use Google Benchmark.

### `qblas_bench` (full)

Sweeps powers of 2 / 4 from 1024 up to 4M elements for L1, 128 to 4096 for L2 and GEMV, 64 to 1024 for GEMM and SYRK. Useful for tuning.

```bash
./build/bench/qblas_bench --benchmark_min_time=0.3s --benchmark_repetitions=3
```

### `qblas_bench_compare` (size-matched to legacy)

Same sizes as the original `benchmarks/benchmark.cpp` on the legacy `main` branch, so before/after comparison is direct.

```bash
./build/bench/qblas_bench_compare --benchmark_min_time=0.3s --benchmark_repetitions=3 \
    --benchmark_report_aggregates_only=true
```

### Recommended environment

On many-core hosts, pin threads for stable timing:

```bash
OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close \
    ./build/bench/qblas_bench --benchmark_min_time=0.3s
```

Use `OMP_NUM_THREADS=1` to measure pure single-thread kernel quality.

### Performance numbers

On AMD EPYC 7V13 (96 cores, AVX2 tier, OMP_NUM_THREADS=16, median of 3):

| Routine | Size | Throughput (M items/sec) |
| ------- | ---- | -----------------------: |
| qdot    | 1M   | 429.6                    |
| qaxpy   | 64K  | 387.8                    |
| qnrm2   | 64K  | 409.8                    |
| qasum   | 64K  | 853.7                    |
| qscal   | 64K  | 888.2                    |
| qgemv   | 1600 x 1600 | 431.2             |
| qgemm   | 512 x 512 x 512 | 429.7         |
| qsyrk   | 512 x 512 x 512 | 443.3         |
| qtrmm   | 512 x 512  | 350.3                |
| qtrsm   | 512 x 512  | 335.7                |

For full before/after tables (including single-thread and 96-thread numbers), see [PERF_COMPARISON.md](PERF_COMPARISON.md).

For remaining performance opportunities not yet taken, see [PERF_HEADROOM.md](PERF_HEADROOM.md).

## Repository layout

```
include/qblas/          Public C headers.
src/                    Library sources.
  cpu/qblas_cpu.c       Runtime CPU and cache detection, OMP overhead timing,
                        dispatch table population, threading helpers.
  common/               Internal helpers shared by all kernels.
  kernels/              Per-ISA kernel translation units.
    generic/            Scalar fallback (always built).
    sse2/               x86 SSE2 (q2 width).
    avx2/               x86 AVX2 (q4 width).
    avx512/             x86 AVX-512F (q8 width).
    neon/               AArch64 AdvSIMD (q2 width).
    kernels_template.h  Shared template instantiated per ISA.
  level1/               L1 entry points (dispatch + threading).
  level2/               L2 entry points.
  level3/               qgemm, qsyrk, qtrmm, qtrsm.
tests/                  ctest binaries.
  python/               ctypes wrapper + numpy-vs-qblas test.
bench/                  Google Benchmark binaries.
cmake/                  CMake helpers (architecture detection).
scripts/                bootstrap_sleef.sh.
third_party/            sleef, benchmark, OpenBLAS (for reference) submodules.
```

## Continuous integration

The `.github/workflows/ci.yml` workflow runs on every push and pull request:

- Ubuntu (gcc and clang) and macOS (clang).
- Builds SLEEF via `scripts/bootstrap_sleef.sh`.
- Builds qblas and runs `ctest`.
- Smoke-runs the benchmark.
- On Linux only, re-runs the test suite under each forced dispatch tier (generic / sse2 / avx2) to verify they all give the same results.
- A separate `bench-baseline` job runs the matched-size benchmark suite and uploads results as an artifact for tracking perf over time.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{qblas2025,
  title  = {QBLAS: High-Performance BLAS for IEEE-754 Quadruple Precision},
  author = {Swayam Singh},
  year   = {2025},
  url    = {https://github.com/SwayamInSync/QBLAS},
  note   = {BLAS Levels 1, 2, and 3 for binary128 floating point, built on SLEEF}
}
```

## Acknowledgements

Built on [SLEEF](https://sleef.org) (Naoki Shibata and contributors).
