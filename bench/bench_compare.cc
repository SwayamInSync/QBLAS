/* Apples-to-apples comparison benchmark — sizes match the legacy
 * benchmarks/benchmark.cpp (preserved on `main`) so the two outputs can be
 * lined up by routine and size. */

#include <benchmark/benchmark.h>
#include <qblas/qblas.h>
#include <random>
#include <vector>

namespace {
Sleef_quad qd(double x) { return Sleef_cast_from_doubleq1(x); }
template <typename T>
void fill_random(std::vector<T> &v, unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (auto &q : v) q = qd(d(g));
}
} /* anon */

/* ---- DotProduct sizes: 1000, 10000, 100000, 1000000 ---- */
static void BM_DotProduct(benchmark::State &state) {
    size_t n = static_cast<size_t>(state.range(0));
    std::vector<Sleef_quad> x(n), y(n);
    fill_random(x, 1); fill_random(y, 2);
    Sleef_quad s = qd(0.0);
    for (auto _ : state) {
        s = cblas_qdot(static_cast<int>(n), x.data(), 1, y.data(), 1);
        benchmark::DoNotOptimize(s);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_DotProduct)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

/* ---- AXPY sizes: 1000, 4000, 16000, 64000 ---- */
static void BM_AXPY(benchmark::State &state) {
    size_t n = static_cast<size_t>(state.range(0));
    std::vector<Sleef_quad> x(n), y(n);
    fill_random(x, 1); fill_random(y, 2);
    Sleef_quad a = qd(1.25);
    for (auto _ : state) {
        cblas_qaxpy(static_cast<int>(n), a, x.data(), 1, y.data(), 1);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_AXPY)->Arg(1000)->Arg(4000)->Arg(16000)->Arg(64000);

/* ---- VectorNorm (nrm2) sizes: 1000, 4000, 16000, 64000 ---- */
static void BM_VectorNorm(benchmark::State &state) {
    size_t n = static_cast<size_t>(state.range(0));
    std::vector<Sleef_quad> x(n);
    fill_random(x, 1);
    Sleef_quad s = qd(0.0);
    for (auto _ : state) {
        s = cblas_qnrm2(static_cast<int>(n), x.data(), 1);
        benchmark::DoNotOptimize(s);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_VectorNorm)->Arg(1000)->Arg(4000)->Arg(16000)->Arg(64000);

/* ---- GEMV sizes: 100, 200, 400, 800, 1600 (square) ---- */
static void BM_GEMV(benchmark::State &state) {
    size_t m = static_cast<size_t>(state.range(0)), n = m;
    std::vector<Sleef_quad> A(m*n), x(n), y(m);
    fill_random(A, 1); fill_random(x, 2); fill_random(y, 3);
    Sleef_quad alpha = qd(1.0), beta = qd(0.0);
    for (auto _ : state) {
        cblas_qgemv(QblasRowMajor, QblasNoTrans, m, n, alpha,
                    A.data(), n, x.data(), 1, beta, y.data(), 1);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(m * n));
}
BENCHMARK(BM_GEMV)->Arg(100)->Arg(200)->Arg(400)->Arg(800)->Arg(1600);

/* ---- GEMM sizes: 64, 128, 256, 512 (cubic) ---- */
static void BM_GEMM(benchmark::State &state) {
    size_t n = static_cast<size_t>(state.range(0));
    std::vector<Sleef_quad> A(n*n), B(n*n), C(n*n);
    fill_random(A, 1); fill_random(B, 2); fill_random(C, 3);
    Sleef_quad alpha = qd(1.0), beta = qd(0.0);
    for (auto _ : state) {
        cblas_qgemm(QblasRowMajor, QblasNoTrans, QblasNoTrans,
                    n, n, n, alpha, A.data(), n, B.data(), n,
                    beta, C.data(), n);
        benchmark::DoNotOptimize(C.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n * n * n));
}
BENCHMARK(BM_GEMM)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

BENCHMARK_MAIN();
