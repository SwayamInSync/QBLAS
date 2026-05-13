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

/* New-only L1: asum, scal */
static void BM_ASUM(benchmark::State &state) {
    size_t n = static_cast<size_t>(state.range(0));
    std::vector<Sleef_quad> x(n);
    fill_random(x, 1);
    Sleef_quad s = qd(0.0);
    for (auto _ : state) {
        s = cblas_qasum(static_cast<int>(n), x.data(), 1);
        benchmark::DoNotOptimize(s);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_ASUM)->Arg(1000)->Arg(4000)->Arg(16000)->Arg(64000);

static void BM_SCAL(benchmark::State &state) {
    size_t n = static_cast<size_t>(state.range(0));
    std::vector<Sleef_quad> x(n);
    fill_random(x, 1);
    Sleef_quad a = qd(0.9999);
    for (auto _ : state) {
        cblas_qscal(static_cast<int>(n), a, x.data(), 1);
        benchmark::DoNotOptimize(x.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_SCAL)->Arg(1000)->Arg(4000)->Arg(16000)->Arg(64000);

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

/* ---- SYRK sizes: 64, 128, 256, 512 ---- */
static void BM_SYRK(benchmark::State &state) {
    size_t n = static_cast<size_t>(state.range(0));
    size_t k = n;
    std::vector<Sleef_quad> A(n*k), C(n*n);
    fill_random(A, 1); fill_random(C, 3);
    Sleef_quad alpha = qd(1.0), beta = qd(0.0);
    for (auto _ : state) {
        cblas_qsyrk(QblasRowMajor, QblasUpper, QblasNoTrans, n, k,
                    alpha, A.data(), k, beta, C.data(), n);
        benchmark::DoNotOptimize(C.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n * n * k));
}
BENCHMARK(BM_SYRK)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

/* ---- TRMM sizes: B is m x n square ---- */
static void BM_TRMM(benchmark::State &state) {
    size_t m = static_cast<size_t>(state.range(0)), n = m;
    std::vector<Sleef_quad> A(m*m), B(m*n);
    fill_random(A, 1); fill_random(B, 2);
    Sleef_quad alpha = qd(1.0);
    for (auto _ : state) {
        cblas_qtrmm(QblasRowMajor, QblasLeft, QblasLower, QblasNoTrans, QblasNonUnit,
                    m, n, alpha, A.data(), m, B.data(), n);
        benchmark::DoNotOptimize(B.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(m * m * n / 2));
}
BENCHMARK(BM_TRMM)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

/* ---- TRSM sizes ---- */
static void BM_TRSM(benchmark::State &state) {
    size_t m = static_cast<size_t>(state.range(0)), n = m;
    std::vector<Sleef_quad> A(m*m), B(m*n), B0(m*n);
    fill_random(A, 1); fill_random(B0, 2);
    for (size_t i = 0; i < m; ++i) A[i*m+i] = qd(2.0 + (double)i*0.001);
    Sleef_quad alpha = qd(1.0);
    for (auto _ : state) {
        std::copy(B0.begin(), B0.end(), B.begin());
        cblas_qtrsm(QblasRowMajor, QblasLeft, QblasLower, QblasNoTrans, QblasNonUnit,
                    m, n, alpha, A.data(), m, B.data(), n);
        benchmark::DoNotOptimize(B.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(m * m * n / 2));
}
BENCHMARK(BM_TRSM)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

BENCHMARK_MAIN();
