/* Google Benchmark suite.  `items_per_second` = quad-FMA throughput
 * (items = n for dot/axpy, m*n for gemv, m*n*k for gemm/syrk). */

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

void BM_qdot(benchmark::State &state) {
    size_t n = static_cast<size_t>(state.range(0));
    std::vector<Sleef_quad> x(n), y(n);
    fill_random(x, 1); fill_random(y, 2);
    Sleef_quad s = qd(0.0);
    for (auto _ : state) {
        s = cblas_qdot(static_cast<int>(n), x.data(), 1, y.data(), 1);
        benchmark::DoNotOptimize(s);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
    state.SetLabel("FMA/iter=" + std::to_string(n));
}
BENCHMARK(BM_qdot)->RangeMultiplier(4)->Range(1<<10, 1<<22)->UseRealTime();

void BM_qaxpy(benchmark::State &state) {
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
BENCHMARK(BM_qaxpy)->RangeMultiplier(4)->Range(1<<10, 1<<22)->UseRealTime();

void BM_qscal(benchmark::State &state) {
    size_t n = static_cast<size_t>(state.range(0));
    std::vector<Sleef_quad> x(n);
    fill_random(x, 1);
    Sleef_quad a = qd(0.99);
    for (auto _ : state) {
        cblas_qscal(static_cast<int>(n), a, x.data(), 1);
        benchmark::DoNotOptimize(x.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_qscal)->RangeMultiplier(4)->Range(1<<10, 1<<22)->UseRealTime();

void BM_qnrm2(benchmark::State &state) {
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
BENCHMARK(BM_qnrm2)->RangeMultiplier(4)->Range(1<<10, 1<<22)->UseRealTime();

void BM_qasum(benchmark::State &state) {
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
BENCHMARK(BM_qasum)->RangeMultiplier(4)->Range(1<<10, 1<<22)->UseRealTime();

void BM_qgemv(benchmark::State &state) {
    size_t m = static_cast<size_t>(state.range(0));
    size_t n = m;
    std::vector<Sleef_quad> A(m * n), x(n), y(m);
    fill_random(A, 1); fill_random(x, 2); fill_random(y, 3);
    Sleef_quad alpha = qd(1.0), beta = qd(0.5);
    for (auto _ : state) {
        cblas_qgemv(QblasRowMajor, QblasNoTrans, m, n, alpha,
                    A.data(), n, x.data(), 1, beta, y.data(), 1);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(m * n));
}
BENCHMARK(BM_qgemv)->RangeMultiplier(2)->Range(128, 4096)->UseRealTime();

void BM_qgemv_t(benchmark::State &state) {
    size_t m = static_cast<size_t>(state.range(0));
    size_t n = m;
    std::vector<Sleef_quad> A(m * n), x(m), y(n);
    fill_random(A, 1); fill_random(x, 2); fill_random(y, 3);
    Sleef_quad alpha = qd(1.0), beta = qd(0.5);
    for (auto _ : state) {
        cblas_qgemv(QblasRowMajor, QblasTrans, m, n, alpha,
                    A.data(), n, x.data(), 1, beta, y.data(), 1);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(m * n));
}
BENCHMARK(BM_qgemv_t)->RangeMultiplier(2)->Range(128, 4096)->UseRealTime();

void BM_qgemm(benchmark::State &state) {
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
BENCHMARK(BM_qgemm)->RangeMultiplier(2)->Range(64, 1024)->UseRealTime();

void BM_qgemm_TN(benchmark::State &state) {
    size_t n = static_cast<size_t>(state.range(0));
    std::vector<Sleef_quad> A(n*n), B(n*n), C(n*n);
    fill_random(A, 1); fill_random(B, 2); fill_random(C, 3);
    Sleef_quad alpha = qd(1.0), beta = qd(0.0);
    for (auto _ : state) {
        cblas_qgemm(QblasRowMajor, QblasTrans, QblasNoTrans,
                    n, n, n, alpha, A.data(), n, B.data(), n,
                    beta, C.data(), n);
        benchmark::DoNotOptimize(C.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n * n * n));
}
BENCHMARK(BM_qgemm_TN)->RangeMultiplier(2)->Range(64, 1024)->UseRealTime();

void BM_qsyrk(benchmark::State &state) {
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
BENCHMARK(BM_qsyrk)->RangeMultiplier(2)->Range(64, 1024)->UseRealTime();

} /* namespace */

BENCHMARK_MAIN();
