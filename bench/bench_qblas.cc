/* Minimal bench to verify wiring; full suite added in a later commit. */
#include <benchmark/benchmark.h>
#include <qblas/qblas.h>
#include <random>
#include <vector>

static Sleef_quad qd(double x) { return Sleef_cast_from_doubleq1(x); }

static void fill(std::vector<Sleef_quad> &v, unsigned seed = 42) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (auto &q : v) q = qd(d(g));
}

static void BM_qdot(benchmark::State &state) {
    size_t n = static_cast<size_t>(state.range(0));
    std::vector<Sleef_quad> x(n), y(n);
    fill(x, 1); fill(y, 2);
    Sleef_quad s = qd(0.0);
    for (auto _ : state) {
        s = cblas_qdot(static_cast<int>(n), x.data(), 1, y.data(), 1);
        benchmark::DoNotOptimize(s);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_qdot)->RangeMultiplier(8)->Range(1024, 1 << 20);

static void BM_qaxpy(benchmark::State &state) {
    size_t n = static_cast<size_t>(state.range(0));
    std::vector<Sleef_quad> x(n), y(n);
    fill(x, 1); fill(y, 2);
    Sleef_quad a = qd(1.5);
    for (auto _ : state) {
        cblas_qaxpy(static_cast<int>(n), a, x.data(), 1, y.data(), 1);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_qaxpy)->RangeMultiplier(8)->Range(1024, 1 << 20);

BENCHMARK_MAIN();
