#include "include/quadblas/quadblas.hpp"
#include <benchmark/benchmark.h>
#include <random>
#include <memory>
#include <iostream>
#include <algorithm>

class QuadRandom
{
private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist;

public:
    QuadRandom(unsigned seed = 42) : gen(seed), dist(-1.0, 1.0) {}

    Sleef_quad next()
    {
        return Sleef_cast_from_doubleq1(dist(gen));
    }

    void fill_array(Sleef_quad *arr, size_t n)
    {
        for (size_t i = 0; i < n; ++i)
        {
            arr[i] = next();
        }
    }
};

template <typename T>
class AlignedArray
{
private:
    T *data_;
    size_t size_;

public:
    AlignedArray(size_t size) : size_(size)
    {
        data_ = QuadBLAS::aligned_alloc<T>(size);
        if (!data_)
        {
            throw std::bad_alloc();
        }
    }

    ~AlignedArray()
    {
        QuadBLAS::aligned_free(data_);
    }

    T *data() { return data_; }
    const T *data() const { return data_; }
    size_t size() const { return size_; }

    T &operator[](size_t i) { return data_[i]; }
    const T &operator[](size_t i) const { return data_[i]; }
};

// Level 1 BLAS Benchmarks

static void BM_DotProduct(benchmark::State &state)
{
    const size_t n = state.range(0);

    AlignedArray<Sleef_quad> x(n), y(n);
    QuadRandom rng;
    rng.fill_array(x.data(), n);
    rng.fill_array(y.data(), n);

    Sleef_quad result;

    for (auto _ : state)
    {
        result = QuadBLAS::dot(n, x.data(), 1, y.data(), 1);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * 2 * sizeof(Sleef_quad));
    state.counters["GFLOPS"] = benchmark::Counter(
        2.0 * n * state.iterations(),
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000);
}

static void BM_VectorNorm(benchmark::State &state)
{
    const size_t n = state.range(0);

    QuadBLAS::DefaultVector<> x(n);
    QuadRandom rng;
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = rng.next();
    }

    Sleef_quad result;

    for (auto _ : state)
    {
        result = x.norm();
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.counters["GFLOPS"] = benchmark::Counter(
        2.0 * n * state.iterations(),
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000);
}

// y = alpha * x + y
static void BM_AXPY(benchmark::State &state)
{
    const size_t n = state.range(0);

    QuadBLAS::DefaultVector<> x(n), y(n);
    QuadRandom rng;
    Sleef_quad alpha = rng.next();

    for (size_t i = 0; i < n; ++i)
    {
        x[i] = rng.next();
        y[i] = rng.next();
    }

    for (auto _ : state)
    {
        y.axpy(alpha, x);
        benchmark::DoNotOptimize(y.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.counters["GFLOPS"] = benchmark::Counter(
        2.0 * n * state.iterations(),
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000);
}

// Level 2 BLAS Benchmarks

static void BM_GEMV(benchmark::State &state)
{
    const size_t n = state.range(0);
    const size_t m = n;

    AlignedArray<Sleef_quad> A(m * n), x(n), y(m);
    QuadRandom rng;

    rng.fill_array(A.data(), m * n);
    rng.fill_array(x.data(), n);
    rng.fill_array(y.data(), m);

    Sleef_quad alpha = Sleef_cast_from_doubleq1(1.0);
    Sleef_quad beta = Sleef_cast_from_doubleq1(0.0);

    for (auto _ : state)
    {
        QuadBLAS::gemv(QuadBLAS::Layout::RowMajor, m, n, alpha,
                       A.data(), n, x.data(), 1, beta, y.data(), 1);
        benchmark::DoNotOptimize(y.data());
    }

    state.SetItemsProcessed(state.iterations() * m * n);
    state.counters["GFLOPS"] = benchmark::Counter(
        2.0 * m * n * state.iterations(),
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000);
    state.counters["MatrixSize"] = n;
}

// Level 3 BLAS Benchmarks

static void BM_GEMM(benchmark::State &state)
{
    const size_t n = state.range(0);
    const size_t m = n, k = n;

    AlignedArray<Sleef_quad> A(m * k), B(k * n), C(m * n);
    QuadRandom rng;

    rng.fill_array(A.data(), m * k);
    rng.fill_array(B.data(), k * n);
    rng.fill_array(C.data(), m * n);

    Sleef_quad alpha = Sleef_cast_from_doubleq1(1.0);
    Sleef_quad beta = Sleef_cast_from_doubleq1(0.0);

    for (auto _ : state)
    {
        QuadBLAS::gemm(QuadBLAS::Layout::RowMajor, m, n, k, alpha,
                       A.data(), k, B.data(), n, beta, C.data(), n);
        benchmark::DoNotOptimize(C.data());
    }

    state.SetItemsProcessed(state.iterations() * m * n * k);
    state.counters["GFLOPS"] = benchmark::Counter(
        2.0 * m * n * k * state.iterations(),
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000);
    state.counters["MatrixSize"] = n;
}

// Threading Scalability Benchmarks

static void BM_ThreadingScalability(benchmark::State &state)
{
    const int num_threads = state.range(0);
    const size_t n = 10000;

    QuadBLAS::set_num_threads(num_threads);

    AlignedArray<Sleef_quad> x(n), y(n);
    QuadRandom rng;
    rng.fill_array(x.data(), n);
    rng.fill_array(y.data(), n);

    Sleef_quad result;

    for (auto _ : state)
    {
        result = QuadBLAS::dot(n, x.data(), 1, y.data(), 1);
        benchmark::DoNotOptimize(result);
    }

    state.counters["Threads"] = num_threads;
    state.counters["GFLOPS"] = benchmark::Counter(
        2.0 * n * state.iterations(),
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000);

    static double baseline_time = 0.0;
    if (num_threads == 1)
    {
        baseline_time = state.counters["GFLOPS"];
    }
    else if (baseline_time > 0.0)
    {
        double current_gflops = state.counters["GFLOPS"];
        double speedup = current_gflops / baseline_time;
        double efficiency = speedup / num_threads * 100.0;
        state.counters["Efficiency%"] = efficiency;
    }
}

// Numerical Precision Test
static void BM_NumericalPrecision(benchmark::State &state)
{
    QuadBLAS::DefaultVector<> x(3), y(3);

    x[0] = Sleef_cast_from_doubleq1(1e20);
    x[1] = Sleef_cast_from_doubleq1(1.0);
    x[2] = Sleef_cast_from_doubleq1(-1e20);

    y[0] = y[1] = y[2] = Sleef_cast_from_doubleq1(1.0);

    Sleef_quad result;

    for (auto _ : state)
    {
        result = x.dot(y);
        benchmark::DoNotOptimize(result);
    }

    double final_result = Sleef_cast_to_doubleq1(result);
    state.counters["Result"] = final_result;
    state.counters["Error"] = std::abs(final_result - 1.0);
}

// Memory Layout Benchmarks

static void BM_RowMajorGEMV(benchmark::State &state)
{
    const size_t n = state.range(0);
    QuadBLAS::MatrixRowMajor A(n, n);
    QuadBLAS::VectorRowMajor x(n), y(n);

    QuadRandom rng;
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = rng.next();
        for (size_t j = 0; j < n; ++j)
        {
            A(i, j) = rng.next();
        }
    }

    for (auto _ : state)
    {
        A.gemv(Sleef_cast_from_doubleq1(1.0), x, Sleef_cast_from_doubleq1(0.0), y);
        benchmark::DoNotOptimize(y.data());
    }

    state.counters["Layout"] = 0;
    state.counters["GFLOPS"] = benchmark::Counter(
        2.0 * n * n * state.iterations(),
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000);
}

static void BM_ColMajorGEMV(benchmark::State &state)
{
    const size_t n = state.range(0);
    QuadBLAS::MatrixColMajor A(n, n);
    QuadBLAS::VectorColMajor x(n), y(n);

    QuadRandom rng;
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = rng.next();
        for (size_t j = 0; j < n; ++j)
        {
            A(i, j) = rng.next();
        }
    }

    for (auto _ : state)
    {
        A.gemv(Sleef_cast_from_doubleq1(1.0), x, Sleef_cast_from_doubleq1(0.0), y);
        benchmark::DoNotOptimize(y.data());
    }

    state.counters["Layout"] = 1;
    state.counters["GFLOPS"] = benchmark::Counter(
        2.0 * n * n * state.iterations(),
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000);
}

// ===============================================================================
// REGISTER BENCHMARKS WITH PARAMETER RANGES
// ===============================================================================

// Level 1 BLAS - Vector sizes from 1K to 1M
BENCHMARK(BM_DotProduct)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_VectorNorm)
    ->RangeMultiplier(4)
    ->Range(1000, 64000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_AXPY)
    ->RangeMultiplier(4)
    ->Range(1000, 64000)
    ->Unit(benchmark::kMicrosecond);

// Level 2 BLAS - Matrix sizes from 100x100 to 2000x2000
BENCHMARK(BM_GEMV)
    ->RangeMultiplier(2)
    ->Range(100, 1600)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_RowMajorGEMV)
    ->RangeMultiplier(2)
    ->Range(100, 800)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ColMajorGEMV)
    ->RangeMultiplier(2)
    ->Range(100, 800)
    ->Unit(benchmark::kMillisecond);

// Level 3 BLAS - Matrix sizes from 64x64 to 1024x1024
BENCHMARK(BM_GEMM)
    ->RangeMultiplier(2)
    ->Range(64, 1024)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(2.0); // Ensure sufficient measurement time

// Threading scalability - Test 1, 2, 4, 8, 16 threads
BENCHMARK(BM_ThreadingScalability)
    ->DenseRange(1, std::min(16, QuadBLAS::get_num_threads()), 1)
    ->Unit(benchmark::kMicrosecond);

// Numerical precision test
BENCHMARK(BM_NumericalPrecision)
    ->Iterations(1000)
    ->Unit(benchmark::kNanosecond);

// Custom main function with QuadBLAS info
int main(int argc, char **argv)
{
    // Print system information
    std::cout << "=== QuadBLAS Google Benchmark Suite ===" << std::endl;
    std::cout << "Library Version: " << QuadBLAS::VERSION << std::endl;
    std::cout << "Available Threads: " << QuadBLAS::get_num_threads() << std::endl;

#ifdef QUADBLAS_X86_64
    std::cout << "Platform: x86-64 with SIMD vectorization" << std::endl;
#elif defined(QUADBLAS_AARCH64)
    std::cout << "Platform: AArch64 with NEON vectorization" << std::endl;
#else
    std::cout << "Platform: Generic (scalar fallback)" << std::endl;
#endif

#ifdef _OPENMP
    std::cout << "OpenMP: Enabled" << std::endl;
#else
    std::cout << "OpenMP: Disabled" << std::endl;
#endif

    std::cout << "SIMD Vector Size: " << QuadBLAS::VECTOR_SIZE << " elements" << std::endl;
    std::cout << "Memory Alignment: " << QuadBLAS::ALIGNMENT << " bytes" << std::endl;

    // Add macOS-specific system info
#ifdef __APPLE__
    std::cout << "Note: macOS warnings about CPU frequency/thread affinity are normal and don't affect timing accuracy" << std::endl;
#endif

    std::cout << "=========================================" << std::endl;

    // Initialize and run benchmarks
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    return 0;
}