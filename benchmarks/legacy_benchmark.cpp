#include "include/quadblas/quadblas.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>

// Helper functions
double to_double(Sleef_quad q) {
    return static_cast<double>(Sleef_cast_to_doubleq1(q));
}

Sleef_quad from_double(double d) {
    return Sleef_cast_from_doubleq1(d);
}

// High-precision timing with warmup and averaging
class BenchTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        return duration.count() / 1e6;
    }

    double stop_s() {
        return stop_ms() / 1000.0;
    }
};

// Compiler barrier to prevent dead code elimination of results
template <typename T>
inline void do_not_optimize(const T &value) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r,m"(value) : "memory");
#else
    volatile T sink = value;
    (void)sink;
#endif
}

// Run a callable multiple times and return median time
template <typename Func>
double benchmark_median(Func &&fn, int warmup = 2, int trials = 5) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        fn();
    }

    // Timed trials
    std::vector<double> times(trials);
    BenchTimer timer;
    for (int i = 0; i < trials; ++i) {
        timer.start();
        fn();
        times[i] = timer.stop_ms();
    }

    // Return median
    std::sort(times.begin(), times.end());
    return times[trials / 2];
}

// Random number generator for consistent tests
class QuadRandom {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist;

public:
    QuadRandom(unsigned seed = 12345) : gen(seed), dist(-1.0, 1.0) {}

    Sleef_quad next() {
        return from_double(dist(gen));
    }

    void fill_array(Sleef_quad* arr, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            arr[i] = next();
        }
    }
};

// ================================================================================
// NAIVE IMPLEMENTATIONS (No optimizations - baseline for comparison)
// ================================================================================

Sleef_quad naive_dot(size_t n, const Sleef_quad* x, const Sleef_quad* y) {
    Sleef_quad result = from_double(0.0);
    for (size_t i = 0; i < n; ++i) {
        result = Sleef_fmaq1_u05(x[i], y[i], result);
    }
    return result;
}

void naive_gemv(size_t m, size_t n, Sleef_quad alpha,
                const Sleef_quad* A, size_t lda,
                const Sleef_quad* x, Sleef_quad beta, Sleef_quad* y) {
    for (size_t i = 0; i < m; ++i) {
        Sleef_quad sum = from_double(0.0);
        for (size_t j = 0; j < n; ++j) {
            sum = Sleef_fmaq1_u05(A[i * lda + j], x[j], sum);
        }
        y[i] = Sleef_fmaq1_u05(alpha, sum, Sleef_mulq1_u05(beta, y[i]));
    }
}

void naive_gemm(size_t m, size_t n, size_t k, Sleef_quad alpha,
                const Sleef_quad* A, size_t lda,
                const Sleef_quad* B, size_t ldb,
                Sleef_quad beta, Sleef_quad* C, size_t ldc) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            Sleef_quad sum = from_double(0.0);
            for (size_t l = 0; l < k; ++l) {
                sum = Sleef_fmaq1_u05(A[i * lda + l], B[l * ldb + j], sum);
            }
            C[i * ldc + j] = Sleef_fmaq1_u05(alpha, sum, Sleef_mulq1_u05(beta, C[i * ldc + j]));
        }
    }
}

// ================================================================================
// BENCHMARK FUNCTIONS (with warmup and median timing)
// ================================================================================

void benchmark_dot_product() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "DOT PRODUCT BENCHMARK (Vector . Vector)\n";
    std::cout << std::string(80, '=') << "\n";

    std::vector<size_t> sizes = {1000, 5000, 10000, 50000, 100000};

    std::cout << std::setw(10) << "Size"
              << std::setw(15) << "Naive (ms)"
              << std::setw(15) << "QuadBLAS (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "Error" << "\n";
    std::cout << std::string(80, '-') << "\n";

    QuadRandom rng;

    for (size_t n : sizes) {
        Sleef_quad* x = QuadBLAS::aligned_alloc<Sleef_quad>(n);
        Sleef_quad* y = QuadBLAS::aligned_alloc<Sleef_quad>(n);
        rng.fill_array(x, n);
        rng.fill_array(y, n);

        Sleef_quad naive_result, quadblas_result;

        double naive_time = benchmark_median([&]() {
            naive_result = naive_dot(n, x, y);
            do_not_optimize(naive_result);
        });

        double quadblas_time = benchmark_median([&]() {
            quadblas_result = QuadBLAS::dot(n, x, 1, y, 1);
            do_not_optimize(quadblas_result);
        });

        double speedup = naive_time / quadblas_time;
        double error = std::abs(to_double(naive_result) - to_double(quadblas_result));

        std::cout << std::setw(10) << n
                  << std::setw(15) << std::fixed << std::setprecision(3) << naive_time
                  << std::setw(15) << std::fixed << std::setprecision(3) << quadblas_time
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup
                  << std::setw(15) << std::scientific << std::setprecision(2) << error
                  << "\n";

        QuadBLAS::aligned_free(x);
        QuadBLAS::aligned_free(y);
    }
}

void benchmark_matrix_vector() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "MATRIX-VECTOR BENCHMARK (Matrix x Vector)\n";
    std::cout << std::string(80, '=') << "\n";

    std::vector<size_t> sizes = {100, 300, 500, 1000, 1500};

    std::cout << std::setw(10) << "Size"
              << std::setw(15) << "Naive (ms)"
              << std::setw(15) << "QuadBLAS (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "GFLOPS" << "\n";
    std::cout << std::string(80, '-') << "\n";

    QuadRandom rng;

    for (size_t n : sizes) {
        size_t m = n;
        Sleef_quad* A = QuadBLAS::aligned_alloc<Sleef_quad>(m * n);
        Sleef_quad* x = QuadBLAS::aligned_alloc<Sleef_quad>(n);
        Sleef_quad* y = QuadBLAS::aligned_alloc<Sleef_quad>(m);

        rng.fill_array(A, m * n);
        rng.fill_array(x, n);

        Sleef_quad alpha = from_double(1.0);
        Sleef_quad beta = from_double(0.0);

        double naive_time = benchmark_median([&]() {
            for (size_t i = 0; i < m; ++i) y[i] = from_double(0.0);
            naive_gemv(m, n, alpha, A, n, x, beta, y);
        });

        double quadblas_time = benchmark_median([&]() {
            for (size_t i = 0; i < m; ++i) y[i] = from_double(0.0);
            QuadBLAS::gemv(QuadBLAS::Layout::RowMajor, m, n, alpha, A, n, x, 1, beta, y, 1);
        });

        double speedup = naive_time / quadblas_time;
        double gflops = (2.0 * m * n) / (quadblas_time * 1e6);

        std::cout << std::setw(10) << n
                  << std::setw(15) << std::fixed << std::setprecision(3) << naive_time
                  << std::setw(15) << std::fixed << std::setprecision(3) << quadblas_time
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup
                  << std::setw(15) << std::fixed << std::setprecision(3) << gflops
                  << "\n";

        QuadBLAS::aligned_free(A);
        QuadBLAS::aligned_free(x);
        QuadBLAS::aligned_free(y);
    }
}

void benchmark_matrix_matrix() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "MATRIX-MATRIX BENCHMARK (Matrix x Matrix)\n";
    std::cout << std::string(80, '=') << "\n";

    // Reduced to practical sizes for quad precision
    // 500x500 GEMM = 250M quad FMAs which already takes seconds
    std::vector<size_t> sizes = {32, 64, 128, 200, 500};

    std::cout << std::setw(10) << "Size"
              << std::setw(15) << "Naive (s)"
              << std::setw(15) << "QuadBLAS (s)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "GFLOPS" << "\n";
    std::cout << std::string(80, '-') << "\n";

    QuadRandom rng;

    for (size_t n : sizes) {
        size_t m = n, k = n;

        Sleef_quad* A = QuadBLAS::aligned_alloc<Sleef_quad>(m * k);
        Sleef_quad* B = QuadBLAS::aligned_alloc<Sleef_quad>(k * n);
        Sleef_quad* C = QuadBLAS::aligned_alloc<Sleef_quad>(m * n);

        rng.fill_array(A, m * k);
        rng.fill_array(B, k * n);

        Sleef_quad alpha = from_double(1.0);
        Sleef_quad beta = from_double(0.0);

        // Fewer trials for large sizes
        int trials = (n <= 128) ? 5 : 3;
        int warmup = (n <= 128) ? 2 : 1;

        double naive_time = benchmark_median([&]() {
            for (size_t i = 0; i < m * n; ++i) C[i] = from_double(0.0);
            naive_gemm(m, n, k, alpha, A, k, B, n, beta, C, n);
        }, warmup, trials) / 1000.0; // Convert to seconds

        double quadblas_time = benchmark_median([&]() {
            for (size_t i = 0; i < m * n; ++i) C[i] = from_double(0.0);
            QuadBLAS::gemm(QuadBLAS::Layout::RowMajor, m, n, k, alpha, A, k, B, n, beta, C, n);
        }, warmup, trials) / 1000.0;

        double speedup = naive_time / quadblas_time;
        double gflops = (2.0 * m * n * k) / (quadblas_time * 1e9);

        std::cout << std::setw(10) << n
                  << std::setw(15) << std::fixed << std::setprecision(3) << naive_time
                  << std::setw(15) << std::fixed << std::setprecision(3) << quadblas_time
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup
                  << std::setw(15) << std::fixed << std::setprecision(3) << gflops
                  << "\n";

        QuadBLAS::aligned_free(A);
        QuadBLAS::aligned_free(B);
        QuadBLAS::aligned_free(C);
    }
}

void benchmark_threading_scaling() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "THREADING SCALABILITY BENCHMARK\n";
    std::cout << std::string(80, '=') << "\n";

    const size_t n = 50000; // Large enough for threading to matter

    Sleef_quad* x = QuadBLAS::aligned_alloc<Sleef_quad>(n);
    Sleef_quad* y = QuadBLAS::aligned_alloc<Sleef_quad>(n);

    QuadRandom rng;
    rng.fill_array(x, n);
    rng.fill_array(y, n);

    std::cout << std::setw(12) << "Threads"
              << std::setw(15) << "Time (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "Efficiency" << "\n";
    std::cout << std::string(60, '-') << "\n";

    double baseline_time = 0.0;
    int max_threads = QuadBLAS::get_num_threads();

    for (int threads = 1; threads <= max_threads; threads *= 2) {
        QuadBLAS::set_num_threads(threads);

        Sleef_quad result;
        double avg_time = benchmark_median([&]() {
            result = QuadBLAS::dot(n, x, 1, y, 1);
            do_not_optimize(result);
        }, 3, 10);

        (void)result;

        if (threads == 1) baseline_time = avg_time;

        double speedup = baseline_time / avg_time;
        double efficiency = speedup / threads * 100.0;

        std::cout << std::setw(12) << threads
                  << std::setw(15) << std::fixed << std::setprecision(3) << avg_time
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup
                  << std::setw(14) << std::fixed << std::setprecision(1) << efficiency << "%"
                  << "\n";

        if (threads >= 16) break;
    }

    QuadBLAS::set_num_threads(max_threads);

    QuadBLAS::aligned_free(x);
    QuadBLAS::aligned_free(y);
}

void print_system_info() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "SYSTEM INFORMATION\n";
    std::cout << std::string(80, '=') << "\n";

    std::cout << "QuadBLAS Version: " << quadblas_get_version() << "\n";
    std::cout << "Available CPU threads: " << QuadBLAS::get_num_threads() << "\n";

#ifdef QUADBLAS_X86_64
    std::cout << "Platform: x86-64 with SIMD vectorization\n";
#elif defined(QUADBLAS_AARCH64)
    std::cout << "Platform: AArch64 with NEON vectorization\n";
#else
    std::cout << "Platform: Generic (scalar fallback)\n";
#endif

#ifdef _OPENMP
    std::cout << "OpenMP: Enabled\n";
#else
    std::cout << "OpenMP: Disabled\n";
#endif

    std::cout << "SIMD Vector Size: " << QuadBLAS::VECTOR_SIZE << " quad elements\n";
    std::cout << "Memory Alignment: " << QuadBLAS::ALIGNMENT << " bytes\n";
    std::cout << "Quad precision size: " << sizeof(Sleef_quad) << " bytes\n";
}

int main() {
    print_system_info();

    try {
        benchmark_dot_product();
        benchmark_matrix_vector();
        benchmark_matrix_matrix();
        benchmark_threading_scaling();

        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "All benchmarks completed.\n";
        std::cout << std::string(80, '=') << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Benchmark error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
