#include "include/quadblas/quadblas.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

static double to_double(Sleef_quad q) {
    return static_cast<double>(Sleef_cast_to_doubleq1(q));
}

static Sleef_quad from_double(double d) {
    return Sleef_cast_from_doubleq1(d);
}

// Compiler barrier to prevent dead-code elimination
template <typename T>
static void do_not_optimize(T const &value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

static void fill_random(Sleef_quad *arr, size_t n, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < n; ++i)
        arr[i] = from_double(dist(gen));
}

// Returns median time in milliseconds over `trials` runs (with 2 warmup runs)
template <typename Func>
static double benchmark_median(Func fn, int trials = 7) {
    // warmup
    for (int i = 0; i < 2; ++i) fn();

    std::vector<double> times(trials);
    for (int i = 0; i < trials; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    return times[trials / 2]; // median
}

// ---- Benchmarks ----

static void bench_dot() {
    std::cout << "\n=== DOT PRODUCT ===\n";
    std::cout << std::setw(10) << "N"
              << std::setw(15) << "Time (ms)" << "\n";
    std::cout << std::string(25, '-') << "\n";

    for (size_t n : {1000, 5000, 10000, 50000}) {
        Sleef_quad *x = QuadBLAS::aligned_alloc<Sleef_quad>(n);
        Sleef_quad *y = QuadBLAS::aligned_alloc<Sleef_quad>(n);
        fill_random(x, n, 1);
        fill_random(y, n, 2);

        double ms = benchmark_median([&]() {
            Sleef_quad r = QuadBLAS::dot(n, x, 1, y, 1);
            do_not_optimize(r);
        });

        std::cout << std::setw(10) << n
                  << std::setw(15) << std::fixed << std::setprecision(3) << ms << "\n";

        QuadBLAS::aligned_free(x);
        QuadBLAS::aligned_free(y);
    }
}

static void bench_axpy() {
    std::cout << "\n=== AXPY ===\n";
    std::cout << std::setw(10) << "N"
              << std::setw(15) << "Time (ms)" << "\n";
    std::cout << std::string(25, '-') << "\n";

    Sleef_quad alpha = from_double(2.5);

    for (size_t n : {1000, 5000, 10000, 50000}) {
        Sleef_quad *x = QuadBLAS::aligned_alloc<Sleef_quad>(n);
        Sleef_quad *y = QuadBLAS::aligned_alloc<Sleef_quad>(n);
        Sleef_quad *y_backup = QuadBLAS::aligned_alloc<Sleef_quad>(n);
        fill_random(x, n, 3);
        fill_random(y, n, 4);
        std::copy(y, y + n, y_backup);

        double ms = benchmark_median([&]() {
            std::copy(y_backup, y_backup + n, y); // restore y each run
            QuadBLAS::axpy(n, alpha, x, 1, y, 1);
            do_not_optimize(y[0]);
        });

        std::cout << std::setw(10) << n
                  << std::setw(15) << std::fixed << std::setprecision(3) << ms << "\n";

        QuadBLAS::aligned_free(x);
        QuadBLAS::aligned_free(y);
        QuadBLAS::aligned_free(y_backup);
    }
}

static void bench_gemv() {
    std::cout << "\n=== GEMV ===\n";
    std::cout << std::setw(10) << "N"
              << std::setw(15) << "Time (ms)" << "\n";
    std::cout << std::string(25, '-') << "\n";

    Sleef_quad alpha = from_double(1.0);
    Sleef_quad beta  = from_double(0.0);

    for (size_t n : {100, 300, 500, 1000}) {
        Sleef_quad *A = QuadBLAS::aligned_alloc<Sleef_quad>(n * n);
        Sleef_quad *x = QuadBLAS::aligned_alloc<Sleef_quad>(n);
        Sleef_quad *y = QuadBLAS::aligned_alloc<Sleef_quad>(n);
        fill_random(A, n * n, 5);
        fill_random(x, n, 6);

        double ms = benchmark_median([&]() {
            for (size_t i = 0; i < n; ++i)
                y[i] = from_double(0.0);
            QuadBLAS::gemv(QuadBLAS::Layout::RowMajor, n, n, alpha, A, n, x, 1, beta, y, 1);
            do_not_optimize(y[0]);
        });

        std::cout << std::setw(10) << n
                  << std::setw(15) << std::fixed << std::setprecision(3) << ms << "\n";

        QuadBLAS::aligned_free(A);
        QuadBLAS::aligned_free(x);
        QuadBLAS::aligned_free(y);
    }
}

static void bench_gemm() {
    std::cout << "\n=== GEMM ===\n";
    std::cout << std::setw(10) << "N"
              << std::setw(15) << "Time (ms)" << "\n";
    std::cout << std::string(25, '-') << "\n";

    Sleef_quad alpha = from_double(1.0);
    Sleef_quad beta  = from_double(0.0);

    for (size_t n : {32, 64, 128, 200, 300}) {
        Sleef_quad *A = QuadBLAS::aligned_alloc<Sleef_quad>(n * n);
        Sleef_quad *B = QuadBLAS::aligned_alloc<Sleef_quad>(n * n);
        Sleef_quad *C = QuadBLAS::aligned_alloc<Sleef_quad>(n * n);
        fill_random(A, n * n, 7);
        fill_random(B, n * n, 8);

        int trials = (n <= 128) ? 5 : 3;

        double ms = benchmark_median([&]() {
            for (size_t i = 0; i < n * n; ++i)
                C[i] = from_double(0.0);
            QuadBLAS::gemm(QuadBLAS::Layout::RowMajor, n, n, n, alpha, A, n, B, n, beta, C, n);
            do_not_optimize(C[0]);
        }, trials);

        std::cout << std::setw(10) << n
                  << std::setw(15) << std::fixed << std::setprecision(3) << ms << "\n";

        QuadBLAS::aligned_free(A);
        QuadBLAS::aligned_free(B);
        QuadBLAS::aligned_free(C);
    }
}

int main() {
    std::cout << "QuadBLAS Performance Comparison Benchmark\n";
    std::cout << "==========================================\n";
    std::cout << "Version: " << quadblas_get_version() << "\n";
    std::cout << "Threads: " << QuadBLAS::get_num_threads() << "\n";

    bench_dot();
    bench_axpy();
    bench_gemv();
    bench_gemm();

    std::cout << "\nDone.\n";
    return 0;
}
