#include "include/quadblas/quadblas.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>

// Helper function to convert Sleef_quad to double for printing
double to_double(Sleef_quad q) {
    return static_cast<double>(Sleef_cast_to_doubleq1(q));
}

// Helper function to create Sleef_quad from double
Sleef_quad from_double(double d) {
    return Sleef_cast_from_doubleq1(d);
}

// Timing utility
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // milliseconds
    }
};

// Test vector-vector dot product
void test_dot_product() {
    std::cout << "\n=== Testing Vector Dot Product ===\n";
    
    const size_t n = 10000;
    QuadBLAS::DefaultVector<> x(n), y(n);
    
    // Initialize with simple values
    for (size_t i = 0; i < n; ++i) {
        x[i] = from_double(i + 1);
        y[i] = from_double(2.0);
    }
    
    Timer timer;
    timer.start();
    Sleef_quad result = x.dot(y);
    double time_ms = timer.stop();
    
    // Expected result: sum of 2*(i+1) for i=0 to n-1 = 2 * n*(n+1)/2 = n*(n+1)
    double expected = static_cast<double>(n * (n + 1));
    double actual = to_double(result);
    
    std::cout << "Vector size: " << n << std::endl;
    std::cout << "Expected result: " << std::fixed << std::setprecision(1) << expected << std::endl;
    std::cout << "Actual result:   " << std::fixed << std::setprecision(1) << actual << std::endl;
    std::cout << "Error:           " << std::scientific << std::abs(expected - actual) << std::endl;
    std::cout << "Time:            " << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
    
    // Test C interface
    double c_result = quadblas_qdot(static_cast<int>(n), x.data(), 1, y.data(), 1);
    std::cout << "C interface result: " << std::fixed << std::setprecision(1) << c_result << std::endl;
}

// Test matrix-vector multiplication
void test_matrix_vector() {
    std::cout << "\n=== Testing Matrix-Vector Multiplication ===\n";
    
    const size_t m = 1000, n = 1000;
    QuadBLAS::DefaultMatrix<> A(m, n);
    QuadBLAS::DefaultVector<> x(n), y(m);
    
    // Initialize matrix A as identity matrix + small perturbation
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                A(i, j) = from_double(2.0);
            } else {
                A(i, j) = from_double(0.1);
            }
        }
    }
    
    // Initialize vector x
    for (size_t i = 0; i < n; ++i) {
        x[i] = from_double(1.0);
    }
    
    Timer timer;
    timer.start();
    A.gemv(from_double(1.0), x, from_double(0.0), y);
    double time_ms = timer.stop();
    
    std::cout << "Matrix size: " << m << "x" << n << std::endl;
    std::cout << "First few y values: ";
    for (size_t i = 0; i < std::min(size_t(5), m); ++i) {
        std::cout << std::fixed << std::setprecision(1) << to_double(y[i]) << " ";
    }
    std::cout << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
    
    // Test C interface
    QuadBLAS::DefaultVector<> y2(m);
    quadblas_qgemv('R', 'N', static_cast<int>(m), static_cast<int>(n), 1.0,
                   A.data(), static_cast<int>(A.leading_dimension()),
                   x.data(), 1, 0.0, y2.data(), 1);
    
    std::cout << "C interface - First few y values: ";
    for (size_t i = 0; i < std::min(size_t(5), m); ++i) {
        std::cout << std::fixed << std::setprecision(1) << to_double(y2[i]) << " ";
    }
    std::cout << std::endl;
}

// Test matrix-matrix multiplication
void test_matrix_matrix() {
    std::cout << "\n=== Testing Matrix-Matrix Multiplication ===\n";
    
    const size_t m = 500, n = 500, k = 500;
    QuadBLAS::DefaultMatrix<> A(m, k), B(k, n), C(m, n);
    
    // Initialize matrices
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            A(i, j) = from_double(dist(gen));
        }
    }
    
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < n; ++j) {
            B(i, j) = from_double(dist(gen));
        }
    }
    
    Timer timer;
    timer.start();
    A.gemm(from_double(1.0), B, from_double(0.0), C);
    double time_ms = timer.stop();
    
    double gflops = (2.0 * m * n * k) / (time_ms * 1e6);
    
    std::cout << "Matrix sizes: " << m << "x" << k << " * " << k << "x" << n 
              << " = " << m << "x" << n << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
    std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    
    // Check a few elements
    std::cout << "Sample C values: ";
    for (size_t i = 0; i < std::min(size_t(3), m); ++i) {
        std::cout << std::scientific << std::setprecision(3) << to_double(C(i, 0)) << " ";
    }
    std::cout << std::endl;
    
    // Test C interface
    QuadBLAS::DefaultMatrix<> C2(m, n);
    timer.start();
    quadblas_qgemm('R', 'N', 'N', static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                   1.0, A.data(), static_cast<int>(A.leading_dimension()),
                   B.data(), static_cast<int>(B.leading_dimension()),
                   0.0, C2.data(), static_cast<int>(C2.leading_dimension()));
    double time_ms_c = timer.stop();
    
    std::cout << "C interface time: " << std::fixed << std::setprecision(3) << time_ms_c << " ms" << std::endl;
}

// Performance comparison with different thread counts
void performance_comparison() {
    std::cout << "\n=== Performance Scaling Test ===\n";
    
    const size_t n = 5000;
    QuadBLAS::DefaultVector<> x(n), y(n);
    
    // Initialize vectors
    for (size_t i = 0; i < n; ++i) {
        x[i] = from_double(static_cast<double>(i + 1));
        y[i] = from_double(2.0);
    }
    
    std::cout << "Thread Count | Time (ms) | Speedup\n";
    std::cout << "-------------|-----------|--------\n";
    
    double baseline_time = 0.0;
    
    for (int threads = 1; threads <= QuadBLAS::get_num_threads(); threads *= 2) {
        QuadBLAS::set_num_threads(threads);
        
        Timer timer;
        const int iterations = 100;
        
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            volatile Sleef_quad result = x.dot(y);
            (void)result; // Suppress unused variable warning
        }
        double time_ms = timer.stop() / iterations;
        
        if (threads == 1) baseline_time = time_ms;
        double speedup = baseline_time / time_ms;
        
        std::cout << std::setw(12) << threads 
                  << " | " << std::setw(9) << std::fixed << std::setprecision(3) << time_ms
                  << " | " << std::setw(6) << std::fixed << std::setprecision(2) << speedup
                  << std::endl;
    }
}

// Test numerical precision
void test_precision() {
    std::cout << "\n=== Testing Numerical Precision ===\n";
    
    // Test with values that would lose precision in double
    const size_t n = 3;
    QuadBLAS::DefaultVector<> x(n), y(n);
    
    // Use values that demonstrate quad precision advantage
    x[0] = from_double(1e20);
    x[1] = from_double(1.0);
    x[2] = from_double(-1e20);
    
    y[0] = from_double(1.0);
    y[1] = from_double(1.0);
    y[2] = from_double(1.0);
    
    Sleef_quad quad_result = x.dot(y);
    
    // Compare with double precision calculation
    double double_sum = 1e20 * 1.0 + 1.0 * 1.0 + (-1e20) * 1.0;
    
    std::cout << "Test case: (1e20, 1, -1e20) • (1, 1, 1)" << std::endl;
    std::cout << "Quad precision result:   " << std::fixed << std::setprecision(1) << to_double(quad_result) << std::endl;
    std::cout << "Double precision result: " << std::fixed << std::setprecision(1) << double_sum << std::endl;
    std::cout << "Expected result:         1.0" << std::endl;
    
    // Test constants precision
    std::cout << "\nPrecision of mathematical constants:" << std::endl;
    std::cout << "π (quad): " << std::setprecision(30) << to_double(SLEEF_M_PIq) << std::endl;
    std::cout << "e (quad): " << std::setprecision(30) << to_double(SLEEF_M_Eq) << std::endl;
}

// Simple test to verify basic functionality
void test_basic() {
    std::cout << "\n=== Basic Functionality Test ===\n";
    
    // Test simple dot product
    const size_t n = 5;
    QuadBLAS::DefaultVector<> x(n), y(n);
    
    for (size_t i = 0; i < n; ++i) {
        x[i] = from_double(i + 1);  // 1, 2, 3, 4, 5
        y[i] = from_double(1.0);    // 1, 1, 1, 1, 1
    }
    
    Sleef_quad result = x.dot(y);
    double expected = 1 + 2 + 3 + 4 + 5;  // = 15
    
    std::cout << "Simple dot product test:" << std::endl;
    std::cout << "Result: " << to_double(result) << std::endl;
    std::cout << "Expected: " << expected << std::endl;
    std::cout << "Error: " << std::abs(to_double(result) - expected) << std::endl;
    
    if (std::abs(to_double(result) - expected) < 1e-10) {
        std::cout << "✓ PASS" << std::endl;
    } else {
        std::cout << "✗ FAIL" << std::endl;
    }
}

// Main test function
int main() {
    std::cout << "=== QuadBLAS Test Suite ===\n";
    std::cout << quadblas_get_version() << std::endl;
    std::cout << "Available threads: " << QuadBLAS::get_num_threads() << std::endl;
    
    // Check platform support
#ifdef QUADBLAS_X86_64
    std::cout << "Platform: x86-64 with vectorization" << std::endl;
#elif defined(QUADBLAS_AARCH64)
    std::cout << "Platform: AArch64 with vectorization" << std::endl;
#else
    std::cout << "Platform: Generic (scalar fallback)" << std::endl;
#endif

#ifdef _OPENMP
    std::cout << "OpenMP: Enabled" << std::endl;
#else
    std::cout << "OpenMP: Disabled" << std::endl;
#endif
    
    try {
        test_basic();
        test_dot_product();
        test_matrix_vector();
        test_matrix_matrix();
        test_precision();
        
#ifdef _OPENMP
        performance_comparison();
#endif
        
        std::cout << "\n=== All tests completed successfully! ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}