#include "include/quadblas/quadblas.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cassert>
#include <string>
#include <chrono>

// Helper functions
double to_double(Sleef_quad q) {
    return static_cast<double>(Sleef_cast_to_doubleq1(q));
}

Sleef_quad from_double(double d) {
    return Sleef_cast_from_doubleq1(d);
}

// Test assertion helper with detailed error reporting
void assert_quad_equal(Sleef_quad actual, Sleef_quad expected, double rtol = 1e-15, double atol = 1e-15, const std::string& message = "") {
    double actual_d = to_double(actual);
    double expected_d = to_double(expected);
    double error = std::abs(actual_d - expected_d);
    double tolerance = atol + rtol * std::abs(expected_d);
    
    if (error > tolerance) {
        std::cerr << "❌ Assertion failed: " << message << std::endl;
        std::cerr << "   Expected: " << std::setprecision(20) << expected_d << std::endl;
        std::cerr << "   Actual:   " << std::setprecision(20) << actual_d << std::endl;
        std::cerr << "   Error:    " << std::setprecision(20) << error << std::endl;
        std::cerr << "   Tolerance:" << std::setprecision(20) << tolerance << std::endl;
        std::exit(1);
    }
}

// Timer class for performance measurement
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        return duration.count() / 1e6; // Convert to milliseconds
    }
};

/**
 * Test 1: Large square matrices
 * Equivalent to Python's test_large_square_matrices
 */
void test_large_square_matrices(size_t size) {
    std::cout << "🔬 Testing large square matrices (size=" << size << "x" << size << ")..." << std::endl;
    
    Timer timer;
    timer.start();
    
    // Create matrices using QuadBLAS
    QuadBLAS::DefaultMatrix<> A(size, size);
    QuadBLAS::DefaultMatrix<> B(size, size);
    QuadBLAS::DefaultMatrix<> result(size, size);
    
    // Initialize A: near-diagonal (1.0 on diagonal, 0.1 elsewhere)
    // Python: A_vals = [1.0 if i == j else 0.1 for i in range(size) for j in range(size)]
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            A(i, j) = (i == j) ? from_double(1.0) : from_double(0.1);
        }
    }
    
    // Initialize B: all ones
    // Python: B_vals = [1.0] * (size * size)
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            B(i, j) = from_double(1.0);
        }
    }
    
    // Initialize result to zero
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            result(i, j) = from_double(0.0);
        }
    }
    
    // Perform matrix multiplication: result = 1.0 * A * B + 0.0 * result
    // Equivalent to Python's: result = np.matmul(A, B)
    A.gemm(from_double(1.0), B, from_double(0.0), result);
    
    double compute_time = timer.stop_ms();
    
    // Verify results
    // Python: expected_value = 1.0 + 0.1 * (size - 1)
    // Each element = sum of a row in A = 1.0 + 0.1*(size-1)
    double expected_value = 1.0 + 0.1 * (size - 1);
    Sleef_quad expected_quad = from_double(expected_value);
    
    // Check diagonal and off-diagonal elements
    assert_quad_equal(result(0, 0), expected_quad, 1e-15, 1e-15, "result[0,0]");
    
    if (size > 1) {
        assert_quad_equal(result(0, 1), expected_quad, 1e-15, 1e-15, "result[0,1]");
    }
    
    // Additional verification: check a few more elements
    if (size > 2) {
        assert_quad_equal(result(1, 0), expected_quad, 1e-15, 1e-15, "result[1,0]");
        assert_quad_equal(result(size/2, size/2), expected_quad, 1e-15, 1e-15, "result[size/2,size/2]");
    }
    
    std::cout << "   ✓ Expected value: " << expected_value << std::endl;
    std::cout << "   ✓ Computation time: " << std::fixed << std::setprecision(2) << compute_time << " ms" << std::endl;
    std::cout << "   ✓ Large square matrices test (size=" << size << ") PASSED!" << std::endl;
    std::cout << std::endl;
}

/**
 * Test 2: Large vector operations  
 * Equivalent to Python's test_large_vector_operations
 */
void test_large_vector_operations() {
    std::cout << "🔬 Testing large vector operations..." << std::endl;
    
    const size_t size = 1000;
    
    Timer timer;
    timer.start();
    
    // Create vectors using QuadBLAS
    QuadBLAS::DefaultVector<> x(size), y(size);
    
    // Initialize vectors
    // Python: x_vals = [1.0] * size, y_vals = [2.0] * size
    for (size_t i = 0; i < size; ++i) {
        x[i] = from_double(1.0);  // x = [1.0, 1.0, ...]
        y[i] = from_double(2.0);  // y = [2.0, 2.0, ...]
    }
    
    // Perform dot product
    // Equivalent to Python's: result = np.matmul(x, y)
    Sleef_quad result = x.dot(y);
    
    double compute_time = timer.stop_ms();
    
    // Expected = size * 1.0 * 2.0 = 2000.0
    // Python: expected = size * 1.0 * 2.0
    double expected = size * 1.0 * 2.0;
    Sleef_quad expected_quad = from_double(expected);
    
    assert_quad_equal(result, expected_quad, 1e-15, 1e-15, "large vector dot product");
    
    std::cout << "   ✓ Vector size: " << size << std::endl;
    std::cout << "   ✓ Expected result: " << expected << std::endl;
    std::cout << "   ✓ Actual result: " << to_double(result) << std::endl;
    std::cout << "   ✓ Computation time: " << std::fixed << std::setprecision(2) << compute_time << " ms" << std::endl;
    std::cout << "   ✓ Large vector operations test PASSED!" << std::endl;
    std::cout << std::endl;
}

/**
 * Test 3: Rectangular large matrices
 * Equivalent to Python's test_rectangular_large_matrices
 */
void test_rectangular_large_matrices() {
    std::cout << "🔬 Testing rectangular large matrices..." << std::endl;
    
    const size_t m = 100, n = 80, k = 120;
    
    Timer timer;
    timer.start();
    
    // Create matrices using QuadBLAS
    QuadBLAS::DefaultMatrix<> A(m, k);
    QuadBLAS::DefaultMatrix<> B(k, n);
    QuadBLAS::DefaultMatrix<> result(m, n);
    
    // Initialize A with pattern: (i + j + 1) % 10
    // Python: A_vals = [(i + j + 1) % 10 for i in range(m) for j in range(k)]
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            double val = (i + j + 1) % 10;
            A(i, j) = from_double(val);
        }
    }
    
    // Initialize B with pattern: (i + j + 1) % 10
    // Python: B_vals = [(i + j + 1) % 10 for i in range(k) for j in range(n)]
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double val = (i + j + 1) % 10;
            B(i, j) = from_double(val);
        }
    }
    
    // Initialize result to zero
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result(i, j) = from_double(0.0);
        }
    }
    
    // Perform matrix multiplication: result = 1.0 * A * B + 0.0 * result
    // Equivalent to Python's: result = np.matmul(A, B)
    A.gemm(from_double(1.0), B, from_double(0.0), result);
    
    double compute_time = timer.stop_ms();
    
    // Verify that result doesn't contain NaN or inf
    // Python: Check first few elements for NaN/inf
    size_t check_elements = 0;
    for (size_t i = 0; i < std::min(size_t(10), m); ++i) {
        for (size_t j = 0; j < std::min(size_t(10), n); ++j) {
            double val = to_double(result(i, j));
            
            if (std::isnan(val)) {
                std::cerr << "❌ NaN found at position (" << i << "," << j << ")" << std::endl;
                std::exit(1);
            }
            
            if (std::isinf(val)) {
                std::cerr << "❌ Inf found at position (" << i << "," << j << ")" << std::endl;
                std::exit(1);
            }
            check_elements++;
        }
    }
    
    // Additional verification: compute expected value for one element manually
    // Let's verify result(0, 0) by computing the dot product manually
    Sleef_quad expected_00 = from_double(0.0);
    for (size_t l = 0; l < k; ++l) {
        expected_00 = Sleef_fmaq1_u05(A(0, l), B(l, 0), expected_00);
    }
    
    assert_quad_equal(result(0, 0), expected_00, 1e-12, 1e-12, "result[0,0] manual verification");
    
    std::cout << "   ✓ Matrix dimensions: A(" << m << "x" << k << ") × B(" << k << "x" << n << ") = C(" << m << "x" << n << ")" << std::endl;
    std::cout << "   ✓ Checked " << check_elements << " elements for NaN/Inf" << std::endl;
    std::cout << "   ✓ Manual verification: result[0,0] = " << to_double(result(0, 0)) << std::endl;
    std::cout << "   ✓ Computation time: " << std::fixed << std::setprecision(2) << compute_time << " ms" << std::endl;
    std::cout << "   ✓ Rectangular large matrices test PASSED!" << std::endl;
    std::cout << std::endl;
}

/**
 * Additional performance test to stress test the library
 */
void test_performance_stress() {
    std::cout << "🔬 Performance stress test..." << std::endl;
    
    const size_t size = 500;
    
    QuadBLAS::DefaultMatrix<> A(size, size);
    QuadBLAS::DefaultMatrix<> B(size, size);
    QuadBLAS::DefaultMatrix<> result(size, size);
    
    // Initialize with random-like pattern
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            A(i, j) = from_double(std::sin(i * 0.1 + j * 0.1));
            B(i, j) = from_double(std::cos(i * 0.1 + j * 0.1));
            result(i, j) = from_double(0.0);
        }
    }
    
    Timer timer;
    timer.start();
    
    // Perform matrix multiplication
    A.gemm(from_double(1.0), B, from_double(0.0), result);
    
    double compute_time = timer.stop_ms();
    double gflops = (2.0 * size * size * size) / (compute_time * 1e6);  // GFLOP/s
    
    // Verify no NaN/Inf in random sample
    bool all_finite = true;
    for (size_t i = 0; i < size && all_finite; i += size/10) {
        for (size_t j = 0; j < size && all_finite; j += size/10) {
            double val = to_double(result(i, j));
            if (std::isnan(val) || std::isinf(val)) {
                all_finite = false;
            }
        }
    }
    
    if (!all_finite) {
        std::cerr << "❌ Performance stress test failed: found NaN/Inf values" << std::endl;
        std::exit(1);
    }
    
    std::cout << "   ✓ Matrix size: " << size << "x" << size << std::endl;
    std::cout << "   ✓ Computation time: " << std::fixed << std::setprecision(2) << compute_time << " ms" << std::endl;
    std::cout << "   ✓ Performance: " << std::fixed << std::setprecision(3) << gflops << " GFLOPS" << std::endl;
    std::cout << "   ✓ Performance stress test PASSED!" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "🧮 QuadBLAS Large Matrix Comprehensive Tests" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Library: " << quadblas_get_version() << std::endl;
    std::cout << "Threads: " << QuadBLAS::get_num_threads() << std::endl;
    
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
    
    std::cout << std::string(60, '=') << std::endl;
    std::cout << std::endl;
    
    try {
        // Test 1: Large square matrices with different sizes (parametrized test)
        // Python: @pytest.mark.parametrize("size", [50, 100, 200])
        std::vector<size_t> sizes = {50, 100, 200};
        for (size_t size : sizes) {
            test_large_square_matrices(size);
        }
        
        // Test 2: Large vector operations
        test_large_vector_operations();
        
        // Test 3: Rectangular large matrices
        test_rectangular_large_matrices();
        
        // Additional: Performance stress test
        test_performance_stress();
        
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "🎉 ALL TESTS PASSED! 🎉" << std::endl;
        std::cout << "QuadBLAS is working correctly for large matrix operations." << std::endl;
        std::cout << "The library demonstrates excellent numerical stability" << std::endl;
        std::cout << "and performance with quadruple precision arithmetic." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Unknown test error occurred" << std::endl;
        return 1;
    }
    
    return 0;
}