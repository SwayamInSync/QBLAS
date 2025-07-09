#include "include/quadblas/quadblas.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>

// ===============================================================================
// HELPER FUNCTIONS AND UTILITIES
// ===============================================================================

// Helper functions for quad precision conversion
double to_double(Sleef_quad q) {
    return static_cast<double>(Sleef_cast_to_doubleq1(q));
}

Sleef_quad from_double(double d) {
    return Sleef_cast_from_doubleq1(d);
}

// Test result tracking
struct TestResult {
    std::string name;
    bool passed;
    double max_error;
    double avg_error;
    size_t error_count;
    size_t total_elements;
    std::string details;
    
    TestResult(const std::string& n) : name(n), passed(false), max_error(0), avg_error(0), 
                                      error_count(0), total_elements(0) {}
};

class TestSuite {
private:
    std::vector<TestResult> results;
    double tolerance = 1e-12;  // Stricter tolerance for quad precision
    
public:
    void set_tolerance(double tol) { tolerance = tol; }
    
    void add_result(const TestResult& result) {
        results.push_back(result);
        std::cout << "[" << (result.passed ? "PASS" : "FAIL") << "] " << result.name;
        if (!result.passed) {
            std::cout << " (max_error: " << std::scientific << result.max_error << ")";
        }
        std::cout << std::endl;
        if (!result.details.empty()) {
            std::cout << "    " << result.details << std::endl;
        }
    }
    
    void print_summary() {
        size_t passed = 0, failed = 0;
        for (const auto& r : results) {
            if (r.passed) passed++; else failed++;
        }
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "TEST SUMMARY" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Total tests: " << results.size() << std::endl;
        std::cout << "Passed: " << passed << std::endl;
        std::cout << "Failed: " << failed << std::endl;
        
        if (failed > 0) {
            std::cout << "\nFAILED TESTS:" << std::endl;
            for (const auto& r : results) {
                if (!r.passed) {
                    std::cout << "  - " << r.name << " (error: " << std::scientific << r.max_error << ")" << std::endl;
                }
            }
        }
        
        std::cout << "\nOverall result: " << (failed == 0 ? "✓ ALL TESTS PASSED" : "✗ SOME TESTS FAILED") << std::endl;
    }
    
    double get_tolerance() const { return tolerance; }
};

// Random number generator for reproducible tests
class TestRandom {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> uniform_dist;
    std::normal_distribution<double> normal_dist;
    
public:
    TestRandom(unsigned seed = 42) : gen(seed), uniform_dist(-1.0, 1.0), normal_dist(0.0, 1.0) {}
    
    Sleef_quad uniform() { return from_double(uniform_dist(gen)); }
    Sleef_quad normal() { return from_double(normal_dist(gen)); }
    Sleef_quad range(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return from_double(dist(gen));
    }
    
    void fill_uniform(Sleef_quad* arr, size_t n) {
        for (size_t i = 0; i < n; ++i) arr[i] = uniform();
    }
    
    void fill_normal(Sleef_quad* arr, size_t n) {
        for (size_t i = 0; i < n; ++i) arr[i] = normal();
    }
};

// Utility to compare arrays and compute error statistics
TestResult compare_arrays(const std::string& test_name, const Sleef_quad* expected, 
                         const Sleef_quad* actual, size_t n, double tolerance) {
    TestResult result(test_name);
    result.total_elements = n;
    
    double sum_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double error = std::abs(to_double(expected[i]) - to_double(actual[i]));
        result.max_error = std::max(result.max_error, error);
        sum_error += error;
        if (error > tolerance) {
            result.error_count++;
        }
    }
    
    result.avg_error = sum_error / n;
    result.passed = (result.error_count == 0);
    
    if (!result.passed) {
        result.details = "Errors in " + std::to_string(result.error_count) + "/" + std::to_string(n) + " elements";
    }
    
    return result;
}

// ===============================================================================
// REFERENCE IMPLEMENTATIONS (GUARANTEED CORRECT)
// ===============================================================================

// Reference dot product
Sleef_quad reference_dot(size_t n, const Sleef_quad* x, size_t incx, 
                        const Sleef_quad* y, size_t incy) {
    Sleef_quad result = from_double(0.0);
    for (size_t i = 0; i < n; ++i) {
        result = Sleef_fmaq1_u05(x[i * incx], y[i * incy], result);
    }
    return result;
}

// Reference AXPY: y = alpha * x + y
void reference_axpy(size_t n, Sleef_quad alpha, const Sleef_quad* x, size_t incx,
                   Sleef_quad* y, size_t incy) {
    for (size_t i = 0; i < n; ++i) {
        y[i * incy] = Sleef_fmaq1_u05(alpha, x[i * incx], y[i * incy]);
    }
}

// Reference GEMV: y = alpha * A * x + beta * y
void reference_gemv(bool row_major, size_t m, size_t n, Sleef_quad alpha,
                   const Sleef_quad* A, size_t lda, const Sleef_quad* x, size_t incx,
                   Sleef_quad beta, Sleef_quad* y, size_t incy) {
    for (size_t i = 0; i < m; ++i) {
        Sleef_quad sum = from_double(0.0);
        for (size_t j = 0; j < n; ++j) {
            size_t a_idx = row_major ? i * lda + j : j * lda + i;
            sum = Sleef_fmaq1_u05(A[a_idx], x[j * incx], sum);
        }
        y[i * incy] = Sleef_fmaq1_u05(alpha, sum, Sleef_mulq1_u05(beta, y[i * incy]));
    }
}

// Reference GEMM: C = alpha * A * B + beta * C
void reference_gemm(bool row_major, size_t m, size_t n, size_t k, Sleef_quad alpha,
                   const Sleef_quad* A, size_t lda, const Sleef_quad* B, size_t ldb,
                   Sleef_quad beta, Sleef_quad* C, size_t ldc) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            Sleef_quad sum = from_double(0.0);
            for (size_t l = 0; l < k; ++l) {
                size_t a_idx = row_major ? i * lda + l : l * lda + i;
                size_t b_idx = row_major ? l * ldb + j : j * ldb + l;
                sum = Sleef_fmaq1_u05(A[a_idx], B[b_idx], sum);
            }
            size_t c_idx = row_major ? i * ldc + j : j * ldc + i;
            C[c_idx] = Sleef_fmaq1_u05(alpha, sum, Sleef_mulq1_u05(beta, C[c_idx]));
        }
    }
}

// ===============================================================================
// LEVEL 1 BLAS TESTS (Vector-Vector Operations)
// ===============================================================================

void test_level1_blas(TestSuite& suite) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "LEVEL 1 BLAS TESTS (Vector-Vector Operations)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    TestRandom rng;
    
    // Test 1: Small dot product
    {
        const size_t n = 10;
        std::vector<Sleef_quad> x(n), y(n);
        
        // Simple test pattern
        for (size_t i = 0; i < n; ++i) {
            x[i] = from_double(i + 1);  // 1, 2, 3, ..., 10
            y[i] = from_double(1.0);    // 1, 1, 1, ..., 1
        }
        
        Sleef_quad expected = reference_dot(n, x.data(), 1, y.data(), 1);
        Sleef_quad actual = QuadBLAS::dot(n, x.data(), 1, y.data(), 1);
        
        double error = std::abs(to_double(expected) - to_double(actual));
        TestResult result("DOT Small (n=10)");
        result.passed = error < suite.get_tolerance();
        result.max_error = error;
        result.total_elements = 1;
        suite.add_result(result);
    }
    
    // Test 2: Large dot product (tests vectorization)
    {
        const size_t n = 100000;
        std::vector<Sleef_quad> x(n), y(n);
        rng.fill_uniform(x.data(), n);
        rng.fill_uniform(y.data(), n);
        
        Sleef_quad expected = reference_dot(n, x.data(), 1, y.data(), 1);
        Sleef_quad actual = QuadBLAS::dot(n, x.data(), 1, y.data(), 1);
        
        double error = std::abs(to_double(expected) - to_double(actual));
        TestResult result("DOT Large (n=100k)");
        result.passed = error < suite.get_tolerance() * n;  // Allow for accumulation error
        result.max_error = error;
        result.total_elements = 1;
        suite.add_result(result);
    }
    
    // Test 3: Dot product with strides
    {
        const size_t n = 1000;
        const size_t incx = 3, incy = 2;
        std::vector<Sleef_quad> x(n * incx), y(n * incy);
        rng.fill_uniform(x.data(), n * incx);
        rng.fill_uniform(y.data(), n * incy);
        
        Sleef_quad expected = reference_dot(n, x.data(), incx, y.data(), incy);
        Sleef_quad actual = QuadBLAS::dot(n, x.data(), incx, y.data(), incy);
        
        double error = std::abs(to_double(expected) - to_double(actual));
        TestResult result("DOT with strides");
        result.passed = error < suite.get_tolerance() * n;
        result.max_error = error;
        suite.add_result(result);
    }
    
    // Test 4: AXPY operation
    {
        const size_t n = 5000;
        Sleef_quad alpha = from_double(2.5);
        std::vector<Sleef_quad> x(n), y_expected(n), y_actual(n);
        
        rng.fill_uniform(x.data(), n);
        rng.fill_uniform(y_expected.data(), n);
        std::copy(y_expected.begin(), y_expected.end(), y_actual.begin());
        
        reference_axpy(n, alpha, x.data(), 1, y_expected.data(), 1);
        
        // Test using C++ interface
        QuadBLAS::DefaultVector<> qx(n), qy(n);
        for (size_t i = 0; i < n; ++i) {
            qx[i] = x[i];
            qy[i] = y_actual[i];
        }
        qy.axpy(alpha, qx);
        for (size_t i = 0; i < n; ++i) {
            y_actual[i] = qy[i];
        }
        
        TestResult result = compare_arrays("AXPY operation", y_expected.data(), y_actual.data(), n, suite.get_tolerance());
        suite.add_result(result);
    }
    
    // Test 5: Vector norm
    {
        const size_t n = 1000;
        std::vector<Sleef_quad> x(n);
        
        // Use a simple pattern for exact computation
        for (size_t i = 0; i < n; ++i) {
            x[i] = from_double(i + 1);
        }
        
        // Expected norm = sqrt(1^2 + 2^2 + ... + n^2) = sqrt(n(n+1)(2n+1)/6)
        double sum_squares = n * (n + 1) * (2 * n + 1) / 6.0;
        Sleef_quad expected_norm = from_double(std::sqrt(sum_squares));
        
        QuadBLAS::DefaultVector<> qx(n);
        for (size_t i = 0; i < n; ++i) qx[i] = x[i];
        Sleef_quad actual_norm = qx.norm();
        
        double error = std::abs(to_double(expected_norm) - to_double(actual_norm));
        TestResult result("Vector norm");
        result.passed = error < suite.get_tolerance() * std::sqrt(n);
        result.max_error = error;
        suite.add_result(result);
    }
}

// ===============================================================================
// LEVEL 2 BLAS TESTS (Matrix-Vector Operations)
// ===============================================================================

void test_level2_blas(TestSuite& suite) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "LEVEL 2 BLAS TESTS (Matrix-Vector Operations)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    TestRandom rng;
    
    // Test 1: Small GEMV
    {
        const size_t m = 3, n = 3;
        std::vector<Sleef_quad> A = {
            from_double(1), from_double(2), from_double(3),
            from_double(4), from_double(5), from_double(6),
            from_double(7), from_double(8), from_double(9)
        };
        std::vector<Sleef_quad> x = {from_double(1), from_double(1), from_double(1)};
        std::vector<Sleef_quad> y_expected(m), y_actual(m);
        
        // Initialize y
        for (size_t i = 0; i < m; ++i) {
            y_expected[i] = y_actual[i] = from_double(0.0);
        }
        
        reference_gemv(true, m, n, from_double(1.0), A.data(), n, x.data(), 1, 
                      from_double(0.0), y_expected.data(), 1);
        
        QuadBLAS::gemv(QuadBLAS::Layout::RowMajor, m, n, from_double(1.0), A.data(), n,
                      x.data(), 1, from_double(0.0), y_actual.data(), 1);
        
        TestResult result = compare_arrays("GEMV Small (3x3)", y_expected.data(), y_actual.data(), m, suite.get_tolerance());
        suite.add_result(result);
    }
    
    // Test 2: Medium GEMV (tests optimization paths)
    {
        const size_t m = 100, n = 100;
        std::vector<Sleef_quad> A(m * n), x(n), y_expected(m), y_actual(m);
        
        rng.fill_uniform(A.data(), m * n);
        rng.fill_uniform(x.data(), n);
        rng.fill_uniform(y_expected.data(), m);
        std::copy(y_expected.begin(), y_expected.end(), y_actual.begin());
        
        Sleef_quad alpha = from_double(1.5), beta = from_double(0.5);
        
        reference_gemv(true, m, n, alpha, A.data(), n, x.data(), 1, beta, y_expected.data(), 1);
        QuadBLAS::gemv(QuadBLAS::Layout::RowMajor, m, n, alpha, A.data(), n, x.data(), 1, 
                      beta, y_actual.data(), 1);
        
        TestResult result = compare_arrays("GEMV Medium (100x100)", y_expected.data(), y_actual.data(), m, suite.get_tolerance() * n);
        suite.add_result(result);
    }
    
    // Test 3: Large GEMV (tests parallel execution)
    {
        const size_t m = 2000, n = 2000;
        std::vector<Sleef_quad> A(m * n), x(n), y_expected(m), y_actual(m);
        
        rng.fill_normal(A.data(), m * n);
        rng.fill_normal(x.data(), n);
        
        for (size_t i = 0; i < m; ++i) {
            y_expected[i] = y_actual[i] = from_double(0.0);
        }
        
        reference_gemv(true, m, n, from_double(1.0), A.data(), n, x.data(), 1, 
                      from_double(0.0), y_expected.data(), 1);
        QuadBLAS::gemv(QuadBLAS::Layout::RowMajor, m, n, from_double(1.0), A.data(), n,
                      x.data(), 1, from_double(0.0), y_actual.data(), 1);
        
        TestResult result = compare_arrays("GEMV Large (2000x2000)", y_expected.data(), y_actual.data(), m, suite.get_tolerance() * n);
        suite.add_result(result);
    }
    
    // Test 4: GEMV with C++ interface
    {
        const size_t m = 50, n = 50;
        QuadBLAS::DefaultMatrix<> A(m, n);
        QuadBLAS::DefaultVector<> x(n), y_expected(m), y_actual(m);
        
        // Initialize with specific pattern
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                A(i, j) = from_double((i + j + 1) * 0.1);
            }
        }
        for (size_t i = 0; i < n; ++i) {
            x[i] = from_double(i + 1);
        }
        
        // Compute expected using reference
        std::vector<Sleef_quad> A_flat(m * n), x_flat(n), y_flat(m);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                A_flat[i * n + j] = A(i, j);
            }
        }
        for (size_t i = 0; i < n; ++i) x_flat[i] = x[i];
        for (size_t i = 0; i < m; ++i) y_flat[i] = from_double(0.0);
        
        reference_gemv(true, m, n, from_double(1.0), A_flat.data(), n, x_flat.data(), 1,
                      from_double(0.0), y_flat.data(), 1);
        
        // Test QuadBLAS C++ interface
        for (size_t i = 0; i < m; ++i) y_actual[i] = from_double(0.0);
        A.gemv(from_double(1.0), x, from_double(0.0), y_actual);
        
        std::vector<Sleef_quad> y_actual_flat(m);
        for (size_t i = 0; i < m; ++i) y_actual_flat[i] = y_actual[i];
        
        TestResult result = compare_arrays("GEMV C++ interface", y_flat.data(), y_actual_flat.data(), m, suite.get_tolerance() * n);
        suite.add_result(result);
    }
}

// ===============================================================================
// LEVEL 3 BLAS TESTS (Matrix-Matrix Operations)
// ===============================================================================

void test_level3_blas(TestSuite& suite) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "LEVEL 3 BLAS TESTS (Matrix-Matrix Operations)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    TestRandom rng;
    
    // Test 1: Tiny GEMM (2x2) - tests basic correctness
    {
        const size_t m = 2, n = 2, k = 2;
        std::vector<Sleef_quad> A = {from_double(1), from_double(2), from_double(3), from_double(4)};
        std::vector<Sleef_quad> B = {from_double(5), from_double(6), from_double(7), from_double(8)};
        std::vector<Sleef_quad> C_expected(m * n), C_actual(m * n);
        
        // Initialize C to zero
        for (size_t i = 0; i < m * n; ++i) {
            C_expected[i] = C_actual[i] = from_double(0.0);
        }
        
        reference_gemm(true, m, n, k, from_double(1.0), A.data(), k, B.data(), n,
                      from_double(0.0), C_expected.data(), n);
        QuadBLAS::gemm(QuadBLAS::Layout::RowMajor, m, n, k, from_double(1.0), A.data(), k,
                      B.data(), n, from_double(0.0), C_actual.data(), n);
        
        TestResult result = compare_arrays("GEMM Tiny (2x2)", C_expected.data(), C_actual.data(), m * n, suite.get_tolerance());
        suite.add_result(result);
    }
    
    // Test 2: Small GEMM (uses simple implementation)
    {
        const size_t m = 16, n = 16, k = 16;
        std::vector<Sleef_quad> A(m * k), B(k * n), C_expected(m * n), C_actual(m * n);
        
        rng.fill_uniform(A.data(), m * k);
        rng.fill_uniform(B.data(), k * n);
        
        for (size_t i = 0; i < m * n; ++i) {
            C_expected[i] = C_actual[i] = from_double(0.0);
        }
        
        reference_gemm(true, m, n, k, from_double(1.0), A.data(), k, B.data(), n,
                      from_double(0.0), C_expected.data(), n);
        QuadBLAS::gemm(QuadBLAS::Layout::RowMajor, m, n, k, from_double(1.0), A.data(), k,
                      B.data(), n, from_double(0.0), C_actual.data(), n);
        
        TestResult result = compare_arrays("GEMM Small (16x16)", C_expected.data(), C_actual.data(), m * n, suite.get_tolerance() * k);
        suite.add_result(result);
    }
    
    // Test 3: Boundary GEMM (at the threshold) - this is critical!
    {
        const size_t m = 32, n = 32, k = 32;  // Exactly at threshold
        std::vector<Sleef_quad> A(m * k), B(k * n), C_expected(m * n), C_actual(m * n);
        
        rng.fill_uniform(A.data(), m * k);
        rng.fill_uniform(B.data(), k * n);
        
        for (size_t i = 0; i < m * n; ++i) {
            C_expected[i] = C_actual[i] = from_double(0.0);
        }
        
        reference_gemm(true, m, n, k, from_double(1.0), A.data(), k, B.data(), n,
                      from_double(0.0), C_expected.data(), n);
        QuadBLAS::gemm(QuadBLAS::Layout::RowMajor, m, n, k, from_double(1.0), A.data(), k,
                      B.data(), n, from_double(0.0), C_actual.data(), n);
        
        TestResult result = compare_arrays("GEMM Boundary (32x32)", C_expected.data(), C_actual.data(), m * n, suite.get_tolerance() * k);
        suite.add_result(result);
    }
    
    // Test 4: Medium GEMM (triggers blocked algorithm) - this should expose bugs!
    {
        const size_t m = 64, n = 64, k = 64;  // Beyond threshold - uses blocked algorithm
        std::vector<Sleef_quad> A(m * k), B(k * n), C_expected(m * n), C_actual(m * n);
        
        // Use patterns that would expose indexing errors
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < k; ++j) {
                A[i * k + j] = from_double(i + j + 1);  // Simple pattern
            }
        }
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < n; ++j) {
                B[i * n + j] = from_double((i * j + 1) % 7);  // Another pattern
            }
        }
        
        for (size_t i = 0; i < m * n; ++i) {
            C_expected[i] = C_actual[i] = from_double(0.0);
        }
        
        reference_gemm(true, m, n, k, from_double(1.0), A.data(), k, B.data(), n,
                      from_double(0.0), C_expected.data(), n);
        QuadBLAS::gemm(QuadBLAS::Layout::RowMajor, m, n, k, from_double(1.0), A.data(), k,
                      B.data(), n, from_double(0.0), C_actual.data(), n);
        
        TestResult result = compare_arrays("GEMM Medium (64x64) - BLOCKED ALGORITHM", C_expected.data(), C_actual.data(), m * n, suite.get_tolerance() * k);
        suite.add_result(result);
    }
    
    // Test 5: Large GEMM (tests scalability)
    {
        const size_t m = 200, n = 200, k = 200;
        std::vector<Sleef_quad> A(m * k), B(k * n), C_expected(m * n), C_actual(m * n);
        
        std::cout << "    Generating large matrices (this may take a moment)..." << std::endl;
        rng.fill_normal(A.data(), m * k);
        rng.fill_normal(B.data(), k * n);
        
        for (size_t i = 0; i < m * n; ++i) {
            C_expected[i] = C_actual[i] = from_double(0.0);
        }
        
        std::cout << "    Computing reference result..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        reference_gemm(true, m, n, k, from_double(1.0), A.data(), k, B.data(), n,
                      from_double(0.0), C_expected.data(), n);
        auto ref_time = std::chrono::high_resolution_clock::now() - start;
        
        std::cout << "    Computing QuadBLAS result..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        QuadBLAS::gemm(QuadBLAS::Layout::RowMajor, m, n, k, from_double(1.0), A.data(), k,
                      B.data(), n, from_double(0.0), C_actual.data(), n);
        auto qblas_time = std::chrono::high_resolution_clock::now() - start;
        
        TestResult result = compare_arrays("GEMM Large (200x200)", C_expected.data(), C_actual.data(), m * n, suite.get_tolerance() * k);
        
        auto ref_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ref_time).count();
        auto qblas_ms = std::chrono::duration_cast<std::chrono::milliseconds>(qblas_time).count();
        result.details += " | Time: ref=" + std::to_string(ref_ms) + "ms, qblas=" + std::to_string(qblas_ms) + "ms";
        
        suite.add_result(result);
    }
    
    // Test 6: GEMM with alpha and beta scaling
    {
        const size_t m = 20, n = 20, k = 20;
        std::vector<Sleef_quad> A(m * k), B(k * n), C_expected(m * n), C_actual(m * n);
        
        rng.fill_uniform(A.data(), m * k);
        rng.fill_uniform(B.data(), k * n);
        rng.fill_uniform(C_expected.data(), m * n);
        std::copy(C_expected.begin(), C_expected.end(), C_actual.begin());
        
        Sleef_quad alpha = from_double(2.5), beta = from_double(1.5);
        
        reference_gemm(true, m, n, k, alpha, A.data(), k, B.data(), n, beta, C_expected.data(), n);
        QuadBLAS::gemm(QuadBLAS::Layout::RowMajor, m, n, k, alpha, A.data(), k,
                      B.data(), n, beta, C_actual.data(), n);
        
        TestResult result = compare_arrays("GEMM with alpha/beta scaling", C_expected.data(), C_actual.data(), m * n, suite.get_tolerance() * k);
        suite.add_result(result);
    }
    
    // Test 7: GEMM C++ interface
    {
        const size_t m = 25, n = 25, k = 25;
        QuadBLAS::DefaultMatrix<> A(m, k), B(k, n), C_expected(m, n), C_actual(m, n);
        
        // Initialize matrices
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < k; ++j) {
                A(i, j) = rng.uniform();
            }
        }
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < n; ++j) {
                B(i, j) = rng.uniform();
            }
        }
        
        // Compute expected using reference
        std::vector<Sleef_quad> A_flat(m * k), B_flat(k * n), C_flat(m * n);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < k; ++j) {
                A_flat[i * k + j] = A(i, j);
            }
        }
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < n; ++j) {
                B_flat[i * n + j] = B(i, j);
            }
        }
        for (size_t i = 0; i < m * n; ++i) C_flat[i] = from_double(0.0);
        
        reference_gemm(true, m, n, k, from_double(1.0), A_flat.data(), k, B_flat.data(), n,
                      from_double(0.0), C_flat.data(), n);
        
        // Test C++ interface
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                C_actual(i, j) = from_double(0.0);
            }
        }
        A.gemm(from_double(1.0), B, from_double(0.0), C_actual);
        
        std::vector<Sleef_quad> C_actual_flat(m * n);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                C_actual_flat[i * n + j] = C_actual(i, j);
            }
        }
        
        TestResult result = compare_arrays("GEMM C++ interface", C_flat.data(), C_actual_flat.data(), m * n, suite.get_tolerance() * k);
        suite.add_result(result);
    }
}

// ===============================================================================
// EDGE CASE AND STRESS TESTS
// ===============================================================================

void test_edge_cases(TestSuite& suite) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "EDGE CASE AND STRESS TESTS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    TestRandom rng;
    
    // Test 1: Non-square matrices
    {
        const size_t m = 47, n = 31, k = 23;  // Prime numbers to stress algorithms
        std::vector<Sleef_quad> A(m * k), B(k * n), C_expected(m * n), C_actual(m * n);
        
        rng.fill_uniform(A.data(), m * k);
        rng.fill_uniform(B.data(), k * n);
        
        for (size_t i = 0; i < m * n; ++i) {
            C_expected[i] = C_actual[i] = from_double(0.0);
        }
        
        reference_gemm(true, m, n, k, from_double(1.0), A.data(), k, B.data(), n,
                      from_double(0.0), C_expected.data(), n);
        QuadBLAS::gemm(QuadBLAS::Layout::RowMajor, m, n, k, from_double(1.0), A.data(), k,
                      B.data(), n, from_double(0.0), C_actual.data(), n);
        
        TestResult result = compare_arrays("Non-square matrices (47x31x23)", C_expected.data(), C_actual.data(), m * n, suite.get_tolerance() * k);
        suite.add_result(result);
    }
    
    // Test 2: Very small matrices (1x1)
    {
        const size_t m = 1, n = 1, k = 1;
        Sleef_quad A = from_double(3.0), B = from_double(4.0), C_expected = from_double(0.0), C_actual = from_double(0.0);
        
        reference_gemm(true, m, n, k, from_double(1.0), &A, k, &B, n, from_double(0.0), &C_expected, n);
        QuadBLAS::gemm(QuadBLAS::Layout::RowMajor, m, n, k, from_double(1.0), &A, k, &B, n, from_double(0.0), &C_actual, n);
        
        TestResult result = compare_arrays("Minimal matrices (1x1)", &C_expected, &C_actual, 1, suite.get_tolerance());
        suite.add_result(result);
    }
    
    // Test 3: Identity matrix multiplication
    {
        const size_t n = 50;
        std::vector<Sleef_quad> I(n * n), A(n * n), C_expected(n * n), C_actual(n * n);
        
        // Create identity matrix
        for (size_t i = 0; i < n * n; ++i) I[i] = from_double(0.0);
        for (size_t i = 0; i < n; ++i) I[i * n + i] = from_double(1.0);
        
        rng.fill_uniform(A.data(), n * n);
        
        // I * A should equal A
        for (size_t i = 0; i < n * n; ++i) {
            C_expected[i] = A[i];
            C_actual[i] = from_double(0.0);
        }
        
        QuadBLAS::gemm(QuadBLAS::Layout::RowMajor, n, n, n, from_double(1.0), I.data(), n,
                      A.data(), n, from_double(0.0), C_actual.data(), n);
        
        TestResult result = compare_arrays("Identity matrix multiplication", C_expected.data(), C_actual.data(), n * n, suite.get_tolerance());
        suite.add_result(result);
    }
    
    // Test 4: Numerical precision test
    {
        std::cout << "    Testing numerical precision with extreme values..." << std::endl;
        const size_t n = 10;
        QuadBLAS::DefaultVector<> x(n), y(n);
        
        // Test case that would lose precision in double
        x[0] = from_double(1e20);
        x[1] = from_double(1.0);
        x[2] = from_double(-1e20);
        for (size_t i = 3; i < n; ++i) x[i] = from_double(0.0);
        
        for (size_t i = 0; i < n; ++i) y[i] = from_double(1.0);
        
        Sleef_quad result = x.dot(y);
        double result_double = to_double(result);
        
        TestResult test_result("Numerical precision test");
        test_result.passed = std::abs(result_double - 1.0) < 1e-10;  // Should be exactly 1.0
        test_result.max_error = std::abs(result_double - 1.0);
        test_result.total_elements = 1;
        if (!test_result.passed) {
            test_result.details = "Expected 1.0, got " + std::to_string(result_double);
        }
        suite.add_result(test_result);
    }
}

// ===============================================================================
// MAIN TEST PROGRAM
// ===============================================================================

int main() {
    std::cout << "QuadBLAS Comprehensive Correctness Test Suite" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    std::cout << "System Information:" << std::endl;
    std::cout << "  " << quadblas_get_version() << std::endl;
    std::cout << "  Available threads: " << QuadBLAS::get_num_threads() << std::endl;
    
#ifdef QUADBLAS_X86_64
    std::cout << "  Platform: x86-64 with SIMD vectorization" << std::endl;
#elif defined(QUADBLAS_AARCH64)
    std::cout << "  Platform: AArch64 with NEON vectorization" << std::endl;
#else
    std::cout << "  Platform: Generic (scalar fallback)" << std::endl;
#endif

#ifdef _OPENMP
    std::cout << "  OpenMP: Enabled" << std::endl;
#else
    std::cout << "  OpenMP: Disabled" << std::endl;
#endif

    TestSuite suite;
    suite.set_tolerance(1e-12);  // Stricter tolerance for quad precision
    
    try {
        test_level1_blas(suite);
        test_level2_blas(suite);
        test_level3_blas(suite);
        test_edge_cases(suite);
        
        suite.print_summary();
        
    } catch (const std::exception& e) {
        std::cerr << "\nFATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}