#include "quadblas.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>

// Helper functions
double to_double(Sleef_quad q) {
    return static_cast<double>(Sleef_cast_to_doubleq1(q));
}

Sleef_quad from_double(double d) {
    return Sleef_cast_from_doubleq1(d);
}

void test_quadVector_fixed() {
    std::cout << "\n=== Testing Fixed QuadVector ===\n";
    
    QuadBLAS::QuadVector v1(from_double(2.0), from_double(3.0));
    QuadBLAS::QuadVector v2(from_double(4.0), from_double(5.0));
    
    std::cout << "v1: [" << to_double(v1.get(0)) << ", " << to_double(v1.get(1)) << "]" << std::endl;
    std::cout << "v2: [" << to_double(v2.get(0)) << ", " << to_double(v2.get(1)) << "]" << std::endl;
    
    // Test store with workaround
    Sleef_quad result1[2], result2[2];
    v1.store(result1);
    v2.store(result2);
    
    std::cout << "Stored v1: [" << to_double(result1[0]) << ", " << to_double(result1[1]) << "]" << std::endl;
    std::cout << "Stored v2: [" << to_double(result2[0]) << ", " << to_double(result2[1]) << "]" << std::endl;
    
    // Test arithmetic
    QuadBLAS::QuadVector sum = v1 + v2;
    QuadBLAS::QuadVector prod = v1 * v2;
    
    std::cout << "v1 + v2: [" << to_double(sum.get(0)) << ", " << to_double(sum.get(1)) << "]" << std::endl;
    std::cout << "v1 * v2: [" << to_double(prod.get(0)) << ", " << to_double(prod.get(1)) << "]" << std::endl;
    
    // Test FMA: v1 * v2 + sum (should be [2*4+6, 3*5+8] = [14, 23])
    QuadBLAS::QuadVector fma_result = v1.fma(v2, sum);
    std::cout << "v1*v2 + sum: [" << to_double(fma_result.get(0)) << ", " << to_double(fma_result.get(1)) 
              << "] (expected [14, 23])" << std::endl;
}

void test_dot_fixed() {
    std::cout << "\n=== Testing Fixed Dot Product ===\n";
    
    const size_t n = 5;
    Sleef_quad x[5] = {from_double(1), from_double(2), from_double(3), from_double(4), from_double(5)};
    Sleef_quad y[5] = {from_double(1), from_double(1), from_double(1), from_double(1), from_double(1)};
    
    std::cout << "x: [1, 2, 3, 4, 5]" << std::endl;
    std::cout << "y: [1, 1, 1, 1, 1]" << std::endl;
    
    // Test vectorized kernel
    Sleef_quad vectorized_result = QuadBLAS::dot_kernel_vectorized(x, y, n);
    std::cout << "Vectorized dot: " << to_double(vectorized_result) << " (expected 15)" << std::endl;
    
    // Test full dot function
    Sleef_quad full_result = QuadBLAS::dot(n, x, 1, y, 1);
    std::cout << "Full dot function: " << to_double(full_result) << " (expected 15)" << std::endl;
}

void test_vector_class_fixed() {
    std::cout << "\n=== Testing Fixed Vector Class ===\n";
    
    const size_t n = 5;
    QuadBLAS::DefaultVector<> x(n), y(n);
    
    for (size_t i = 0; i < n; ++i) {
        x[i] = from_double(i + 1);  // 1, 2, 3, 4, 5
        y[i] = from_double(1.0);    // 1, 1, 1, 1, 1
    }
    
    std::cout << "x: [1, 2, 3, 4, 5]" << std::endl;
    std::cout << "y: [1, 1, 1, 1, 1]" << std::endl;
    
    Sleef_quad result = x.dot(y);
    std::cout << "Vector class dot: " << to_double(result) << " (expected 15)" << std::endl;
    
    // Test norm
    Sleef_quad norm_result = x.norm();
    double expected_norm = std::sqrt(1 + 4 + 9 + 16 + 25); // sqrt(55) ≈ 7.416
    std::cout << "Vector norm: " << to_double(norm_result) << " (expected " << expected_norm << ")" << std::endl;
}

void test_simple_gemv() {
    std::cout << "\n=== Testing Simple GEMV ===\n";
    
    // 2x2 matrix test
    const size_t m = 2, n = 2;
    QuadBLAS::DefaultMatrix<> A(m, n);
    QuadBLAS::DefaultVector<> x(n), y(m);
    
    // A = [[1, 2], [3, 4]]
    A(0, 0) = from_double(1.0); A(0, 1) = from_double(2.0);
    A(1, 0) = from_double(3.0); A(1, 1) = from_double(4.0);
    
    // x = [1, 1]
    x[0] = from_double(1.0);
    x[1] = from_double(1.0);
    
    std::cout << "A = [[1, 2], [3, 4]]" << std::endl;
    std::cout << "x = [1, 1]" << std::endl;
    
    A.gemv(from_double(1.0), x, from_double(0.0), y);
    
    std::cout << "A*x = [" << to_double(y[0]) << ", " << to_double(y[1]) 
              << "] (expected [3, 7])" << std::endl;
}

int main() {
    std::cout << "=== Testing QuadBLAS Bug Fixes ===\n";
    std::cout << quadblas_get_version() << std::endl;
    
    try {
        test_quadVector_fixed();
        test_dot_fixed();
        test_vector_class_fixed();
        test_simple_gemv();
        
        std::cout << "\n=== Fix verification completed! ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}