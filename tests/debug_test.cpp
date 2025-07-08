#include "quadblas.hpp"
#include <iostream>
#include <iomanip>

// Helper functions
double to_double(Sleef_quad q) {
    return static_cast<double>(Sleef_cast_to_doubleq1(q));
}

Sleef_quad from_double(double d) {
    return Sleef_cast_from_doubleq1(d);
}

void test_sleef_basic() {
    std::cout << "\n=== Testing SLEEF Basic Operations ===\n";
    
    Sleef_quad a = from_double(2.0);
    Sleef_quad b = from_double(3.0);
    
    Sleef_quad sum = Sleef_addq1_u05(a, b);
    Sleef_quad prod = Sleef_mulq1_u05(a, b);
    Sleef_quad fma_result = Sleef_fmaq1_u05(a, b, from_double(1.0)); // a*b + 1
    
    std::cout << "a = " << to_double(a) << std::endl;
    std::cout << "b = " << to_double(b) << std::endl;
    std::cout << "a + b = " << to_double(sum) << " (expected 5)" << std::endl;
    std::cout << "a * b = " << to_double(prod) << " (expected 6)" << std::endl;
    std::cout << "a*b + 1 = " << to_double(fma_result) << " (expected 7)" << std::endl;
}

void test_simd_wrapper() {
    std::cout << "\n=== Testing SIMD Wrapper ===\n";
    
    QuadBLAS::QuadVector v1(from_double(2.0), from_double(3.0));
    QuadBLAS::QuadVector v2(from_double(4.0), from_double(5.0));
    
    std::cout << "v1: [" << to_double(v1.get(0)) << ", " << to_double(v1.get(1)) << "]" << std::endl;
    std::cout << "v2: [" << to_double(v2.get(0)) << ", " << to_double(v2.get(1)) << "]" << std::endl;
    
    QuadBLAS::QuadVector sum = v1 + v2;
    QuadBLAS::QuadVector prod = v1 * v2;
    
    std::cout << "v1 + v2: [" << to_double(sum.get(0)) << ", " << to_double(sum.get(1)) << "]" << std::endl;
    std::cout << "v1 * v2: [" << to_double(prod.get(0)) << ", " << to_double(prod.get(1)) << "]" << std::endl;
    
    Sleef_quad h_sum = sum.horizontal_sum();
    std::cout << "Horizontal sum of v1+v2: " << to_double(h_sum) << " (expected 14)" << std::endl;
}

void test_manual_dot() {
    std::cout << "\n=== Testing Manual Dot Product ===\n";
    
    const size_t n = 5;
    Sleef_quad x[5] = {from_double(1), from_double(2), from_double(3), from_double(4), from_double(5)};
    Sleef_quad y[5] = {from_double(1), from_double(1), from_double(1), from_double(1), from_double(1)};
    
    std::cout << "x: [";
    for (size_t i = 0; i < n; ++i) {
        std::cout << to_double(x[i]);
        if (i < n-1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "y: [";
    for (size_t i = 0; i < n; ++i) {
        std::cout << to_double(y[i]);
        if (i < n-1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Manual scalar dot product
    Sleef_quad manual_result = from_double(0.0);
    for (size_t i = 0; i < n; ++i) {
        manual_result = Sleef_fmaq1_u05(x[i], y[i], manual_result);
    }
    std::cout << "Manual scalar dot: " << to_double(manual_result) << " (expected 15)" << std::endl;
    
    // Test vectorized kernel
    Sleef_quad vectorized_result = QuadBLAS::dot_kernel_vectorized(x, y, n);
    std::cout << "Vectorized dot: " << to_double(vectorized_result) << " (expected 15)" << std::endl;
    
    // Test full dot function
    Sleef_quad full_result = QuadBLAS::dot(n, x, 1, y, 1);
    std::cout << "Full dot function: " << to_double(full_result) << " (expected 15)" << std::endl;
}

void test_vector_class() {
    std::cout << "\n=== Testing Vector Class ===\n";
    
    const size_t n = 5;
    QuadBLAS::DefaultVector<> x(n), y(n);
    
    std::cout << "Initializing vectors..." << std::endl;
    for (size_t i = 0; i < n; ++i) {
        x[i] = from_double(i + 1);  // 1, 2, 3, 4, 5
        y[i] = from_double(1.0);    // 1, 1, 1, 1, 1
    }
    
    std::cout << "x: [";
    for (size_t i = 0; i < n; ++i) {
        std::cout << to_double(x[i]);
        if (i < n-1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "y: [";
    for (size_t i = 0; i < n; ++i) {
        std::cout << to_double(y[i]);
        if (i < n-1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    Sleef_quad result = x.dot(y);
    std::cout << "Vector class dot: " << to_double(result) << " (expected 15)" << std::endl;
}

void test_memory_alignment() {
    std::cout << "\n=== Testing Memory Alignment ===\n";
    
    Sleef_quad* aligned_ptr = QuadBLAS::aligned_alloc<Sleef_quad>(10);
    if (aligned_ptr) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(aligned_ptr);
        std::cout << "Aligned pointer: " << std::hex << addr << std::dec << std::endl;
        std::cout << "Alignment check: " << (addr % QuadBLAS::ALIGNMENT == 0 ? "PASS" : "FAIL") << std::endl;
        
        // Test basic operations with aligned memory
        for (int i = 0; i < 5; ++i) {
            aligned_ptr[i] = from_double(i + 1);
        }
        
        std::cout << "Aligned array: [";
        for (int i = 0; i < 5; ++i) {
            std::cout << to_double(aligned_ptr[i]);
            if (i < 4) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        QuadBLAS::aligned_free(aligned_ptr);
    } else {
        std::cout << "ERROR: Could not allocate aligned memory!" << std::endl;
    }
}

void test_load_store() {
    std::cout << "\n=== Testing SIMD Load/Store ===\n";
    
    Sleef_quad data[4] = {from_double(1.0), from_double(2.0), from_double(3.0), from_double(4.0)};
    
    std::cout << "Original data: [" << to_double(data[0]) << ", " << to_double(data[1]) 
              << ", " << to_double(data[2]) << ", " << to_double(data[3]) << "]" << std::endl;
    
    // Test loading first pair
    QuadBLAS::QuadVector vec1 = QuadBLAS::QuadVector::load(&data[0]);
    std::cout << "Loaded vec1: [" << to_double(vec1.get(0)) << ", " << to_double(vec1.get(1)) << "]" << std::endl;
    
    // Test loading second pair
    QuadBLAS::QuadVector vec2 = QuadBLAS::QuadVector::load(&data[2]);
    std::cout << "Loaded vec2: [" << to_double(vec2.get(0)) << ", " << to_double(vec2.get(1)) << "]" << std::endl;
    
    // Test store
    Sleef_quad result[2];
    vec1.store(result);
    std::cout << "Stored vec1: [" << to_double(result[0]) << ", " << to_double(result[1]) << "]" << std::endl;
}

int main() {
    std::cout << "=== QuadBLAS Debug Tests ===\n";
    std::cout << quadblas_get_version() << std::endl;
    
#ifdef QUADBLAS_X86_64
    std::cout << "Platform: x86-64" << std::endl;
#else
    std::cout << "Platform: Generic" << std::endl;
#endif
    
    try {
        test_sleef_basic();
        test_simd_wrapper();
        test_memory_alignment();
        test_load_store();
        test_manual_dot();
        test_vector_class();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}