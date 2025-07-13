# QuadBLAS (QBLAS)

QuadBLAS is a high-performance linear algebra library implementing BLAS-compliant routines for IEEE 754 quadruple precision (binary128) floating-point arithmetic. Built as a header-only templated library on top of the SLEEF vectorized mathematical library, QuadBLAS provides optimized implementations of fundamental linear algebra operations with significant performance improvements over naive implementations.



## Technical Features

- **Quadruple Precision Arithmetic**: Full IEEE 754 binary128 (128-bit) floating-point support
- **SIMD Vectorization**: Platform-optimized implementations for x86-64 SSE/AVX and ARM64 NEON instruction sets
- **Parallel Execution**: OpenMP-based multi-threading with configurable parallelization thresholds
- **Memory Hierarchy Optimization**: Multi-level cache blocking algorithms for optimal data locality
- **Header-Only Design**: Template-based implementation enabling compile-time optimizations
- **BLAS-Compliant Interface**: Standard Level 1, 2, and 3 BLAS routine signatures
- **Cross-Platform Compatibility**: Support for Windows, Linux, and macOS operating systems
- **Dual API Design**: Both C and C++ interfaces for integration flexibility

## Performance Characteristics

QuadBLAS demonstrates substantial performance improvements over naive implementations through algorithmic optimizations and parallel execution:

| Operation Category | Problem Scale | Performance Improvement | Throughput |
|-------------------|---------------|------------------------|------------|
| Level 1 BLAS (DDOT) | 10^5 elements | 21× over serial implementation | Vectorized execution |
| Level 2 BLAS (DGEMV) | 1500×1500 matrix | 75× over serial implementation | 1.6 GFLOPS sustained |
| Level 3 BLAS (DGEMM) | 1000×1000 matrices | 2.8× over serial implementation | 0.06 GFLOPS sustained |

*Performance measurements conducted on multi-core x86-64 architecture with OpenMP threading enabled.*

## Prerequisites

- ISO C++17 compliant compiler (GCC ≥7.0, Clang ≥5.0, MSVC ≥2017)
- CMake build system (≥3.15)
- [SLEEF vectorized mathematical library](https://sleef.org/) (≥3.6)
- OpenMP runtime (optional, required for multi-threading)

## Installation

For systems with SLEEF built from source or non-standard installation paths:

```bash
# Configure SLEEF library path
export SLEEF_ROOT=/path/to/sleef/installation

# Build QuadBLAS
git clone https://github.com/your-org/QuadBLAS.git
cd QuadBLAS
mkdir build && cd build
cmake ..
make -j$(nproc)

# Execute test suite
./quadblas_test
./quadblas_benchmark
```

## Usage

### Header-Only Integration

For direct integration into existing projects:

```cpp
#include "include/quadblas/quadblas.hpp"

// Link against SLEEF libraries during compilation
// g++ -O3 -fopenmp source.cpp -lsleef -lsleefquad
```

### C++ Template Interface

```cpp
#include "include/quadblas/quadblas.hpp"

int main() {
    using namespace QuadBLAS;
    
    // Instantiate quadruple precision containers
    const size_t n = 1000;
    DefaultVector<> x(n), y(n);
    DefaultMatrix<> A(n, n);
    
    // Initialize with quadruple precision values
    for (size_t i = 0; i < n; ++i) {
        x[i] = Sleef_cast_from_doubleq1(static_cast<double>(i + 1));
        y[i] = Sleef_cast_from_doubleq1(2.0);
        for (size_t j = 0; j < n; ++j) {
            A(i, j) = Sleef_cast_from_doubleq1((i == j) ? 2.0 : 0.1);
        }
    }
    
    // Execute optimized linear algebra operations
    Sleef_quad dot_result = x.dot(y);                    // Level 1 BLAS: DDOT
    Sleef_quad norm_result = x.norm();                   // Level 1 BLAS: DNRM2
    A.gemv(SLEEF_QUAD_C(1.0), x, SLEEF_QUAD_C(0.0), y); // Level 2 BLAS: DGEMV
    
    return 0;
}
```

### C Interface for Language Interoperability

```c
#include "include/quadblas/quadblas.hpp"

// Level 1 BLAS: Dot product computation
double result = quadblas_qdot(n, x_ptr, 1, y_ptr, 1);

// Level 2 BLAS: Matrix-vector multiplication
// y := alpha*A*x + beta*y
quadblas_qgemv('R', 'N', m, n, 1.0, A_ptr, lda, x_ptr, 1, 0.0, y_ptr, 1);

// Level 3 BLAS: Matrix-matrix multiplication  
// C := alpha*A*B + beta*C
quadblas_qgemm('R', 'N', 'N', m, n, k, 1.0, A_ptr, lda, B_ptr, ldb, 0.0, C_ptr, ldc);

// Runtime configuration
quadblas_set_num_threads(16);
int active_threads = quadblas_get_num_threads();
```

### Advanced Configuration

```cpp
// Thread pool configuration for OpenMP execution
QuadBLAS::set_num_threads(16);

// Memory layout specification
QuadBLAS::MatrixRowMajor A_c_style(m, n);     // Row-major storage (C convention)
QuadBLAS::MatrixColMajor A_fortran_style(m, n); // Column-major storage (Fortran convention)

// Manual memory management with SIMD alignment
Sleef_quad* aligned_buffer = QuadBLAS::aligned_alloc<Sleef_quad>(1000);
QuadBLAS::Vector<QuadBLAS::Layout::RowMajor> custom_vector(aligned_buffer, 1000);
// ... computational work ...
QuadBLAS::aligned_free(aligned_buffer);
```

### Numerical Precision Demonstration

```cpp
// Demonstrate quadruple precision numerical stability
QuadBLAS::DefaultVector<> x(3), y(3);

// Configure test case prone to catastrophic cancellation in double precision
x[0] = Sleef_cast_from_doubleq1(1e20);   // Large positive value
x[1] = Sleef_cast_from_doubleq1(1.0);    // Unit value  
x[2] = Sleef_cast_from_doubleq1(-1e20);  // Large negative value

y[0] = y[1] = y[2] = Sleef_cast_from_doubleq1(1.0);

// Comparison of arithmetic precision
double double_precision_result = 1e20 * 1.0 + 1.0 * 1.0 + (-1e20) * 1.0;  // → 0.0 (precision loss)
Sleef_quad quad_precision_result = x.dot(y);  // → 1.0 (mathematically correct)

std::cout << "Double precision computation: " << double_precision_result << std::endl;
std::cout << "Quadruple precision computation: " << Sleef_cast_to_doubleq1(quad_precision_result) << std::endl;
```

## Contributing

We welcome contributions to QuadBLAS development. Please follow established contribution guidelines:

### Development Process

1. Fork the repository and create a feature branch
2. Implement changes following existing code style conventions
3. Add comprehensive tests for new functionality
4. Update documentation and performance benchmarks as appropriate
5. Submit pull requests with detailed change descriptions

### Contribution Areas

- Algorithm optimization and new BLAS routine implementations
- Platform-specific optimizations and hardware support
- Performance analysis and benchmarking improvements
- Documentation enhancement and example development
- Integration support for additional programming languages

## Citation
```bibtex
@software{quadblas2025,
  title = {QuadBLAS: High-Performance Linear Algebra for Quadruple Precision Computing},
  author = {Swayam Singh},
  year = {2025},
  url = {https://github.com/SwayamInSync/QBLAS},
  version = {1.0.0},
  note = {A header-only BLAS library for cross platform IEEE 754 quadruple precision arithmetic}
}
```

## Acknowledgments
```
@software{sleef,
  title = {SLEEF: A Portable Vectorized Library of Elementary Functions},
  author = {Naoki Shibata and contributors},
  url = {https://github.com/shibatch/sleef},
  note = {SIMD Library for Evaluating Elementary Functions}
}
```
