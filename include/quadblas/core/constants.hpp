#ifndef QUADBLAS_CORE_CONSTANTS_HPP
#define QUADBLAS_CORE_CONSTANTS_HPP

#include <cstddef>

namespace QuadBLAS
{

  // Configuration constants
  constexpr size_t VECTOR_SIZE = 2; // Sleef quad vector size
  constexpr size_t CACHE_LINE_SIZE = 64;
  constexpr size_t L1_CACHE_SIZE = 32768;
  constexpr size_t L2_CACHE_SIZE = 262144;
  constexpr size_t PARALLEL_THRESHOLD = 500;
  constexpr size_t GEMM_BLOCK_SIZE = 64;

  // Memory alignment for SIMD operations
  constexpr size_t ALIGNMENT = 32;

} // namespace QuadBLAS

#endif // QUADBLAS_CORE_CONSTANTS_HPP