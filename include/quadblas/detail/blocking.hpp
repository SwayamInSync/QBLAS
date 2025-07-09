#ifndef QUADBLAS_DETAIL_BLOCKING_HPP
#define QUADBLAS_DETAIL_BLOCKING_HPP

#include "../core/constants.hpp"
#include "../core/platform.hpp"
#include <algorithm>

namespace QuadBLAS
{

  // Cache-friendly blocking parameters
  struct BlockingParams
  {
    size_t mc, kc, nc; // Panel sizes for GEMM

    BlockingParams(size_t m = 0, size_t n = 0, size_t k = 0)
    {
      // Calculate optimal blocking sizes based on cache hierarchy
      const size_t quad_size = sizeof(Sleef_quad);
      const size_t l1_quads = L1_CACHE_SIZE / (3 * quad_size);
      const size_t l2_quads = L2_CACHE_SIZE / quad_size;

      if (m * n * k == 0)
      {
        // Default blocking
        mc = std::min(static_cast<size_t>(GEMM_BLOCK_SIZE), l1_quads / 4);
        kc = std::min(static_cast<size_t>(GEMM_BLOCK_SIZE), l1_quads / 4);
        nc = std::min(static_cast<size_t>(GEMM_BLOCK_SIZE), l2_quads / (mc + kc));
      }
      else
      {
        // Adaptive blocking based on matrix sizes
        mc = std::min({m, static_cast<size_t>(GEMM_BLOCK_SIZE), l1_quads / 4});
        kc = std::min({k, static_cast<size_t>(GEMM_BLOCK_SIZE), l1_quads / 4});
        nc = std::min({n, static_cast<size_t>(GEMM_BLOCK_SIZE), l2_quads / (mc + kc)});
      }

      // Ensure sizes are multiples of vector size for better vectorization
      mc = (mc / VECTOR_SIZE) * VECTOR_SIZE;
      nc = (nc / VECTOR_SIZE) * VECTOR_SIZE;

      // Minimum sizes
      mc = std::max(mc, VECTOR_SIZE);
      kc = std::max(kc, static_cast<size_t>(4));
      nc = std::max(nc, VECTOR_SIZE);
    }
  };

} // namespace QuadBLAS

#endif // QUADBLAS_DETAIL_BLOCKING_HPP