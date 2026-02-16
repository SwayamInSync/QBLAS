#ifndef QUADBLAS_DETAIL_BLOCKING_HPP
#define QUADBLAS_DETAIL_BLOCKING_HPP

#include "../core/constants.hpp"
#include "../core/platform.hpp"
#include <algorithm>

namespace QuadBLAS
{

  // Cache-friendly blocking parameters for GEMM
  //
  // Memory hierarchy usage (standard BLAS approach):
  //   L1: Micro-kernel slices: A(MR x kc) + B(kc x NR) + C(MR x NR)
  //   L2: A panel (mc x kc) stays resident
  //   L3/Memory: B panel (kc x nc) streams through
  struct BlockingParams
  {
    size_t mc, kc, nc; // Panel sizes for GEMM

    BlockingParams(size_t m = 0, size_t n = 0, size_t k = 0)
    {
      constexpr size_t MR = 4; // Must match GEMM_MR
      constexpr size_t NR = 4; // Must match GEMM_NR
      const size_t elem_size = sizeof(Sleef_quad);

      // Determine cache sizes (detect at compile time for known platforms)
      size_t l1_size = L1_CACHE_SIZE;
      size_t l2_size = L2_CACHE_SIZE;

#if defined(__APPLE__) && defined(QUADBLAS_AARCH64)
      // Apple Silicon has larger caches
      l1_size = 131072;  // 128KB L1D per core
      l2_size = 4194304; // 4MB L2 per cluster
#endif

      // kc: chosen so micro-kernel data fits in L1
      // Need: (MR * kc + kc * NR + MR * NR) * elem_size <= L1 / 2
      size_t l1_budget = l1_size / (2 * elem_size);
      size_t max_kc = (l1_budget - MR * NR) / (MR + NR);
      kc = std::min(max_kc, static_cast<size_t>(256));

      // mc: A panel (mc x kc) fits in L2 / 2
      size_t l2_budget = l2_size / (2 * elem_size);
      size_t max_mc = l2_budget / kc;
      mc = std::min(max_mc, static_cast<size_t>(256));

      // nc: B panel (kc x nc) can be larger - fits in remaining L2 or streams from L3
      size_t remaining_l2 = l2_budget - mc * kc;
      size_t max_nc = remaining_l2 / kc;
      nc = std::min(max_nc, static_cast<size_t>(512));

      // Clamp to actual matrix dimensions
      if (m > 0) mc = std::min(mc, m);
      if (k > 0) kc = std::min(kc, k);
      if (n > 0) nc = std::min(nc, n);

      // Align to micro-kernel tile sizes for clean tiling
      mc = (mc / MR) * MR;
      nc = (nc / NR) * NR;

      // Ensure minimum sizes
      mc = std::max(mc, MR);
      kc = std::max(kc, static_cast<size_t>(4));
      nc = std::max(nc, NR);
    }
  };

} // namespace QuadBLAS

#endif // QUADBLAS_DETAIL_BLOCKING_HPP
