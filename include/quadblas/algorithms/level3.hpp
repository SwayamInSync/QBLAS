#ifndef QUADBLAS_ALGORITHMS_LEVEL3_HPP
#define QUADBLAS_ALGORITHMS_LEVEL3_HPP

#include "../core/constants.hpp"
#include "../core/types.hpp"
#include "../core/platform.hpp"
#include "../simd/quad_vector.hpp"
#include "../memory/allocation.hpp"
#include "../detail/blocking.hpp"
#include "../threading/openmp_utils.hpp"
#include <algorithm>

namespace QuadBLAS
{

  // Micro-kernel for small matrix blocks - highly optimized inner loop
  inline void gemm_micro_kernel(size_t mr, size_t nr, size_t kc,
                                Sleef_quad alpha,
                                Sleef_quad *A, Sleef_quad *B,
                                Sleef_quad beta, Sleef_quad *C, size_t ldc)
  {

    // Accumulate in registers
    QuadVector c_vec[mr / VECTOR_SIZE][nr / VECTOR_SIZE];

    // Initialize accumulators
    for (size_t i = 0; i < mr / VECTOR_SIZE; ++i)
    {
      for (size_t j = 0; j < nr / VECTOR_SIZE; ++j)
      {
        c_vec[i][j] = QuadVector(SLEEF_QUAD_C(0.0));
      }
    }

    // Main computation loop
    for (size_t k = 0; k < kc; ++k)
    {
      for (size_t i = 0; i < mr / VECTOR_SIZE; ++i)
      {
        QuadVector a_vec = QuadVector::load(&A[i * VECTOR_SIZE * kc + k * VECTOR_SIZE]);

        for (size_t j = 0; j < nr / VECTOR_SIZE; ++j)
        {
          QuadVector b_vec = QuadVector::load(&B[k * nr + j * VECTOR_SIZE]);
          c_vec[i][j] = a_vec.fma(b_vec, c_vec[i][j]); // a*b + c
        }
      }
    }

    // Store results back to C with alpha and beta scaling
    QuadVector alpha_vec(alpha);
    QuadVector beta_vec(beta);

    for (size_t i = 0; i < mr / VECTOR_SIZE; ++i)
    {
      for (size_t j = 0; j < nr / VECTOR_SIZE; ++j)
      {
        Sleef_quad *c_ptr = &C[i * VECTOR_SIZE * ldc + j * VECTOR_SIZE];
        QuadVector c_old = QuadVector::load(c_ptr);
        QuadVector c_new = (c_vec[i][j] * alpha_vec) + (c_old * beta_vec);
        c_new.store(c_ptr);
      }
    }
  }

  // Macro-kernel for medium-sized blocks
  inline void gemm_macro_kernel(size_t mc, size_t nc, size_t kc,
                                Sleef_quad alpha,
                                Sleef_quad *A, Sleef_quad *B,
                                Sleef_quad beta, Sleef_quad *C, size_t ldc)
  {

    constexpr size_t MR = 4; // Micro-panel height
    constexpr size_t NR = 4; // Micro-panel width

    for (size_t i = 0; i < mc; i += MR)
    {
      size_t mr = std::min(MR, mc - i);

      for (size_t j = 0; j < nc; j += NR)
      {
        size_t nr = std::min(NR, nc - j);

        gemm_micro_kernel(mr, nr, kc, alpha,
                          &A[i * kc], &B[j],
                          beta, &C[i * ldc + j], ldc);
      }
    }
  }

  // Simple GEMM implementation for small matrices
  inline void gemm_simple(Layout layout, size_t m, size_t n, size_t k,
                          Sleef_quad alpha,
                          Sleef_quad *A, size_t lda,
                          Sleef_quad *B, size_t ldb,
                          Sleef_quad beta, Sleef_quad *C, size_t ldc)
  {
    // Simple triple loop - guaranteed to work correctly
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (m * n >= PARALLEL_THRESHOLD)
#endif
    for (size_t i = 0; i < m; ++i)
    {
      for (size_t j = 0; j < n; ++j)
      {
        Sleef_quad sum = SLEEF_QUAD_C(0.0);

        for (size_t l = 0; l < k; ++l)
        {
          size_t a_idx = (layout == Layout::RowMajor) ? i * lda + l : l * lda + i;
          size_t b_idx = (layout == Layout::RowMajor) ? l * ldb + j : j * ldb + l;
          sum = Sleef_fmaq1_u05(A[a_idx], B[b_idx], sum);
        }

        size_t c_idx = (layout == Layout::RowMajor) ? i * ldc + j : j * ldc + i;
        C[c_idx] = Sleef_fmaq1_u05(alpha, sum, Sleef_mulq1_u05(beta, C[c_idx]));
      }
    }
  }

  // Main GEMM function: C = alpha * A * B + beta * C
  inline void gemm(Layout layout, size_t m, size_t n, size_t k,
                   Sleef_quad alpha,
                   Sleef_quad *A, size_t lda,
                   Sleef_quad *B, size_t ldb,
                   Sleef_quad beta, Sleef_quad *C, size_t ldc)
  {

    if (m == 0 || n == 0 || k == 0)
      return;

    // Use simple implementation for small matrices to avoid micro-kernel issues
    // The blocked algorithm is optimized for larger matrices
    constexpr size_t SMALL_MATRIX_THRESHOLD = 32;
    if (m <= SMALL_MATRIX_THRESHOLD && n <= SMALL_MATRIX_THRESHOLD && k <= SMALL_MATRIX_THRESHOLD)
    {
      gemm_simple(layout, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
      return;
    }

    BlockingParams params(m, n, k);

    // Allocate temporary packed matrices for better cache performance
    Sleef_quad *A_packed = aligned_alloc<Sleef_quad>(params.mc * params.kc);
    Sleef_quad *B_packed = aligned_alloc<Sleef_quad>(params.kc * params.nc);

    if (!A_packed || !B_packed)
    {
      // Fallback to simple implementation if allocation fails
      aligned_free(A_packed);
      aligned_free(B_packed);
      gemm_simple(layout, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
      return;
    }

    // Blocked GEMM implementation
    for (size_t kk = 0; kk < k; kk += params.kc)
    {
      size_t kc = std::min(params.kc, k - kk);

      for (size_t mm = 0; mm < m; mm += params.mc)
      {
        size_t mc = std::min(params.mc, m - mm);

        // Pack A panel
        for (size_t i = 0; i < mc; ++i)
        {
          for (size_t j = 0; j < kc; ++j)
          {
            size_t src_idx = (layout == Layout::RowMajor) ? (mm + i) * lda + (kk + j) : (kk + j) * lda + (mm + i);
            A_packed[i * kc + j] = A[src_idx];
          }
        }

        for (size_t nn = 0; nn < n; nn += params.nc)
        {
          size_t nc = std::min(params.nc, n - nn);

          // Pack B panel
          for (size_t i = 0; i < kc; ++i)
          {
            for (size_t j = 0; j < nc; ++j)
            {
              size_t src_idx = (layout == Layout::RowMajor) ? (kk + i) * ldb + (nn + j) : (nn + j) * ldb + (kk + i);
              B_packed[i * nc + j] = B[src_idx];
            }
          }

          // Compute C block
          Sleef_quad *C_block = &C[(layout == Layout::RowMajor) ? mm * ldc + nn : nn * ldc + mm];

          gemm_macro_kernel(mc, nc, kc, alpha,
                            A_packed, B_packed,
                            (kk == 0) ? beta : SLEEF_QUAD_C(1.0),
                            C_block, ldc);
        }
      }
    }

    aligned_free(A_packed);
    aligned_free(B_packed);
  }

} // namespace QuadBLAS

#endif // QUADBLAS_ALGORITHMS_LEVEL3_HPP