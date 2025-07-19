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

  inline void gemm_micro_kernel_scalar(size_t mr, size_t nr, size_t kc,
                                       Sleef_quad alpha,
                                       Sleef_quad *A_packed, Sleef_quad *B_packed,
                                       Sleef_quad beta, Sleef_quad *C, size_t ldc)
  {

    for (size_t i = 0; i < mr; ++i)
    {
      for (size_t j = 0; j < nr; ++j)
      {
        Sleef_quad sum = SLEEF_QUAD_C(0.0);

        for (size_t k = 0; k < kc; ++k)
        {
          Sleef_quad a_val = A_packed[i * kc + k];
          Sleef_quad b_val = B_packed[k * nr + j];
          sum = Sleef_fmaq1_u05(a_val, b_val, sum);
        }

        Sleef_quad c_old = C[i * ldc + j];
        C[i * ldc + j] = Sleef_fmaq1_u05(alpha, sum, Sleef_mulq1_u05(beta, c_old));
      }
    }
  }

  inline void gemm_micro_kernel_vectorized(size_t mr, size_t nr, size_t kc,
                                           Sleef_quad alpha,
                                           Sleef_quad *A_packed, Sleef_quad *B_packed,
                                           Sleef_quad beta, Sleef_quad *C, size_t ldc)
  {

    if (mr % VECTOR_SIZE != 0 || nr % VECTOR_SIZE != 0 || mr < VECTOR_SIZE || nr < VECTOR_SIZE)
    {
      gemm_micro_kernel_scalar(mr, nr, kc, alpha, A_packed, B_packed, beta, C, ldc);
      return;
    }

    const size_t mr_vec = mr / VECTOR_SIZE;
    const size_t nr_vec = nr / VECTOR_SIZE;

    Sleef_quad c_acc[mr][nr];

    for (size_t i = 0; i < mr; ++i)
    {
      for (size_t j = 0; j < nr; ++j)
      {
        c_acc[i][j] = SLEEF_QUAD_C(0.0);
      }
    }

    for (size_t i = 0; i < mr; ++i)
    {
      for (size_t j_vec = 0; j_vec < nr_vec; ++j_vec)
      {
        size_t j_start = j_vec * VECTOR_SIZE;
        QuadVector sum_vec(SLEEF_QUAD_C(0.0));

        for (size_t k = 0; k < kc; ++k)
        {
          Sleef_quad a_val = A_packed[i * kc + k];
          QuadVector a_vec(a_val);

          QuadVector b_vec = QuadVector::load(&B_packed[k * nr + j_start]);

          sum_vec = a_vec.fma(b_vec, sum_vec);
        }

        for (size_t lane = 0; lane < VECTOR_SIZE; ++lane)
        {
          c_acc[i][j_start + lane] = sum_vec.get(lane);
        }
      }

      for (size_t j = nr_vec * VECTOR_SIZE; j < nr; ++j)
      {
        for (size_t k = 0; k < kc; ++k)
        {
          c_acc[i][j] = Sleef_fmaq1_u05(A_packed[i * kc + k], B_packed[k * nr + j], c_acc[i][j]);
        }
      }
    }

    for (size_t i = 0; i < mr; ++i)
    {
      for (size_t j = 0; j < nr; ++j)
      {
        Sleef_quad c_old = C[i * ldc + j];
        C[i * ldc + j] = Sleef_fmaq1_u05(alpha, c_acc[i][j], Sleef_mulq1_u05(beta, c_old));
      }
    }
  }

  inline void gemm_micro_kernel(size_t mr, size_t nr, size_t kc,
                                Sleef_quad alpha,
                                Sleef_quad *A_packed, Sleef_quad *B_packed,
                                Sleef_quad beta, Sleef_quad *C, size_t ldc)
  {

    if (mr >= VECTOR_SIZE && nr >= VECTOR_SIZE && (mr * nr >= 8))
    {
      gemm_micro_kernel_vectorized(mr, nr, kc, alpha, A_packed, B_packed, beta, C, ldc);
    }
    else
    {
      gemm_micro_kernel_scalar(mr, nr, kc, alpha, A_packed, B_packed, beta, C, ldc);
    }
  }

  inline void gemm_micro_kernel_unpacked_fallback(size_t mr, size_t nr, size_t kc,
                                                  Sleef_quad alpha,
                                                  Sleef_quad *A_panel, Sleef_quad *B_panel, size_t ldb_panel,
                                                  Sleef_quad beta, Sleef_quad *C, size_t ldc)
  {
    for (size_t i = 0; i < mr; ++i)
    {
      for (size_t j = 0; j < nr; ++j)
      {
        Sleef_quad sum = SLEEF_QUAD_C(0.0);

        // Innermost loop to compute the dot product
        for (size_t k = 0; k < kc; ++k)
        {
          Sleef_quad a_val = A_panel[i * kc + k];
          // Correctly index into the non-contiguous B panel using its leading dimension
          Sleef_quad b_val = B_panel[k * ldb_panel + j];
          sum = Sleef_fmaq1_u05(a_val, b_val, sum);
        }

        // Apply alpha and beta to the final sum
        Sleef_quad c_old = C[i * ldc + j];
        C[i * ldc + j] = Sleef_fmaq1_u05(alpha, sum, Sleef_mulq1_u05(beta, c_old));
      }
    }
  }
  inline void gemm_macro_kernel(size_t mc, size_t nc, size_t kc,
                                Sleef_quad alpha,
                                Sleef_quad *A_packed, Sleef_quad *B_packed,
                                Sleef_quad beta, Sleef_quad *C, size_t ldc)
  {
    constexpr size_t MR = 4;
    constexpr size_t NR = 4;

    for (size_t i = 0; i < mc; i += MR)
    {
      size_t mr = std::min(MR, mc - i);

      for (size_t j = 0; j < nc; j += NR)
      {
        size_t nr = std::min(NR, nc - j);

        // --- FAST PATH ---
        // Try to allocate a small, contiguous buffer for the B sub-matrix to ensure
        // cache-friendly access in the highly optimized micro-kernel.
        Sleef_quad *B_sub = aligned_alloc<Sleef_quad>(kc * nr);
        if (B_sub)
        {
          // Repack the B sub-matrix into the contiguous buffer
          for (size_t k = 0; k < kc; ++k)
          {
            for (size_t jj = 0; jj < nr; ++jj)
            {
              B_sub[k * nr + jj] = B_packed[k * nc + (j + jj)];
            }
          }

          // Call the fast micro-kernel with the packed sub-matrix
          gemm_micro_kernel(mr, nr, kc, alpha,
                            &A_packed[i * kc], B_sub,
                            beta, &C[i * ldc + j], ldc);

          aligned_free(B_sub);
        }
        // --- FALLBACK PATH ---
        // If allocation fails (due to memory pressure), use the fallback kernel
        // that works directly on the non-contiguous B_packed panel. This will be
        // slower but guarantees a correct result, preventing crashes and NaNs.
        else
        {
          gemm_micro_kernel_unpacked_fallback(mr, nr, kc, alpha,
                                              &A_packed[i * kc], // Pointer to A panel
                                              &B_packed[j],      // Pointer to start column in B panel
                                              nc,                // Stride of the B panel
                                              beta, &C[i * ldc + j], ldc);
        }
      }
    }
  }

  inline void gemm_simple(Layout layout, size_t m, size_t n, size_t k,
                          Sleef_quad alpha,
                          Sleef_quad *A, size_t lda,
                          Sleef_quad *B, size_t ldb,
                          Sleef_quad beta, Sleef_quad *C, size_t ldc)
  {
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

  inline void gemm(Layout layout, size_t m, size_t n, size_t k,
                   Sleef_quad alpha,
                   Sleef_quad *A, size_t lda,
                   Sleef_quad *B, size_t ldb,
                   Sleef_quad beta, Sleef_quad *C, size_t ldc)
  {
    if (m == 0 || n == 0 || k == 0)
      return;

    constexpr size_t SMALL_MATRIX_THRESHOLD = 64;
    if (m <= SMALL_MATRIX_THRESHOLD && n <= SMALL_MATRIX_THRESHOLD && k <= SMALL_MATRIX_THRESHOLD)
    {
      gemm_simple(layout, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
      return;
    }

    BlockingParams params(m, n, k);

    Sleef_quad *A_packed = aligned_alloc<Sleef_quad>(params.mc * params.kc);
    Sleef_quad *B_packed = aligned_alloc<Sleef_quad>(params.kc * params.nc);

    if (!A_packed || !B_packed)
    {
      aligned_free(A_packed);
      aligned_free(B_packed);
      gemm_simple(layout, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
      return;
    }

    for (size_t kk = 0; kk < k; kk += params.kc)
    {
      size_t kc = std::min(params.kc, k - kk);

      for (size_t mm = 0; mm < m; mm += params.mc)
      {
        size_t mc = std::min(params.mc, m - mm);

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

          for (size_t i = 0; i < kc; ++i)
          {
            for (size_t j = 0; j < nc; ++j)
            {
              size_t src_idx = (layout == Layout::RowMajor) ? (kk + i) * ldb + (nn + j) : (nn + j) * ldb + (kk + i);
              B_packed[i * nc + j] = B[src_idx];
            }
          }

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

}

#endif