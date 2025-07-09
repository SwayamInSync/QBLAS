#ifndef QUADBLAS_ALGORITHMS_LEVEL2_HPP
#define QUADBLAS_ALGORITHMS_LEVEL2_HPP

#include "../core/constants.hpp"
#include "../core/types.hpp"
#include "../core/platform.hpp"
#include "../threading/openmp_utils.hpp"
#include "level1.hpp"

namespace QuadBLAS
{

  // GEMV: y = alpha * A * x + beta * y
  // Row-major implementation
  inline void gemv_row_major(size_t m, size_t n, Sleef_quad alpha,
                             Sleef_quad *A, size_t lda,
                             Sleef_quad *x, size_t incx,
                             Sleef_quad beta, Sleef_quad *y, size_t incy)
  {

    if (m == 0 || n == 0)
      return;

    const bool use_parallel = m >= PARALLEL_THRESHOLD;

#ifdef _OPENMP
#pragma omp parallel for if (use_parallel)
#endif
    for (size_t i = 0; i < m; ++i)
    {
      // Compute dot product of row i with vector x
      Sleef_quad sum = SLEEF_QUAD_C(0.0);

      Sleef_quad *row = &A[i * lda];

      // Vectorized inner loop
      if (incx == 1)
      {
        sum = dot_kernel_vectorized(row, x, n);
      }
      else
      {
        // Strided access
        for (size_t j = 0; j < n; ++j)
        {
          sum = Sleef_fmaq1_u05(row[j], x[j * incx], sum);
        }
      }

      // y[i] = alpha * sum + beta * y[i]
      size_t y_idx = i * incy;
      y[y_idx] = Sleef_fmaq1_u05(alpha, sum, Sleef_mulq1_u05(beta, y[y_idx]));
    }
  }

  // Column-major implementation
  inline void gemv_col_major(size_t m, size_t n, Sleef_quad alpha,
                             const Sleef_quad *A, size_t lda,
                             const Sleef_quad *x, size_t incx,
                             Sleef_quad beta, Sleef_quad *y, size_t incy)
  {

    if (m == 0 || n == 0)
      return;

    // Scale y by beta first
    for (size_t i = 0; i < m; ++i)
    {
      y[i * incy] = Sleef_mulq1_u05(beta, y[i * incy]);
    }

    // Add alpha * A * x
    for (size_t j = 0; j < n; ++j)
    {
      Sleef_quad x_j = Sleef_mulq1_u05(alpha, x[j * incx]);
      const Sleef_quad *col = &A[j * lda];

#ifdef _OPENMP
#pragma omp parallel for if (m >= PARALLEL_THRESHOLD)
#endif
      for (size_t i = 0; i < m; ++i)
      {
        y[i * incy] = Sleef_fmaq1_u05(col[i], x_j, y[i * incy]);
      }
    }
  }

  // Main GEMV function
  inline void gemv(Layout layout, size_t m, size_t n, Sleef_quad alpha,
                   Sleef_quad *A, size_t lda,
                   Sleef_quad *x, size_t incx,
                   Sleef_quad beta, Sleef_quad *y, size_t incy)
  {

    if (layout == Layout::RowMajor)
    {
      gemv_row_major(m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
    else
    {
      gemv_col_major(m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
  }

} // namespace QuadBLAS

#endif // QUADBLAS_ALGORITHMS_LEVEL2_HPP