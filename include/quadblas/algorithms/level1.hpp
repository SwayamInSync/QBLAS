#ifndef QUADBLAS_ALGORITHMS_LEVEL1_HPP
#define QUADBLAS_ALGORITHMS_LEVEL1_HPP

#include "../core/constants.hpp"
#include "../core/platform.hpp"
#include "../simd/quad_vector.hpp"
#include "../threading/openmp_utils.hpp"
#include <vector>

namespace QuadBLAS
{

  // Vectorized dot product kernel for aligned data
  inline Sleef_quad dot_kernel_vectorized(Sleef_quad *x, Sleef_quad *y, size_t n)
  {
    const size_t vec_n = n / VECTOR_SIZE;
    const size_t remainder = n % VECTOR_SIZE;

    QuadVector sum_vec(SLEEF_QUAD_C(0.0));

    // Vectorized loop
    for (size_t i = 0; i < vec_n; ++i)
    {
      QuadVector x_vec = QuadVector::load(&x[i * VECTOR_SIZE]);
      QuadVector y_vec = QuadVector::load(&y[i * VECTOR_SIZE]);
      sum_vec = x_vec.fma(y_vec, sum_vec); // x*y + sum
    }

    Sleef_quad result = sum_vec.horizontal_sum();

    // Handle remainder
    for (size_t i = vec_n * VECTOR_SIZE; i < n; ++i)
    {
      result = Sleef_fmaq1_u05(x[i], y[i], result);
    }

    return result;
  }

  // Parallel dot product for large vectors
  inline Sleef_quad dot_parallel(Sleef_quad *x, Sleef_quad *y, size_t n)
  {
    if (n < PARALLEL_THRESHOLD)
    {
      return dot_kernel_vectorized(x, y, n);
    }

#ifdef _OPENMP
    const int num_threads = get_num_threads();
    const size_t chunk_size = n / num_threads;

    // Use array for partial results since OpenMP reduction doesn't work with struct types
    std::vector<Sleef_quad> partial_results(num_threads);
    for (int i = 0; i < num_threads; ++i)
    {
      partial_results[i] = SLEEF_QUAD_C(0.0);
    }

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      size_t start = tid * chunk_size;
      size_t end = (tid == num_threads - 1) ? n : start + chunk_size;

      if (start < end)
      {
        partial_results[tid] = dot_kernel_vectorized(&x[start], &y[start], end - start);
      }
    }

    // Combine partial results
    Sleef_quad result = SLEEF_QUAD_C(0.0);
    for (int i = 0; i < num_threads; ++i)
    {
      result = Sleef_addq1_u05(result, partial_results[i]);
    }

    return result;
#else
    return dot_kernel_vectorized(x, y, n);
#endif
  }

  // Main dot product function with stride support
  inline Sleef_quad dot(size_t n, Sleef_quad *x, size_t incx,
                        Sleef_quad *y, size_t incy)
  {
    if (n == 0)
      return SLEEF_QUAD_C(0.0);

    // Fast path for unit strides
    if (incx == 1 && incy == 1)
    {
      return dot_parallel(x, y, n);
    }

    // Strided access
    Sleef_quad result = SLEEF_QUAD_C(0.0);

    if (n >= PARALLEL_THRESHOLD)
    {
#ifdef _OPENMP
      const int num_threads = get_num_threads();

      // Use array for partial results
      std::vector<Sleef_quad> partial_results(num_threads);
      for (int i = 0; i < num_threads; ++i)
      {
        partial_results[i] = SLEEF_QUAD_C(0.0);
      }

#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        size_t chunk_size = n / num_threads;
        size_t start = tid * chunk_size;
        size_t end = (tid == num_threads - 1) ? n : start + chunk_size;

        for (size_t i = start; i < end; ++i)
        {
          partial_results[tid] = Sleef_fmaq1_u05(x[i * incx], y[i * incy], partial_results[tid]);
        }
      }

      // Combine partial results
      for (int i = 0; i < num_threads; ++i)
      {
        result = Sleef_addq1_u05(result, partial_results[i]);
      }
#else
      for (size_t i = 0; i < n; ++i)
      {
        result = Sleef_fmaq1_u05(x[i * incx], y[i * incy], result);
      }
#endif
    }
    else
    {
      for (size_t i = 0; i < n; ++i)
      {
        result = Sleef_fmaq1_u05(x[i * incx], y[i * incy], result);
      }
    }

    return result;
  }

} // namespace QuadBLAS

#endif // QUADBLAS_ALGORITHMS_LEVEL1_HPP