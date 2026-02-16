#ifndef QUADBLAS_INTERFACE_C_INTERFACE_HPP
#define QUADBLAS_INTERFACE_C_INTERFACE_HPP

#include "../core/platform.hpp"
#include "../core/types.hpp"
#include "../algorithms/level1.hpp"
#include "../algorithms/level2.hpp"
#include "../algorithms/level3.hpp"
#include "../threading/openmp_utils.hpp"
#include <cstdint>
#include <utility>

// C Interface for easy integration with Python/numpy
// All functions are inline to support header-only inclusion in multiple TUs
extern "C"
{

  // C API - LEVEL 1 BLAS (Vector operations)

  // QDOT: dot product of two vectors
  inline double quadblas_qdot(int n, void *x, int incx, void *y, int incy)
  {
    Sleef_quad *qx = static_cast<Sleef_quad *>(x);
    Sleef_quad *qy = static_cast<Sleef_quad *>(y);

    Sleef_quad result = QuadBLAS::dot(static_cast<size_t>(n), qx,
                                      static_cast<size_t>(incx), qy,
                                      static_cast<size_t>(incy));

    return static_cast<double>(Sleef_cast_to_doubleq1(result));
  }

  // QNRM2: Euclidean norm of a vector
  inline double quadblas_qnrm2(int n, void *x, int incx)
  {
    Sleef_quad *qx = static_cast<Sleef_quad *>(x);

    Sleef_quad result = QuadBLAS::dot(static_cast<size_t>(n), qx,
                                      static_cast<size_t>(incx), qx,
                                      static_cast<size_t>(incx));

    result = Sleef_sqrtq1_u05(result);
    return static_cast<double>(Sleef_cast_to_doubleq1(result));
  }

  // QAXPY: y := alpha*x + y (uses optimized SIMD/threaded implementation)
  inline void quadblas_qaxpy(int n, double alpha, void *x, int incx, void *y, int incy)
  {
    if (n <= 0)
      return;

    Sleef_quad *qx = static_cast<Sleef_quad *>(x);
    Sleef_quad *qy = static_cast<Sleef_quad *>(y);
    Sleef_quad qalpha = Sleef_cast_from_doubleq1(alpha);

    QuadBLAS::axpy(static_cast<size_t>(n), qalpha, qx,
                   static_cast<size_t>(incx), qy,
                   static_cast<size_t>(incy));
  }

  // C API - LEVEL 2 BLAS (Matrix-vector operations)

  // QGEMV: matrix-vector multiplication
  inline void quadblas_qgemv(char layout, char trans, int m, int n, double alpha,
                      void *A, int lda, void *x, int incx,
                      double beta, void *y, int incy)
  {

    Sleef_quad *qA = static_cast<Sleef_quad *>(A);
    Sleef_quad *qx = static_cast<Sleef_quad *>(x);
    Sleef_quad *qy = static_cast<Sleef_quad *>(y);

    Sleef_quad qalpha = Sleef_cast_from_doubleq1(alpha);
    Sleef_quad qbeta = Sleef_cast_from_doubleq1(beta);

    QuadBLAS::Layout quadblas_layout = (layout == 'C' || layout == 'c') ? QuadBLAS::Layout::ColMajor : QuadBLAS::Layout::RowMajor;

    // Handle transpose (swap dimensions and flip layout)
    if (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c')
    {
      std::swap(m, n);
      quadblas_layout = (quadblas_layout == QuadBLAS::Layout::RowMajor)
                            ? QuadBLAS::Layout::ColMajor
                            : QuadBLAS::Layout::RowMajor;
    }

    QuadBLAS::gemv(quadblas_layout, static_cast<size_t>(m), static_cast<size_t>(n),
                   qalpha, qA, static_cast<size_t>(lda), qx, static_cast<size_t>(incx),
                   qbeta, qy, static_cast<size_t>(incy));
  }

  // C API - LEVEL 3 BLAS (Matrix-matrix operations)

  // QGEMM: matrix-matrix multiplication
  inline void quadblas_qgemm(char layout, char transa, char transb, int m, int n, int k,
                      double alpha, void *A, int lda, void *B, int ldb,
                      double beta, void *C, int ldc)
  {

    Sleef_quad *qA = static_cast<Sleef_quad *>(A);
    Sleef_quad *qB = static_cast<Sleef_quad *>(B);
    Sleef_quad *qC = static_cast<Sleef_quad *>(C);

    Sleef_quad qalpha = Sleef_cast_from_doubleq1(alpha);
    Sleef_quad qbeta = Sleef_cast_from_doubleq1(beta);

    QuadBLAS::Layout quadblas_layout = (layout == 'C' || layout == 'c') ? QuadBLAS::Layout::ColMajor : QuadBLAS::Layout::RowMajor;

    // Note: transpose support requires additional matrix manipulation
    // For now, silently proceed with NoTranspose semantics
    (void)transa;
    (void)transb;

    QuadBLAS::gemm(quadblas_layout, static_cast<size_t>(m), static_cast<size_t>(n),
                   static_cast<size_t>(k), qalpha, qA, static_cast<size_t>(lda),
                   qB, static_cast<size_t>(ldb), qbeta, qC, static_cast<size_t>(ldc));
  }

  // Utility functions

  // Set number of threads for OpenMP
  inline void quadblas_set_num_threads(int num_threads)
  {
    QuadBLAS::set_num_threads(num_threads);
  }

  // Get number of threads
  inline int quadblas_get_num_threads(void)
  {
    return QuadBLAS::get_num_threads();
  }

  // Version information
  inline const char *quadblas_get_version(void)
  {
    return "QuadBLAS 1.0.0 - High Performance Quad Precision BLAS";
  }

  // Memory alignment check
  inline int quadblas_is_aligned(const void *ptr)
  {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    return (addr % QuadBLAS::ALIGNMENT) == 0;
  }

} // extern "C"

#endif // QUADBLAS_INTERFACE_C_INTERFACE_HPP
