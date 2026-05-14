/* Level 2: matrix-vector operations.  `lda` is the leading dimension of
 * A in its natural orientation (stride between rows for row-major,
 * between columns for col-major). */
#ifndef QBLAS_LEVEL2_H
#define QBLAS_LEVEL2_H

#ifndef QBLAS_H
#  include "qblas.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* y := alpha * op(A) * x + beta * y */
QBLAS_API void cblas_qgemv(QBLAS_LAYOUT layout, QBLAS_TRANSPOSE trans,
                           int m, int n,
                           Sleef_quad alpha,
                           const Sleef_quad *A, int lda,
                           const Sleef_quad *x, int incx,
                           Sleef_quad beta,
                           Sleef_quad *y, int incy);

/* A := alpha * x * y^T + A */
QBLAS_API void cblas_qger(QBLAS_LAYOUT layout,
                          int m, int n,
                          Sleef_quad alpha,
                          const Sleef_quad *x, int incx,
                          const Sleef_quad *y, int incy,
                          Sleef_quad *A, int lda);

/* y := alpha * A * x + beta * y, A symmetric; only `uplo` triangle is read. */
QBLAS_API void cblas_qsymv(QBLAS_LAYOUT layout, QBLAS_UPLO uplo,
                           int n,
                           Sleef_quad alpha,
                           const Sleef_quad *A, int lda,
                           const Sleef_quad *x, int incx,
                           Sleef_quad beta,
                           Sleef_quad *y, int incy);

/* Solve op(A) * x = b in place, A triangular. */
QBLAS_API void cblas_qtrsv(QBLAS_LAYOUT layout,
                           QBLAS_UPLO uplo,
                           QBLAS_TRANSPOSE trans,
                           QBLAS_DIAG diag,
                           int n,
                           const Sleef_quad *A, int lda,
                           Sleef_quad *x, int incx);

#ifdef __cplusplus
}
#endif

#endif
