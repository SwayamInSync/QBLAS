/* QBLAS Level 3 — matrix-matrix operations. */
#ifndef QBLAS_LEVEL3_H
#define QBLAS_LEVEL3_H

#ifndef QBLAS_H
#  include "qblas.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* gemm: C := alpha * op(A) * op(B) + beta * C
 *   op(A) is m x k, op(B) is k x n, C is m x n. */
QBLAS_API void cblas_qgemm(QBLAS_LAYOUT layout,
                           QBLAS_TRANSPOSE transa, QBLAS_TRANSPOSE transb,
                           int m, int n, int k,
                           Sleef_quad alpha,
                           const Sleef_quad *A, int lda,
                           const Sleef_quad *B, int ldb,
                           Sleef_quad beta,
                           Sleef_quad *C, int ldc);

/* syrk: C := alpha * op(A) * op(A)^T + beta * C
 *   C is n x n symmetric; only the `uplo` triangle is updated. */
QBLAS_API void cblas_qsyrk(QBLAS_LAYOUT layout,
                           QBLAS_UPLO uplo,
                           QBLAS_TRANSPOSE trans,
                           int n, int k,
                           Sleef_quad alpha,
                           const Sleef_quad *A, int lda,
                           Sleef_quad beta,
                           Sleef_quad *C, int ldc);

/* trmm: B := alpha * op(A) * B   (side==Left)
 *       B := alpha * B * op(A)   (side==Right) */
QBLAS_API void cblas_qtrmm(QBLAS_LAYOUT layout,
                           QBLAS_SIDE side, QBLAS_UPLO uplo,
                           QBLAS_TRANSPOSE transa, QBLAS_DIAG diag,
                           int m, int n,
                           Sleef_quad alpha,
                           const Sleef_quad *A, int lda,
                           Sleef_quad *B, int ldb);

/* trsm: solve op(A) * X = alpha * B   (side==Left)
 *        solve X * op(A) = alpha * B   (side==Right)
 *  in-place; B holds X on return. */
QBLAS_API void cblas_qtrsm(QBLAS_LAYOUT layout,
                           QBLAS_SIDE side, QBLAS_UPLO uplo,
                           QBLAS_TRANSPOSE transa, QBLAS_DIAG diag,
                           int m, int n,
                           Sleef_quad alpha,
                           const Sleef_quad *A, int lda,
                           Sleef_quad *B, int ldb);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* QBLAS_LEVEL3_H */
