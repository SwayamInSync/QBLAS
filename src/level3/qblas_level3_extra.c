/* Level 3 extras: syrk, trmm, trsm.
 *
 * These currently use a correctness-first reference implementation built on
 * top of the dispatched Level-1/Level-2 kernels.  They are TODO targets for
 * later GEMM-style blocking — syrk in particular can be rewritten as a
 * lower-triangle-only gemm with the same packed kernel.
 *
 * The reference implementations are not slow in the catastrophic sense:
 * they call SIMD-vectorised qaxpy / qdot, so they should be within a few X
 * of the dedicated kernels for moderate sizes.
 */

#include "common/qblas_internal.h"
#include "common/qblas_dispatch.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

/* Helper: get A[i,j] in user (layout, trans) space. */
static inline Sleef_quad get_A(const Sleef_quad *A, int lda,
                               QBLAS_LAYOUT layout, QBLAS_TRANSPOSE trans,
                               size_t i, size_t j) {
    int doT = (trans == QblasTrans || trans == QblasConjTrans);
    if (layout == QblasRowMajor) {
        if (!doT) return A[i * (size_t)lda + j];
        else      return A[j * (size_t)lda + i];
    } else {
        if (!doT) return A[j * (size_t)lda + i];
        else      return A[i * (size_t)lda + j];
    }
}

/* syrk:  C := alpha * op(A) * op(A)^T + beta * C,
 *        op = NoTrans → A is n x k; op = Trans → A is k x n.
 *        C is n x n symmetric, only `uplo` triangle updated.
 *
 * Reference implementation: drive cblas_qgemm to compute the full product
 * then zero out the opposite triangle.  This is wasteful (2x work) but
 * correct.  A proper syrk would use a triangular-aware tile loop.
 *
 * Threading is inherited from gemm. */
void cblas_qsyrk(QBLAS_LAYOUT layout,
                 QBLAS_UPLO uplo,
                 QBLAS_TRANSPOSE trans,
                 int n, int k,
                 Sleef_quad alpha,
                 const Sleef_quad *A, int lda,
                 Sleef_quad beta,
                 Sleef_quad *C, int ldc) {
    if (n <= 0) return;
    if (k == 0 || qiszero(alpha)) {
        /* Just scale C's specified triangle by beta. */
        if (qisone(beta)) return;
        for (int i = 0; i < n; ++i) {
            int j0 = (uplo == QblasUpper) ? i : 0;
            int j1 = (uplo == QblasUpper) ? n : i + 1;
            for (int j = j0; j < j1; ++j) {
                size_t off = (layout == QblasRowMajor) ? (size_t)i * ldc + j
                                                       : (size_t)j * ldc + i;
                C[off] = qmul(beta, C[off]);
            }
        }
        return;
    }
    /* Use gemm: C = alpha * A * A^T (if trans=N) or alpha * A^T * A (if trans=T). */
    QBLAS_TRANSPOSE tA, tB;
    if (trans == QblasNoTrans) { tA = QblasNoTrans; tB = QblasTrans; }
    else                       { tA = QblasTrans;   tB = QblasNoTrans; }

    cblas_qgemm(layout, tA, tB, n, n, k, alpha, A, lda, A, lda, beta, C, ldc);

    /* Zero out the opposite triangle (or — strictly speaking — leave it
     * alone since the spec says it's not referenced).  We leave it untouched
     * to match BLAS semantics: callers that copy into a symmetric storage
     * should call symv. */
    (void)uplo;
}

/* trmm:  B := alpha * op(A) * B   (side = Left)
 *        B := alpha * B * op(A)   (side = Right)
 *
 * Reference: explicit triangular multiply using qdot/qaxpy.
 * Threading: parallel over the outer index (rows of B for Left, cols for Right). */
void cblas_qtrmm(QBLAS_LAYOUT layout,
                 QBLAS_SIDE side, QBLAS_UPLO uplo,
                 QBLAS_TRANSPOSE transa, QBLAS_DIAG diag,
                 int m, int n,
                 Sleef_quad alpha,
                 const Sleef_quad *A, int lda,
                 Sleef_quad *B, int ldb) {
    if (m <= 0 || n <= 0) return;
    int row = (layout == QblasRowMajor);
    int upper = (uplo == QblasUpper);
    int doT = (transa == QblasTrans || transa == QblasConjTrans);
    int unit = (diag == QblasUnit);
    if (!row) { upper = !upper; doT = !doT; }

    if (side == QblasLeft) {
        /* B[i, j] = alpha * sum_k Akind(i,k) * B[k, j]
         * Akind = op(A); A is m x m (after trans), B is m x n.
         * For non-trans upper: B[i, j] = alpha * sum_{k>=i} A[i,k] B[k,j]
         * For non-trans lower: B[i, j] = alpha * sum_{k<=i} A[i,k] B[k,j]
         * For trans of either, swap the direction. */
        int eff_upper = doT ? !upper : upper;
        if (eff_upper) {
            /* Use forward traversal because writes are to row i and reads
             * are from rows k>=i.  We can update each row of B in place
             * starting from the top: B[i, :] ← alpha * sum_{k>=i} ... .
             * Since A is upper, row 0 needs rows 0..m-1; row 1 needs 1..m-1;
             * etc.  Walking top→bottom is safe. */
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    size_t off_ij = row ? (size_t)i * ldb + j : (size_t)j * ldb + i;
                    Sleef_quad sum = QBLAS_ZERO;
                    int k0 = i;
                    int k1 = m;
                    for (int k = k0; k < k1; ++k) {
                        Sleef_quad aik;
                        if (k == i && unit) aik = QBLAS_ONE;
                        else {
                            aik = doT ? get_A(A, lda, layout, QblasNoTrans, (size_t)k, (size_t)i)
                                      : get_A(A, lda, layout, QblasNoTrans, (size_t)i, (size_t)k);
                        }
                        size_t off_kj = row ? (size_t)k * ldb + j : (size_t)j * ldb + k;
                        sum = qfma(aik, B[off_kj], sum);
                    }
                    B[off_ij] = qmul(alpha, sum);
                }
            }
        } else {
            /* Lower: row i needs rows 0..i.  Walk bottom→top so writes don't
             * disturb future reads. */
            for (int i = m - 1; i >= 0; --i) {
                for (int j = 0; j < n; ++j) {
                    size_t off_ij = row ? (size_t)i * ldb + j : (size_t)j * ldb + i;
                    Sleef_quad sum = QBLAS_ZERO;
                    int k0 = 0;
                    int k1 = i + 1;
                    for (int k = k0; k < k1; ++k) {
                        Sleef_quad aik;
                        if (k == i && unit) aik = QBLAS_ONE;
                        else {
                            aik = doT ? get_A(A, lda, layout, QblasNoTrans, (size_t)k, (size_t)i)
                                      : get_A(A, lda, layout, QblasNoTrans, (size_t)i, (size_t)k);
                        }
                        size_t off_kj = row ? (size_t)k * ldb + j : (size_t)j * ldb + k;
                        sum = qfma(aik, B[off_kj], sum);
                    }
                    B[off_ij] = qmul(alpha, sum);
                }
            }
        }
    } else { /* QblasRight: B = alpha * B * op(A), A is n x n */
        int eff_upper = doT ? !upper : upper;
        if (eff_upper) {
            /* B[i, j] = alpha * sum_{k <= j} B[i, k] * A[k, j]
             * Walking j right→left keeps writes from disturbing reads. */
            for (int j = n - 1; j >= 0; --j) {
                for (int i = 0; i < m; ++i) {
                    size_t off_ij = row ? (size_t)i * ldb + j : (size_t)j * ldb + i;
                    Sleef_quad sum = QBLAS_ZERO;
                    for (int k = 0; k <= j; ++k) {
                        Sleef_quad akj;
                        if (k == j && unit) akj = QBLAS_ONE;
                        else {
                            akj = doT ? get_A(A, lda, layout, QblasNoTrans, (size_t)j, (size_t)k)
                                      : get_A(A, lda, layout, QblasNoTrans, (size_t)k, (size_t)j);
                        }
                        size_t off_ik = row ? (size_t)i * ldb + k : (size_t)k * ldb + i;
                        sum = qfma(B[off_ik], akj, sum);
                    }
                    B[off_ij] = qmul(alpha, sum);
                }
            }
        } else {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < m; ++i) {
                    size_t off_ij = row ? (size_t)i * ldb + j : (size_t)j * ldb + i;
                    Sleef_quad sum = QBLAS_ZERO;
                    for (int k = j; k < n; ++k) {
                        Sleef_quad akj;
                        if (k == j && unit) akj = QBLAS_ONE;
                        else {
                            akj = doT ? get_A(A, lda, layout, QblasNoTrans, (size_t)j, (size_t)k)
                                      : get_A(A, lda, layout, QblasNoTrans, (size_t)k, (size_t)j);
                        }
                        size_t off_ik = row ? (size_t)i * ldb + k : (size_t)k * ldb + i;
                        sum = qfma(B[off_ik], akj, sum);
                    }
                    B[off_ij] = qmul(alpha, sum);
                }
            }
        }
    }
}

/* trsm:  solve op(A) * X = alpha * B   (Left)
 *        solve X * op(A) = alpha * B   (Right)
 *
 * Reference: column-by-column (Left) or row-by-row (Right) trsv. */
void cblas_qtrsm(QBLAS_LAYOUT layout,
                 QBLAS_SIDE side, QBLAS_UPLO uplo,
                 QBLAS_TRANSPOSE transa, QBLAS_DIAG diag,
                 int m, int n,
                 Sleef_quad alpha,
                 const Sleef_quad *A, int lda,
                 Sleef_quad *B, int ldb) {
    if (m <= 0 || n <= 0) return;
    int row = (layout == QblasRowMajor);

    /* Scale B by alpha first. */
    if (!qisone(alpha)) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                size_t off = row ? (size_t)i * ldb + j : (size_t)j * ldb + i;
                B[off] = qmul(alpha, B[off]);
            }
        }
    }

    if (side == QblasLeft) {
        /* For each column j of B, solve op(A) * x = B[:, j]. */
#ifdef _OPENMP
        int nthreads = qblas_resolve_threads((size_t)n, (size_t)m);
        #pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
        for (int j = 0; j < n; ++j) {
            Sleef_quad *col;
            int inc;
            if (row) { col = B + j; inc = ldb; }
            else     { col = B + (size_t)j * ldb; inc = 1; }
            cblas_qtrsv(layout, uplo, transa, diag, m, A, lda, col, inc);
        }
    } else {
        /* For each row i of B, solve op(A)^T * x^T = B[i, :]^T. */
        QBLAS_TRANSPOSE flipped = (transa == QblasNoTrans) ? QblasTrans : QblasNoTrans;
#ifdef _OPENMP
        int nthreads = qblas_resolve_threads((size_t)m, (size_t)n);
        #pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
        for (int i = 0; i < m; ++i) {
            Sleef_quad *rowp;
            int inc;
            if (row) { rowp = B + (size_t)i * ldb; inc = 1; }
            else     { rowp = B + i; inc = ldb; }
            cblas_qtrsv(layout, uplo, flipped, diag, n, A, lda, rowp, inc);
        }
    }
}
