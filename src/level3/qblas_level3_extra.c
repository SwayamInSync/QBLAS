/* syrk delegates to qgemm.  trmm/trsm use Goto-style blocking: the
 * diagonal block is solved/multiplied in-place (parallel over columns)
 * and the trailing matrix update is a cblas_qgemm() call, so they
 * inherit gemm's packed kernel, blocking, and threading. */

#include "common/qblas_internal.h"
#include "common/qblas_dispatch.h"

#include <stdlib.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

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

static inline Sleef_quad *B_at(Sleef_quad *B, int ldb, QBLAS_LAYOUT layout,
                               size_t i, size_t j) {
    return (layout == QblasRowMajor) ? B + i * (size_t)ldb + j
                                     : B + j * (size_t)ldb + i;
}

void cblas_qsyrk(QBLAS_LAYOUT layout,
                 QBLAS_UPLO uplo,
                 QBLAS_TRANSPOSE trans,
                 int n, int k,
                 Sleef_quad alpha,
                 const Sleef_quad *A, int lda,
                 Sleef_quad beta,
                 Sleef_quad *C, int ldc) {
    (void)uplo;
    if (n <= 0) return;
    if (k == 0 || qiszero(alpha)) {
        if (qisone(beta)) return;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                *B_at(C, ldc, layout, (size_t)i, (size_t)j) =
                    qmul(beta, *B_at(C, ldc, layout, (size_t)i, (size_t)j));
        return;
    }
    QBLAS_TRANSPOSE tA = (trans == QblasNoTrans) ? QblasNoTrans : QblasTrans;
    QBLAS_TRANSPOSE tB = (trans == QblasNoTrans) ? QblasTrans   : QblasNoTrans;
    cblas_qgemm(layout, tA, tB, n, n, k, alpha, A, lda, A, lda, beta, C, ldc);
}

static void scale_B_inplace(QBLAS_LAYOUT layout, int m, int n,
                            Sleef_quad alpha, Sleef_quad *B, int ldb) {
    if (qisone(alpha)) return;
    if (qiszero(alpha)) {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                *B_at(B, ldb, layout, (size_t)i, (size_t)j) = QBLAS_ZERO;
        return;
    }
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            Sleef_quad *p = B_at(B, ldb, layout, (size_t)i, (size_t)j);
            *p = qmul(alpha, *p);
        }
}

/* Diagonal block solve.  Columns of B are independent so they fan out
 * across threads via omp parallel for. */
static void trsm_left_diag(QBLAS_LAYOUT layout, QBLAS_UPLO uplo,
                           QBLAS_TRANSPOSE transa, QBLAS_DIAG diag,
                           int bs, int n,
                           const Sleef_quad *A, int lda,
                           size_t i0,
                           Sleef_quad *B, int ldb,
                           size_t row_off) {
    int doT = (transa == QblasTrans || transa == QblasConjTrans);
    int upper = (uplo == QblasUpper);
    int eff_upper = doT ? !upper : upper;
    int unit = (diag == QblasUnit);

    int step = eff_upper ? -1 : 1;
    int i_start = eff_upper ? bs - 1 : 0;
    int i_end   = eff_upper ? -1     : bs;

#ifdef _OPENMP
    int nthreads = qblas_resolve_threads((size_t)bs * (size_t)n, (size_t)bs);
    #pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
    for (int j = 0; j < n; ++j) {
        for (int i = i_start; i != i_end; i += step) {
            int k_start = eff_upper ? i + 1 : 0;
            int k_end   = eff_upper ? bs    : i;
            Sleef_quad rhs = *B_at(B, ldb, layout, row_off + (size_t)i, (size_t)j);
            for (int k = k_start; k < k_end; ++k) {
                Sleef_quad aik = doT
                    ? get_A(A, lda, layout, QblasNoTrans, i0 + (size_t)k, i0 + (size_t)i)
                    : get_A(A, lda, layout, QblasNoTrans, i0 + (size_t)i, i0 + (size_t)k);
                Sleef_quad xkj = *B_at(B, ldb, layout, row_off + (size_t)k, (size_t)j);
                rhs = qsub(rhs, qmul(aik, xkj));
            }
            if (!unit) {
                Sleef_quad aii = get_A(A, lda, layout, QblasNoTrans,
                                        i0 + (size_t)i, i0 + (size_t)i);
                rhs = qdiv(rhs, aii);
            }
            *B_at(B, ldb, layout, row_off + (size_t)i, (size_t)j) = rhs;
        }
    }
}

static void trmm_left_diag(QBLAS_LAYOUT layout, QBLAS_UPLO uplo,
                           QBLAS_TRANSPOSE transa, QBLAS_DIAG diag,
                           int bs, int n,
                           const Sleef_quad *A, int lda,
                           size_t i0,
                           Sleef_quad *B, int ldb,
                           size_t row_off) {
    int doT   = (transa == QblasTrans || transa == QblasConjTrans);
    int upper = (uplo == QblasUpper);
    int eff_upper = doT ? !upper : upper;
    int unit  = (diag == QblasUnit);

#ifdef _OPENMP
    int nthreads = qblas_resolve_threads((size_t)bs * (size_t)n, (size_t)bs);
    #pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
    for (int j = 0; j < n; ++j) {
        if (eff_upper) {
            for (int i = 0; i < bs; ++i) {
                Sleef_quad sum = QBLAS_ZERO;
                for (int k = i; k < bs; ++k) {
                    Sleef_quad aik;
                    if (k == i && unit) aik = QBLAS_ONE;
                    else aik = doT
                        ? get_A(A, lda, layout, QblasNoTrans, i0 + (size_t)k, i0 + (size_t)i)
                        : get_A(A, lda, layout, QblasNoTrans, i0 + (size_t)i, i0 + (size_t)k);
                    Sleef_quad bkj = *B_at(B, ldb, layout, row_off + (size_t)k, (size_t)j);
                    sum = qfma(aik, bkj, sum);
                }
                *B_at(B, ldb, layout, row_off + (size_t)i, (size_t)j) = sum;
            }
        } else {
            for (int i = bs - 1; i >= 0; --i) {
                Sleef_quad sum = QBLAS_ZERO;
                for (int k = 0; k <= i; ++k) {
                    Sleef_quad aik;
                    if (k == i && unit) aik = QBLAS_ONE;
                    else aik = doT
                        ? get_A(A, lda, layout, QblasNoTrans, i0 + (size_t)k, i0 + (size_t)i)
                        : get_A(A, lda, layout, QblasNoTrans, i0 + (size_t)i, i0 + (size_t)k);
                    Sleef_quad bkj = *B_at(B, ldb, layout, row_off + (size_t)k, (size_t)j);
                    sum = qfma(aik, bkj, sum);
                }
                *B_at(B, ldb, layout, row_off + (size_t)i, (size_t)j) = sum;
            }
        }
    }
}

/* Returns pointer to op(A)[r0:r0+rb, c0:c0+cb]; the trans flag stays the
 * user's so cblas_qgemm interprets the buffer correctly. */
static const Sleef_quad *opA_block_ptr(const Sleef_quad *A, int lda,
                                       QBLAS_LAYOUT layout,
                                       QBLAS_TRANSPOSE transa,
                                       size_t r0, size_t c0) {
    int doT = (transa == QblasTrans || transa == QblasConjTrans);
    if (layout == QblasRowMajor) {
        if (!doT) return A + r0 * (size_t)lda + c0;
        else      return A + c0 * (size_t)lda + r0;
    } else {
        if (!doT) return A + r0 + c0 * (size_t)lda;
        else      return A + c0 + r0 * (size_t)lda;
    }
}

#define QBLAS_TRSM_NB 64

void cblas_qtrsm(QBLAS_LAYOUT layout,
                 QBLAS_SIDE side, QBLAS_UPLO uplo,
                 QBLAS_TRANSPOSE transa, QBLAS_DIAG diag,
                 int m, int n,
                 Sleef_quad alpha,
                 const Sleef_quad *A, int lda,
                 Sleef_quad *B, int ldb) {
    if (m <= 0 || n <= 0) return;

    scale_B_inplace(layout, m, n, alpha, B, ldb);
    if (qiszero(alpha)) return;

    if (side == QblasLeft) {
        int doT   = (transa == QblasTrans || transa == QblasConjTrans);
        int upper = (uplo == QblasUpper);
        int eff_upper = doT ? !upper : upper;
        int NB = QBLAS_TRSM_NB;
        if (NB > m) NB = m;

        if (m <= NB) {
            trsm_left_diag(layout, uplo, transa, diag, m, n,
                           A, lda, 0, B, ldb, 0);
            return;
        }

        Sleef_quad neg_one = qneg(QBLAS_ONE);

        if (!eff_upper) {
            /* Forward substitution: solve top block, gemm-update rows below. */
            for (int ib = 0; ib < m; ib += NB) {
                int bs = (ib + NB <= m) ? NB : m - ib;
                trsm_left_diag(layout, uplo, transa, diag, bs, n,
                               A, lda, (size_t)ib, B, ldb, (size_t)ib);
                int rest = m - (ib + bs);
                if (rest <= 0) continue;
                const Sleef_quad *Ablk = opA_block_ptr(A, lda, layout, transa,
                                                       (size_t)(ib + bs), (size_t)ib);
                Sleef_quad *Bsub  = B_at(B, ldb, layout, (size_t)(ib + bs), 0);
                Sleef_quad *Bdiag = B_at(B, ldb, layout, (size_t)ib, 0);
                cblas_qgemm(layout, transa, QblasNoTrans,
                            rest, n, bs,
                            neg_one, Ablk, lda, Bdiag, ldb,
                            QBLAS_ONE, Bsub, ldb);
            }
        } else {
            /* Back substitution: walk bottom-up.  Top block may be partial
             * when m % NB != 0. */
            int ib = m;
            while (ib > 0) {
                int bs = ib >= NB ? NB : ib;
                ib -= bs;
                trsm_left_diag(layout, uplo, transa, diag, bs, n,
                               A, lda, (size_t)ib, B, ldb, (size_t)ib);
                if (ib == 0) break;
                const Sleef_quad *Ablk = opA_block_ptr(A, lda, layout, transa,
                                                       0, (size_t)ib);
                Sleef_quad *Bsub  = B;
                Sleef_quad *Bdiag = B_at(B, ldb, layout, (size_t)ib, 0);
                cblas_qgemm(layout, transa, QblasNoTrans,
                            ib, n, bs,
                            neg_one, Ablk, lda, Bdiag, ldb,
                            QBLAS_ONE, Bsub, ldb);
            }
        }
    } else {
        /* X op(A) = alpha B  <=>  op(A)^T X^T = alpha B^T : flip layout
         * and transa, swap m/n, recurse into Left. */
        QBLAS_LAYOUT new_layout = (layout == QblasRowMajor) ? QblasColMajor : QblasRowMajor;
        QBLAS_TRANSPOSE new_ta = (transa == QblasNoTrans) ? QblasTrans : QblasNoTrans;
        cblas_qtrsm(new_layout, QblasLeft, uplo, new_ta, diag,
                    n, m, QBLAS_ONE, A, lda, B, ldb);
    }
}

void cblas_qtrmm(QBLAS_LAYOUT layout,
                 QBLAS_SIDE side, QBLAS_UPLO uplo,
                 QBLAS_TRANSPOSE transa, QBLAS_DIAG diag,
                 int m, int n,
                 Sleef_quad alpha,
                 const Sleef_quad *A, int lda,
                 Sleef_quad *B, int ldb) {
    if (m <= 0 || n <= 0) return;

    if (side == QblasRight) {
        QBLAS_LAYOUT new_layout = (layout == QblasRowMajor) ? QblasColMajor : QblasRowMajor;
        QBLAS_TRANSPOSE new_ta = (transa == QblasNoTrans) ? QblasTrans : QblasNoTrans;
        cblas_qtrmm(new_layout, QblasLeft, uplo, new_ta, diag,
                    n, m, alpha, A, lda, B, ldb);
        return;
    }

    int doT   = (transa == QblasTrans || transa == QblasConjTrans);
    int upper = (uplo == QblasUpper);
    int eff_upper = doT ? !upper : upper;
    int NB = QBLAS_TRSM_NB;
    if (NB > m) NB = m;

    if (m <= NB) {
        trmm_left_diag(layout, uplo, transa, diag, m, n, A, lda, 0, B, ldb, 0);
        scale_B_inplace(layout, m, n, alpha, B, ldb);
        return;
    }

    if (eff_upper) {
        /* Row i depends on rows k >= i; process top-down with gemm-update
         * from below before each diagonal block. */
        for (int ib = 0; ib < m; ib += NB) {
            int bs = (ib + NB <= m) ? NB : m - ib;
            int rest = m - (ib + bs);
            if (rest > 0) {
                const Sleef_quad *Ablk = opA_block_ptr(A, lda, layout, transa,
                                                       (size_t)ib, (size_t)(ib + bs));
                Sleef_quad *Brest = B_at(B, ldb, layout, (size_t)(ib + bs), 0);
                Sleef_quad *Bblk  = B_at(B, ldb, layout, (size_t)ib, 0);
                cblas_qgemm(layout, transa, QblasNoTrans,
                            bs, n, rest,
                            QBLAS_ONE, Ablk, lda, Brest, ldb,
                            QBLAS_ONE, Bblk, ldb);
            }
            trmm_left_diag(layout, uplo, transa, diag, bs, n,
                           A, lda, (size_t)ib, B, ldb, (size_t)ib);
        }
    } else {
        /* Row i depends on rows k <= i; process bottom-up. */
        int ib = m;
        while (ib > 0) {
            int bs = ib >= NB ? NB : ib;
            ib -= bs;
            if (ib > 0) {
                const Sleef_quad *Ablk = opA_block_ptr(A, lda, layout, transa,
                                                       (size_t)ib, 0);
                Sleef_quad *Brest = B;
                Sleef_quad *Bblk  = B_at(B, ldb, layout, (size_t)ib, 0);
                cblas_qgemm(layout, transa, QblasNoTrans,
                            bs, n, ib,
                            QBLAS_ONE, Ablk, lda, Brest, ldb,
                            QBLAS_ONE, Bblk, ldb);
            }
            trmm_left_diag(layout, uplo, transa, diag, bs, n,
                           A, lda, (size_t)ib, B, ldb, (size_t)ib);
        }
    }

    scale_B_inplace(layout, m, n, alpha, B, ldb);
}
