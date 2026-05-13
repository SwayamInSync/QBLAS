/* Level 3 extras: syrk, trmm, trsm.
 *
 *   * syrk delegates to the optimised qgemm.
 *   * trmm and trsm use Goto-style blocking: the small diagonal block is
 *     solved (trsm) or multiplied (trmm) with a naive in-place loop, and
 *     the bulk update is a qgemm call — so they inherit qgemm's threading,
 *     packed micro-kernel, and cache blocking.
 *
 * Layout handling note: for column-major inputs we don't rewrite anything;
 * we just feed the user's pointers + layout flag straight into qgemm and a
 * layout-aware scalar diagonal kernel.  qgemm already collapses the col-
 * major case to row-major via M↔N swap internally.
 */

#include "common/qblas_internal.h"
#include "common/qblas_dispatch.h"

#include <stdlib.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

/* Read A[i,j] honoring user layout + trans flag.  Used only inside the
 * small diagonal block solver, so the per-element branch cost is fine. */
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

/* Row- or col-major B[i,j] reference. */
static inline Sleef_quad *B_at(Sleef_quad *B, int ldb, QBLAS_LAYOUT layout,
                               size_t i, size_t j) {
    return (layout == QblasRowMajor) ? B + i * (size_t)ldb + j
                                     : B + j * (size_t)ldb + i;
}

/* --------------------------------------------------------------------- */
/* syrk:  C := alpha * op(A) * op(A)^T + beta * C
 * Delegates to the full qgemm.  Strict BLAS spec only touches the given
 * triangle, but for a quad-precision library it's almost never a problem
 * to compute the full symmetric product and write both triangles. */
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

/* --------------------------------------------------------------------- */
/* In-place scaling B := alpha * B  (the column index `n` is fixed by ldb,
 * which differs between row- and col-major). */
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

/* --------------------------------------------------------------------- */
/* Small diagonal-block triangular solve for trsm-Left.
 *
 * Solves op(A_diag) * X = B  in-place where A_diag is a `bs x bs` chunk
 * of the user A starting at A[i0,i0] (interpreted with layout + trans).
 *
 * `B` is the corresponding `bs x n` slice of the user B, with rows of B
 * starting at row i0.  Strides follow `layout`.  Walks the right direction
 * (forward/back) based on (uplo, doT). */
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

    /* The columns of B (j-loop) are independent triangular solves against
     * the same diagonal block.  Hand them to OpenMP — each thread chases
     * its own column down/up the bs rows.  This converts the kernel from
     * O(bs²·n) scalar work into O(bs²·n / threads). */
    int nthreads = qblas_resolve_threads((size_t)bs * (size_t)n, (size_t)bs);
#ifdef _OPENMP
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

/* Small diagonal multiply for trmm-Left:
 *   B[i0:i0+bs, :] := op(A_diag) * B[i0:i0+bs, :]    (no alpha here) */
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

    int nthreads = qblas_resolve_threads((size_t)bs * (size_t)n, (size_t)bs);
#ifdef _OPENMP
    #pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
    for (int j = 0; j < n; ++j) {
        if (eff_upper) {
            /* Each row i = 0..bs-1 reads rows k >= i.  Walk top→bottom. */
            for (int i = 0; i < bs; ++i) {
                Sleef_quad sum = QBLAS_ZERO;
                int k_start = i;
                int k_end   = bs;
                for (int k = k_start; k < k_end; ++k) {
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
            /* Lower-effective: row i reads rows k <= i.  Walk bottom→top. */
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

/* Off-diagonal A "view" pointer math used to feed cblas_qgemm.  We pass
 * the raw A buffer + layout flag + the original lda; the trans flag stays
 * the same as the user's `transa`.  qgemm interprets A via layout + trans.
 *
 * For row-major no-trans: A[r, c] at A + r*lda + c.  Sub-block starting at
 * (r0, c0) of size (rb x cb) is at A + r0*lda + c0 with the same lda.
 * For row-major trans: A[r, c] virtual is at A + c*lda + r.  The sub-block
 * (r0, c0) is virtual_A^T(c0..c0+cb, r0..r0+rb), physically at A + r0*1 +
 * c0*lda — same pointer expression, the trans flag tells gemm to read it
 * back as transpose.
 *
 * Concretely we always pass A + (i0 * (layout==Row ? lda : 1)) +
 *                            (j0 * (layout==Row ? 1   : lda))  with the
 * user's trans flag.  But this is "before-transpose" coords.  We want
 * "after-transpose" coords (rows of op(A)).  After-transpose (r, c) =
 * before-transpose (c, r) when trans=T.
 *
 * Easier: introduce a helper that returns the pointer to the sub-block of
 * op(A) at virtual rows [r0..r0+rb), virtual cols [c0..c0+cb). */
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

/* Block size for trmm/trsm — below this just call the naive scalar path. */
#define QBLAS_TRSM_NB 64

/* --------------------------------------------------------------------- */
/* trsm: solve op(A) * X = alpha * B  (side = Left)
 *       solve X * op(A) = alpha * B  (side = Right)
 *
 * Strategy (Left):
 *   1. B := alpha * B in-place.
 *   2. For each NB-row strip of A (in solve order):
 *        - solve diagonal: A[ib..ib+NB, ib..ib+NB] * X = B[ib..ib+NB, :]
 *        - GEMM update: B[i_after..m, :] -= A[i_after..m, ib..ib+NB] * X
 *
 * The GEMM call uses the already-tuned cblas_qgemm.
 */
void cblas_qtrsm(QBLAS_LAYOUT layout,
                 QBLAS_SIDE side, QBLAS_UPLO uplo,
                 QBLAS_TRANSPOSE transa, QBLAS_DIAG diag,
                 int m, int n,
                 Sleef_quad alpha,
                 const Sleef_quad *A, int lda,
                 Sleef_quad *B, int ldb) {
    if (m <= 0 || n <= 0) return;

    /* Pre-scale B by alpha so the blocked solve operates on B in place
     * with alpha = 1 implicitly. */
    scale_B_inplace(layout, m, n, alpha, B, ldb);
    if (qiszero(alpha)) return;

    if (side == QblasLeft) {
        int doT   = (transa == QblasTrans || transa == QblasConjTrans);
        int upper = (uplo == QblasUpper);
        int eff_upper = doT ? !upper : upper;  /* op(A) shape */
        int NB = QBLAS_TRSM_NB;
        if (NB > m) NB = m;

        /* If the problem is small, just run a single diagonal block. */
        if (m <= NB) {
            trsm_left_diag(layout, uplo, transa, diag, m, n,
                           A, lda, 0, B, ldb, 0);
            return;
        }

        Sleef_quad neg_one = qneg(QBLAS_ONE);

        if (!eff_upper) {
            /* Forward substitution: ib = 0, NB, 2*NB, ... */
            for (int ib = 0; ib < m; ib += NB) {
                int bs = (ib + NB <= m) ? NB : m - ib;
                trsm_left_diag(layout, uplo, transa, diag, bs, n,
                               A, lda, (size_t)ib, B, ldb, (size_t)ib);
                int rest = m - (ib + bs);
                if (rest <= 0) continue;
                /* GEMM update: B[ib+bs:m, :] -= op(A)[ib+bs:m, ib:ib+bs] * B[ib:ib+bs, :]
                 *
                 * op(A) block: virtual rows [ib+bs..m), cols [ib..ib+bs). */
                const Sleef_quad *Ablk = opA_block_ptr(A, lda, layout, transa,
                                                       (size_t)(ib + bs), (size_t)ib);
                Sleef_quad *Bsub = B_at(B, ldb, layout, (size_t)(ib + bs), 0);
                Sleef_quad *Bdiag = B_at(B, ldb, layout, (size_t)ib, 0);
                cblas_qgemm(layout, transa, QblasNoTrans,
                            rest, n, bs,
                            neg_one,
                            Ablk, lda,
                            Bdiag, ldb,
                            QBLAS_ONE,
                            Bsub, ldb);
            }
        } else {
            /* Back substitution.  Walk from the bottom upward; each iteration
             * solves a block of rows [ib, ib+bs) and updates rows above it.
             * The top-most block may be smaller than NB when m % NB != 0. */
            int ib = m;
            while (ib > 0) {
                int bs = ib >= NB ? NB : ib;
                ib -= bs;
                trsm_left_diag(layout, uplo, transa, diag, bs, n,
                               A, lda, (size_t)ib, B, ldb, (size_t)ib);
                if (ib == 0) break;
                /* Update B[0:ib, :] -= op(A)[0:ib, ib:ib+bs] * B[ib:ib+bs, :] */
                const Sleef_quad *Ablk = opA_block_ptr(A, lda, layout, transa,
                                                       0, (size_t)ib);
                Sleef_quad *Bsub = B;  /* row 0 of B */
                Sleef_quad *Bdiag = B_at(B, ldb, layout, (size_t)ib, 0);
                cblas_qgemm(layout, transa, QblasNoTrans,
                            ib, n, bs,
                            neg_one,
                            Ablk, lda,
                            Bdiag, ldb,
                            QBLAS_ONE,
                            Bsub, ldb);
            }
        }
    } else {
        /* Right side: solve X * op(A) = B, A is n x n.  Reduce to Left via
         * the identity (X * op(A) = B) ⇔ (op(A)^T * X^T = B^T).  We don't
         * want to physically transpose, so we flip the layout flag — the
         * Left-side code path interprets the same buffer as the transposed
         * matrix.  Side-effect: m and n swap roles. */
        QBLAS_LAYOUT new_layout = (layout == QblasRowMajor) ? QblasColMajor : QblasRowMajor;
        /* op(A)^T means flipping `transa`. */
        QBLAS_TRANSPOSE new_ta = (transa == QblasNoTrans) ? QblasTrans : QblasNoTrans;
        cblas_qtrsm(new_layout, QblasLeft, uplo, new_ta, diag,
                    n, m, QBLAS_ONE, A, lda, B, ldb);
        /* alpha was already applied above */
    }
}

/* --------------------------------------------------------------------- */
/* trmm: B := alpha * op(A) * B  (Left) or alpha * B * op(A)  (Right).
 *
 * Blocked Left strategy:
 *   - Lower-effective (op(A) is lower triangular):
 *       walk ib = m-NB .. 0 step -NB:
 *         GEMM: B[ib..ib+bs, :] += A[ib..ib+bs, 0..ib] * B[0..ib, :]
 *         trmm_left_diag on B[ib..ib+bs, :]
 *   - Upper-effective:
 *       walk ib = 0 .. m step NB:
 *         GEMM: B[ib..ib+bs, :] += A[ib..ib+bs, ib+bs..m] * B[ib+bs..m, :]
 *         trmm_left_diag on B[ib..ib+bs, :]
 *
 * The post-blocked alpha-scale is folded in by scaling B by alpha as a
 * final step.
 */
void cblas_qtrmm(QBLAS_LAYOUT layout,
                 QBLAS_SIDE side, QBLAS_UPLO uplo,
                 QBLAS_TRANSPOSE transa, QBLAS_DIAG diag,
                 int m, int n,
                 Sleef_quad alpha,
                 const Sleef_quad *A, int lda,
                 Sleef_quad *B, int ldb) {
    if (m <= 0 || n <= 0) return;

    if (side == QblasRight) {
        /* Right: B * op(A) = (op(A)^T * B^T)^T.  Flip layout + transa, swap m/n. */
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
        /* Upper-effective: row i depends on rows k >= i.  Process top→bottom
         * (smaller ib first) AFTER its dependencies are still original. */
        for (int ib = 0; ib < m; ib += NB) {
            int bs = (ib + NB <= m) ? NB : m - ib;
            int rest = m - (ib + bs);
            if (rest > 0) {
                /* B[ib..ib+bs, :] += op(A)[ib..ib+bs, ib+bs..m] * B[ib+bs..m, :] */
                const Sleef_quad *Ablk = opA_block_ptr(A, lda, layout, transa,
                                                       (size_t)ib, (size_t)(ib + bs));
                Sleef_quad *Brest = B_at(B, ldb, layout, (size_t)(ib + bs), 0);
                Sleef_quad *Bblk  = B_at(B, ldb, layout, (size_t)ib, 0);
                cblas_qgemm(layout, transa, QblasNoTrans,
                            bs, n, rest,
                            QBLAS_ONE,
                            Ablk, lda,
                            Brest, ldb,
                            QBLAS_ONE,
                            Bblk, ldb);
            }
            /* Diagonal multiply on B[ib..ib+bs, :] (overwrites in-place). */
            trmm_left_diag(layout, uplo, transa, diag, bs, n,
                           A, lda, (size_t)ib, B, ldb, (size_t)ib);
        }
    } else {
        /* Lower-effective: row i depends on rows k <= i.  Process bottom→top.
         * Top block may have size < NB when m % NB != 0. */
        int ib = m;
        while (ib > 0) {
            int bs = ib >= NB ? NB : ib;
            ib -= bs;
            if (ib > 0) {
                /* B[ib..ib+bs, :] += op(A)[ib..ib+bs, 0..ib] * B[0..ib, :] */
                const Sleef_quad *Ablk = opA_block_ptr(A, lda, layout, transa,
                                                       (size_t)ib, 0);
                Sleef_quad *Brest = B;
                Sleef_quad *Bblk  = B_at(B, ldb, layout, (size_t)ib, 0);
                cblas_qgemm(layout, transa, QblasNoTrans,
                            bs, n, ib,
                            QBLAS_ONE,
                            Ablk, lda,
                            Brest, ldb,
                            QBLAS_ONE,
                            Bblk, ldb);
            }
            trmm_left_diag(layout, uplo, transa, diag, bs, n,
                           A, lda, (size_t)ib, B, ldb, (size_t)ib);
        }
    }

    scale_B_inplace(layout, m, n, alpha, B, ldb);
}
