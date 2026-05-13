/* Level 2 entry points: gemv, ger, symv, trsv.
 *
 * Strategy
 * --------
 *   gemv:  Normalise to a "row-major no-trans" view inside the routine and
 *          dispatch to one of two kernels (gemv_n / gemv_t).
 *
 *          Column-major no-trans is equivalent to row-major trans (same data,
 *          flipped dimensions), so we collapse all four (layout, trans)
 *          combos to N or T against the row-major view.
 *
 *   ger:   Outer product update.  We thread over rows of A and call axpy on
 *          each row.  The data layout matters for vectorisation: row-major
 *          → contiguous y; col-major → contiguous x.  We branch accordingly.
 *
 *   symv:  Lower/upper triangle only.  Equivalent to a sym-product, but with
 *          extra work for the off-diagonal.  We implement it as two passes
 *          rather than special-casing the triangle.
 *
 *   trsv:  Forward/back substitution.  Inherently sequential along one
 *          dimension — implemented scalar.
 */

#include "common/qblas_internal.h"
#include "common/qblas_dispatch.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

/* Normalise gemv into a "row-major, optional T" call.
 *
 * Given the user-facing (layout, trans, m, n, A, lda):
 *   - layout=Row, trans=N:   y_m = A_mxn * x_n           → gemv_n(m, n, A, lda)
 *   - layout=Row, trans=T:   y_n = A^T_nxm * x_m         → gemv_t(m, n, A, lda)
 *   - layout=Col, trans=N:   y_m = A_mxn * x_n;
 *        Col-major A is identical to Row-major A^T with rows=n, cols=m
 *        → gemv_t(n, m, A, lda)
 *   - layout=Col, trans=T:   y_n = A^T_nxm * x_m;
 *        Col-major A^T is Row-major A (rows=n, cols=m, same buffer)
 *        → gemv_n(n, m, A, lda)
 */
void cblas_qgemv(QBLAS_LAYOUT layout, QBLAS_TRANSPOSE trans,
                 int m, int n,
                 Sleef_quad alpha,
                 const Sleef_quad *A, int lda,
                 const Sleef_quad *x, int incx,
                 Sleef_quad beta,
                 Sleef_quad *y, int incy) {
    if (m <= 0 || n <= 0) return;

    int doT = (trans == QblasTrans || trans == QblasConjTrans);
    int row = (layout == QblasRowMajor);

    /* nrows / ncols in the row-major view we'll dispatch against. */
    size_t rm_rows, rm_cols;
    int do_kernel_T;
    if (row && !doT)        { rm_rows = (size_t)m; rm_cols = (size_t)n; do_kernel_T = 0; }
    else if (row &&  doT)   { rm_rows = (size_t)m; rm_cols = (size_t)n; do_kernel_T = 1; }
    else if (!row && !doT)  { rm_rows = (size_t)n; rm_cols = (size_t)m; do_kernel_T = 1; }
    else /* !row && doT */  { rm_rows = (size_t)n; rm_cols = (size_t)m; do_kernel_T = 0; }

    /* Output length and input length wrt the row-major dispatch. */
    size_t y_len = do_kernel_T ? rm_cols : rm_rows;
    size_t x_len = do_kernel_T ? rm_rows : rm_cols;
    (void)x_len;

    /* Rebase pointers if strides are negative. */
    ptrdiff_t ox = (incx < 0) ? (ptrdiff_t)(x_len - 1) * (-incx) : 0;
    ptrdiff_t oy = (incy < 0) ? (ptrdiff_t)(y_len - 1) * (-incy) : 0;

    if (do_kernel_T) {
        int nthreads = qblas_resolve_threads(rm_rows * rm_cols, 1);
        if (nthreads > 1 && rm_rows * rm_cols >= qblas_tune()->gemv_thread_threshold) {
#ifdef _OPENMP
            /* gemv_t output y has length rm_cols.  We can parallelise along
             * rm_cols by treating each column of A^T as an independent dot
             * product against x — that is, dot of column j of A with x.
             *
             * The kernel-side gemv_t walks over rows of A doing an axpy into
             * y, which doesn't decompose by j cleanly without a private y
             * buffer.  Easier: do a parallel for over j, computing each y[j]
             * by dotting A[:, j] (a strided column) with x. */
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (size_t j = 0; j < y_len; ++j) {
                Sleef_quad s = qblas_dispatch_qdot(rm_rows,
                                                   A + j, (ptrdiff_t)lda,
                                                   x + ox, (ptrdiff_t)incx);
                Sleef_quad scaled = qmul(alpha, s);
                if (qiszero(beta)) y[oy + (ptrdiff_t)j * incy] = scaled;
                else y[oy + (ptrdiff_t)j * incy] =
                        qfma(beta, y[oy + (ptrdiff_t)j * incy], scaled);
            }
            return;
#endif
        }
        qblas_dispatch_qgemv_t(rm_rows, rm_cols, alpha, A, (size_t)lda,
                               x + ox, incx, beta, y + oy, incy);
    } else {
        int nthreads = qblas_resolve_threads(rm_rows * rm_cols, 1);
        if (nthreads > 1 && rm_rows * rm_cols >= qblas_tune()->gemv_thread_threshold) {
#ifdef _OPENMP
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nt  = omp_get_num_threads();
                size_t chunk = rm_rows / nt;
                size_t rem   = rm_rows % nt;
                size_t start = (size_t)tid * chunk + (tid < (int)rem ? (size_t)tid : rem);
                size_t cnt   = chunk + (tid < (int)rem ? 1u : 0u);
                qblas_dispatch_qgemv_n(cnt, rm_cols, alpha,
                                       A + start * (size_t)lda, (size_t)lda,
                                       x + ox, incx, beta,
                                       y + oy + (ptrdiff_t)start * incy, incy);
            }
            return;
#endif
        }
        qblas_dispatch_qgemv_n(rm_rows, rm_cols, alpha, A, (size_t)lda,
                               x + ox, incx, beta, y + oy, incy);
    }
}

/* ----------------- ger -------------------------------------------- */
void cblas_qger(QBLAS_LAYOUT layout,
                int m, int n,
                Sleef_quad alpha,
                const Sleef_quad *x, int incx,
                const Sleef_quad *y, int incy,
                Sleef_quad *A, int lda) {
    if (m <= 0 || n <= 0 || qiszero(alpha)) return;
    ptrdiff_t ox = (incx < 0) ? (ptrdiff_t)(m - 1) * (-incx) : 0;
    ptrdiff_t oy = (incy < 0) ? (ptrdiff_t)(n - 1) * (-incy) : 0;
    if (layout == QblasRowMajor) {
        /* A[i, j] += alpha * x[i] * y[j]; each row gets a separate axpy. */
        int nthreads = qblas_resolve_threads((size_t)m, (size_t)n);
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        #endif
        for (int i = 0; i < m; ++i) {
            Sleef_quad ax = qmul(alpha, x[ox + (ptrdiff_t)i * incx]);
            if (qiszero(ax)) continue;
            qblas_dispatch_qaxpy((size_t)n, ax, y + oy, incy, A + (size_t)i * lda, 1);
        }
    } else {
        /* Col-major: A[i, j] at A + j*lda + i; columns are contiguous. */
        int nthreads = qblas_resolve_threads((size_t)n, (size_t)m);
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        #endif
        for (int j = 0; j < n; ++j) {
            Sleef_quad ay = qmul(alpha, y[oy + (ptrdiff_t)j * incy]);
            if (qiszero(ay)) continue;
            qblas_dispatch_qaxpy((size_t)m, ay, x + ox, incx, A + (size_t)j * lda, 1);
        }
    }
}

/* ----------------- symv ------------------------------------------- */
/* Symmetric matrix-vector multiply.  We do not assume A's "other" triangle
 * is populated; we reference only the indicated `uplo` half.
 *
 * y = alpha * A * x + beta * y, A is n x n symmetric.
 *
 * Approach: first scale y by beta, then walk the upper (or lower) triangle.
 * For each row i, the diagonal element contributes A[i,i]*x[i] to y[i], and
 * each off-diagonal A[i,j] (j>i for Upper) contributes to both y[i] and y[j]
 * by symmetry. */
void cblas_qsymv(QBLAS_LAYOUT layout, QBLAS_UPLO uplo,
                 int n,
                 Sleef_quad alpha,
                 const Sleef_quad *A, int lda,
                 const Sleef_quad *x, int incx,
                 Sleef_quad beta,
                 Sleef_quad *y, int incy) {
    if (n <= 0) return;

    /* Column-major symmetric is layout-flipped; we can re-use the row-major
     * implementation by flipping uplo. */
    if (layout == QblasColMajor) {
        uplo = (uplo == QblasUpper) ? QblasLower : QblasUpper;
    }

    ptrdiff_t ox = (incx < 0) ? (ptrdiff_t)(n - 1) * (-incx) : 0;
    ptrdiff_t oy = (incy < 0) ? (ptrdiff_t)(n - 1) * (-incy) : 0;

    /* Scale y by beta first. */
    if (qiszero(beta)) {
        for (int i = 0; i < n; ++i) y[oy + (ptrdiff_t)i * incy] = QBLAS_ZERO;
    } else if (!qisone(beta)) {
        for (int i = 0; i < n; ++i)
            y[oy + (ptrdiff_t)i * incy] = qmul(beta, y[oy + (ptrdiff_t)i * incy]);
    }

    if (qiszero(alpha)) return;

    if (uplo == QblasUpper) {
        /* Row-major upper triangle: A[i, j] for j >= i. */
        for (int i = 0; i < n; ++i) {
            Sleef_quad xi = x[ox + (ptrdiff_t)i * incx];
            Sleef_quad axi = qmul(alpha, xi);
            Sleef_quad yi  = y[oy + (ptrdiff_t)i * incy];
            /* Diagonal */
            yi = qfma(axi, A[(size_t)i * lda + i], yi);
            /* Off-diagonal: row i, cols j>i */
            for (int j = i + 1; j < n; ++j) {
                Sleef_quad aij = A[(size_t)i * lda + j];
                yi = qfma(axi, aij, yi);
                /* And the symmetric A[j,i] contribution into y[j]. */
                y[oy + (ptrdiff_t)j * incy] =
                    qfma(qmul(alpha, A[(size_t)i * lda + j]),
                         x[ox + (ptrdiff_t)i * incx],
                         y[oy + (ptrdiff_t)j * incy]);
            }
            y[oy + (ptrdiff_t)i * incy] = yi;
        }
    } else {
        /* Lower. */
        for (int i = 0; i < n; ++i) {
            Sleef_quad xi = x[ox + (ptrdiff_t)i * incx];
            Sleef_quad axi = qmul(alpha, xi);
            Sleef_quad yi  = y[oy + (ptrdiff_t)i * incy];
            yi = qfma(axi, A[(size_t)i * lda + i], yi);
            for (int j = 0; j < i; ++j) {
                Sleef_quad aij = A[(size_t)i * lda + j];
                yi = qfma(axi, aij, yi);
                y[oy + (ptrdiff_t)j * incy] =
                    qfma(qmul(alpha, A[(size_t)i * lda + j]),
                         x[ox + (ptrdiff_t)i * incx],
                         y[oy + (ptrdiff_t)j * incy]);
            }
            y[oy + (ptrdiff_t)i * incy] = yi;
        }
    }
}

/* ----------------- trsv ------------------------------------------- */
/* Solve op(A) * x = b in place.  Scalar, sequential.  In practice trsv is
 * always memory-bound at this scale, so SIMD inside the trailing axpy is
 * what wins. */
void cblas_qtrsv(QBLAS_LAYOUT layout,
                 QBLAS_UPLO uplo,
                 QBLAS_TRANSPOSE trans,
                 QBLAS_DIAG diag,
                 int n,
                 const Sleef_quad *A, int lda,
                 Sleef_quad *x, int incx) {
    if (n <= 0) return;

    /* Flip layout into row-major view. */
    int doT = (trans == QblasTrans || trans == QblasConjTrans);
    if (layout == QblasColMajor) {
        uplo = (uplo == QblasUpper) ? QblasLower : QblasUpper;
        doT = !doT;
    }
    int upper = (uplo == QblasUpper);
    int unit  = (diag == QblasUnit);

    ptrdiff_t ox = (incx < 0) ? (ptrdiff_t)(n - 1) * (-incx) : 0;

    if (!doT) {
        if (!upper) {
            /* Forward substitution: i = 0..n-1 */
            for (int i = 0; i < n; ++i) {
                Sleef_quad sum = x[ox + (ptrdiff_t)i * incx];
                for (int j = 0; j < i; ++j)
                    sum = qsub(sum,
                               qmul(A[(size_t)i * lda + j],
                                    x[ox + (ptrdiff_t)j * incx]));
                if (!unit) sum = qdiv(sum, A[(size_t)i * lda + i]);
                x[ox + (ptrdiff_t)i * incx] = sum;
            }
        } else {
            /* Back substitution: i = n-1..0 */
            for (int i = n - 1; i >= 0; --i) {
                Sleef_quad sum = x[ox + (ptrdiff_t)i * incx];
                for (int j = i + 1; j < n; ++j)
                    sum = qsub(sum,
                               qmul(A[(size_t)i * lda + j],
                                    x[ox + (ptrdiff_t)j * incx]));
                if (!unit) sum = qdiv(sum, A[(size_t)i * lda + i]);
                x[ox + (ptrdiff_t)i * incx] = sum;
            }
        }
    } else {
        /* Transpose triangular solve. */
        if (!upper) {
            /* A is lower, A^T is upper → back-substitution over A^T means
             * iterate from i = n-1 down using column i of A as a row. */
            for (int i = n - 1; i >= 0; --i) {
                Sleef_quad sum = x[ox + (ptrdiff_t)i * incx];
                for (int j = i + 1; j < n; ++j)
                    sum = qsub(sum,
                               qmul(A[(size_t)j * lda + i],
                                    x[ox + (ptrdiff_t)j * incx]));
                if (!unit) sum = qdiv(sum, A[(size_t)i * lda + i]);
                x[ox + (ptrdiff_t)i * incx] = sum;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                Sleef_quad sum = x[ox + (ptrdiff_t)i * incx];
                for (int j = 0; j < i; ++j)
                    sum = qsub(sum,
                               qmul(A[(size_t)j * lda + i],
                                    x[ox + (ptrdiff_t)j * incx]));
                if (!unit) sum = qdiv(sum, A[(size_t)i * lda + i]);
                x[ox + (ptrdiff_t)i * incx] = sum;
            }
        }
    }
}
