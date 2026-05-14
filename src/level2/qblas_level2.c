#include "common/qblas_internal.h"
#include "common/qblas_dispatch.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

/* Collapse all four (layout, trans) gemv combos to a row-major view that
 * dispatches to one of qgemv_n / qgemv_t.  Col-major no-trans is identical
 * to row-major trans with M and N swapped, and vice versa. */
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

    size_t rm_rows, rm_cols;
    int do_kernel_T;
    if      (row  && !doT) { rm_rows = (size_t)m; rm_cols = (size_t)n; do_kernel_T = 0; }
    else if (row  &&  doT) { rm_rows = (size_t)m; rm_cols = (size_t)n; do_kernel_T = 1; }
    else if (!row && !doT) { rm_rows = (size_t)n; rm_cols = (size_t)m; do_kernel_T = 1; }
    else                   { rm_rows = (size_t)n; rm_cols = (size_t)m; do_kernel_T = 0; }

    size_t y_len = do_kernel_T ? rm_cols : rm_rows;
    size_t x_len = do_kernel_T ? rm_rows : rm_cols;
    (void)x_len;

    ptrdiff_t ox = (incx < 0) ? (ptrdiff_t)(x_len - 1) * (-incx) : 0;
    ptrdiff_t oy = (incy < 0) ? (ptrdiff_t)(y_len - 1) * (-incy) : 0;

    if (do_kernel_T) {
        int nthreads = qblas_resolve_threads(rm_rows * rm_cols, 1);
        if (nthreads > 1 && rm_rows * rm_cols >= qblas_tune()->gemv_thread_threshold) {
#ifdef _OPENMP
            /* Parallelise over output columns: each y[j] is a strided dot
             * of A's column j with x.  Avoids needing per-thread y buffers. */
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

/* For each row i: y[i] += alpha * sum_j A[i,j] * x[j], using the indicated
 * triangle of A and the symmetric pair on the other side. */
void cblas_qsymv(QBLAS_LAYOUT layout, QBLAS_UPLO uplo,
                 int n,
                 Sleef_quad alpha,
                 const Sleef_quad *A, int lda,
                 const Sleef_quad *x, int incx,
                 Sleef_quad beta,
                 Sleef_quad *y, int incy) {
    if (n <= 0) return;

    /* Symmetric col-major is row-major with the triangle flipped. */
    if (layout == QblasColMajor) {
        uplo = (uplo == QblasUpper) ? QblasLower : QblasUpper;
    }

    ptrdiff_t ox = (incx < 0) ? (ptrdiff_t)(n - 1) * (-incx) : 0;
    ptrdiff_t oy = (incy < 0) ? (ptrdiff_t)(n - 1) * (-incy) : 0;

    if (qiszero(beta)) {
        for (int i = 0; i < n; ++i) y[oy + (ptrdiff_t)i * incy] = QBLAS_ZERO;
    } else if (!qisone(beta)) {
        for (int i = 0; i < n; ++i)
            y[oy + (ptrdiff_t)i * incy] = qmul(beta, y[oy + (ptrdiff_t)i * incy]);
    }

    if (qiszero(alpha)) return;

    if (uplo == QblasUpper) {
        for (int i = 0; i < n; ++i) {
            Sleef_quad xi  = x[ox + (ptrdiff_t)i * incx];
            Sleef_quad axi = qmul(alpha, xi);
            Sleef_quad yi  = y[oy + (ptrdiff_t)i * incy];
            yi = qfma(axi, A[(size_t)i * lda + i], yi);
            for (int j = i + 1; j < n; ++j) {
                Sleef_quad aij = A[(size_t)i * lda + j];
                yi = qfma(axi, aij, yi);
                y[oy + (ptrdiff_t)j * incy] =
                    qfma(qmul(alpha, A[(size_t)i * lda + j]),
                         x[ox + (ptrdiff_t)i * incx],
                         y[oy + (ptrdiff_t)j * incy]);
            }
            y[oy + (ptrdiff_t)i * incy] = yi;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            Sleef_quad xi  = x[ox + (ptrdiff_t)i * incx];
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

void cblas_qtrsv(QBLAS_LAYOUT layout,
                 QBLAS_UPLO uplo,
                 QBLAS_TRANSPOSE trans,
                 QBLAS_DIAG diag,
                 int n,
                 const Sleef_quad *A, int lda,
                 Sleef_quad *x, int incx) {
    if (n <= 0) return;

    /* Reinterpret col-major + (uplo, trans) as row-major with both flipped. */
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
        if (!upper) {
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
