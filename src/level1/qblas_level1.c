/* Public Level 1 entry points.  Each routine validates inputs, decides whether
 * to thread the work, and delegates to the dispatch-table kernel.  Threading
 * for Level 1 is parallel-for over disjoint contiguous slices. */

#include "common/qblas_internal.h"
#include "common/qblas_dispatch.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

static inline ptrdiff_t neg_offset(int inc, int n) {
    return (inc < 0) ? (ptrdiff_t)(n - 1) * (ptrdiff_t)(-inc) : 0;
}

/* ------------------------------------------------------------------ */
Sleef_quad cblas_qdot(int n,
                      const Sleef_quad *x, int incx,
                      const Sleef_quad *y, int incy) {
    if (n <= 0) return QBLAS_ZERO;
    if (incx == 1 && incy == 1) {
        int nthreads = qblas_resolve_threads((size_t)n, 1);
        if (nthreads > 1 && n >= qblas_tune()->l1_thread_threshold) {
#ifdef _OPENMP
            Sleef_quad totals[256] = {0};
            if (nthreads > 256) nthreads = 256;
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nt  = omp_get_num_threads();
                size_t chunk = (size_t)n / nt;
                size_t rem   = (size_t)n % nt;
                size_t start = (size_t)tid * chunk + (tid < (int)rem ? (size_t)tid : rem);
                size_t cnt   = chunk + (tid < (int)rem ? 1u : 0u);
                totals[tid] = qblas_dispatch_qdot(cnt, x + start, 1, y + start, 1);
            }
            Sleef_quad acc = QBLAS_ZERO;
            for (int t = 0; t < nthreads; ++t) acc = qadd(acc, totals[t]);
            return acc;
#endif
        }
        return qblas_dispatch_qdot((size_t)n, x, 1, y, 1);
    }
    /* Strided: rebase pointers for negative strides so the kernel doesn't
     * have to. */
    ptrdiff_t ox = neg_offset(incx, n);
    ptrdiff_t oy = neg_offset(incy, n);
    return qblas_dispatch_qdot((size_t)n, x + ox, (ptrdiff_t)incx,
                                            y + oy, (ptrdiff_t)incy);
}

/* ------------------------------------------------------------------ */
Sleef_quad cblas_qnrm2(int n, const Sleef_quad *x, int incx) {
    if (n <= 0) return QBLAS_ZERO;
    /* dot(x,x) is sufficient — no overflow guard for now; quad range
     * covers ~10^4932 so true overflow is extremely rare in practice. */
    Sleef_quad s = cblas_qdot(n, x, incx, x, incx);
    return qsqrt(s);
}

/* ------------------------------------------------------------------ */
Sleef_quad cblas_qasum(int n, const Sleef_quad *x, int incx) {
    if (n <= 0) return QBLAS_ZERO;
    if (incx == 1) {
        int nthreads = qblas_resolve_threads((size_t)n, 1);
        if (nthreads > 1 && n >= qblas_tune()->l1_thread_threshold) {
#ifdef _OPENMP
            Sleef_quad totals[256] = {0};
            if (nthreads > 256) nthreads = 256;
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nt  = omp_get_num_threads();
                size_t chunk = (size_t)n / nt;
                size_t rem   = (size_t)n % nt;
                size_t start = (size_t)tid * chunk + (tid < (int)rem ? (size_t)tid : rem);
                size_t cnt   = chunk + (tid < (int)rem ? 1u : 0u);
                totals[tid] = qblas_dispatch_qasum(cnt, x + start, 1);
            }
            Sleef_quad acc = QBLAS_ZERO;
            for (int t = 0; t < nthreads; ++t) acc = qadd(acc, totals[t]);
            return acc;
#endif
        }
        return qblas_dispatch_qasum((size_t)n, x, 1);
    }
    ptrdiff_t ox = neg_offset(incx, n);
    return qblas_dispatch_qasum((size_t)n, x + ox, (ptrdiff_t)incx);
}

/* ------------------------------------------------------------------ */
size_t cblas_iqamax(int n, const Sleef_quad *x, int incx) {
    if (n <= 0) return 0;
    ptrdiff_t ox = neg_offset(incx, n);
    return qblas_dispatch_qiamax((size_t)n, x + ox, (ptrdiff_t)incx);
}

/* ------------------------------------------------------------------ */
void cblas_qaxpy(int n, Sleef_quad alpha,
                 const Sleef_quad *x, int incx,
                 Sleef_quad *y,       int incy) {
    if (n <= 0 || qiszero(alpha)) return;
    if (incx == 1 && incy == 1) {
        int nthreads = qblas_resolve_threads((size_t)n, 2);
        if (nthreads > 1 && n >= qblas_tune()->l1_thread_threshold) {
#ifdef _OPENMP
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nt  = omp_get_num_threads();
                size_t chunk = (size_t)n / nt;
                size_t rem   = (size_t)n % nt;
                size_t start = (size_t)tid * chunk + (tid < (int)rem ? (size_t)tid : rem);
                size_t cnt   = chunk + (tid < (int)rem ? 1u : 0u);
                qblas_dispatch_qaxpy(cnt, alpha, x + start, 1, y + start, 1);
            }
            return;
#endif
        }
        qblas_dispatch_qaxpy((size_t)n, alpha, x, 1, y, 1);
        return;
    }
    ptrdiff_t ox = neg_offset(incx, n);
    ptrdiff_t oy = neg_offset(incy, n);
    qblas_dispatch_qaxpy((size_t)n, alpha, x + ox, incx, y + oy, incy);
}

/* ------------------------------------------------------------------ */
void cblas_qscal(int n, Sleef_quad alpha, Sleef_quad *x, int incx) {
    if (n <= 0) return;
    if (incx == 1) {
        int nthreads = qblas_resolve_threads((size_t)n, 1);
        if (nthreads > 1 && n >= qblas_tune()->l1_thread_threshold) {
#ifdef _OPENMP
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nt  = omp_get_num_threads();
                size_t chunk = (size_t)n / nt;
                size_t rem   = (size_t)n % nt;
                size_t start = (size_t)tid * chunk + (tid < (int)rem ? (size_t)tid : rem);
                size_t cnt   = chunk + (tid < (int)rem ? 1u : 0u);
                qblas_dispatch_qscal(cnt, alpha, x + start, 1);
            }
            return;
#endif
        }
        qblas_dispatch_qscal((size_t)n, alpha, x, 1);
        return;
    }
    ptrdiff_t ox = neg_offset(incx, n);
    qblas_dispatch_qscal((size_t)n, alpha, x + ox, incx);
}

/* ------------------------------------------------------------------ */
void cblas_qcopy(int n,
                 const Sleef_quad *x, int incx,
                 Sleef_quad *y,       int incy) {
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        memcpy(y, x, (size_t)n * sizeof(Sleef_quad));
        return;
    }
    ptrdiff_t ix = neg_offset(incx, n);
    ptrdiff_t iy = neg_offset(incy, n);
    for (int i = 0; i < n; ++i) {
        y[iy] = x[ix];
        ix += incx;
        iy += incy;
    }
}

/* ------------------------------------------------------------------ */
void cblas_qswap(int n,
                 Sleef_quad *x, int incx,
                 Sleef_quad *y, int incy) {
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; ++i) {
            Sleef_quad t = x[i]; x[i] = y[i]; y[i] = t;
        }
        return;
    }
    ptrdiff_t ix = neg_offset(incx, n);
    ptrdiff_t iy = neg_offset(incy, n);
    for (int i = 0; i < n; ++i) {
        Sleef_quad t = x[ix]; x[ix] = y[iy]; y[iy] = t;
        ix += incx;
        iy += incy;
    }
}
