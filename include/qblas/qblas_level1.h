/* QBLAS Level 1 — vector operations.
 * Names follow the BLAS convention with a 'q' prefix for the quad family,
 * matching OpenBLAS's xdouble (`q`) naming (cblas_qdot, cblas_qaxpy, ...).
 *
 * All vectors are arrays of `Sleef_quad` (128-bit). Stride parameters (`incx`,
 * `incy`) follow Fortran BLAS semantics: positive strides walk forward; a
 * stride of 0 is undefined behaviour; negative strides are accepted and walk
 * the array in reverse (the canonical BLAS interpretation — element k is at
 * `x[(n-1-k)*(-incx)]`).
 */
#ifndef QBLAS_LEVEL1_H
#define QBLAS_LEVEL1_H

#ifndef QBLAS_H
#  include "qblas.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* dot:  result = sum_i x[i] * y[i] */
QBLAS_API Sleef_quad cblas_qdot(int n,
                                const Sleef_quad *x, int incx,
                                const Sleef_quad *y, int incy);

/* nrm2: result = sqrt(sum_i x[i]^2)  (single accurate sweep, no overflow guard) */
QBLAS_API Sleef_quad cblas_qnrm2(int n, const Sleef_quad *x, int incx);

/* asum: result = sum_i |x[i]| */
QBLAS_API Sleef_quad cblas_qasum(int n, const Sleef_quad *x, int incx);

/* iamax: index of element with max |x[i]|  (0-based, like CBLAS) */
QBLAS_API size_t cblas_iqamax(int n, const Sleef_quad *x, int incx);

/* axpy: y[i] := alpha * x[i] + y[i] */
QBLAS_API void cblas_qaxpy(int n, Sleef_quad alpha,
                           const Sleef_quad *x, int incx,
                           Sleef_quad *y,       int incy);

/* scal: x[i] := alpha * x[i] */
QBLAS_API void cblas_qscal(int n, Sleef_quad alpha,
                           Sleef_quad *x, int incx);

/* copy: y[i] := x[i] */
QBLAS_API void cblas_qcopy(int n,
                           const Sleef_quad *x, int incx,
                           Sleef_quad *y,       int incy);

/* swap: x[i] <-> y[i] */
QBLAS_API void cblas_qswap(int n,
                           Sleef_quad *x, int incx,
                           Sleef_quad *y, int incy);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* QBLAS_LEVEL1_H */
