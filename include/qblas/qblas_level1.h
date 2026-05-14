/* Level 1: vector operations.  Negative strides are accepted and walk
 * the array in reverse per the BLAS convention; incx/incy of 0 are UB. */
#ifndef QBLAS_LEVEL1_H
#define QBLAS_LEVEL1_H

#ifndef QBLAS_H
#  include "qblas.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

QBLAS_API Sleef_quad cblas_qdot(int n,
                                const Sleef_quad *x, int incx,
                                const Sleef_quad *y, int incy);

QBLAS_API Sleef_quad cblas_qnrm2(int n, const Sleef_quad *x, int incx);
QBLAS_API Sleef_quad cblas_qasum(int n, const Sleef_quad *x, int incx);
QBLAS_API size_t     cblas_iqamax(int n, const Sleef_quad *x, int incx);

QBLAS_API void cblas_qaxpy(int n, Sleef_quad alpha,
                           const Sleef_quad *x, int incx,
                           Sleef_quad *y,       int incy);

QBLAS_API void cblas_qscal(int n, Sleef_quad alpha,
                           Sleef_quad *x, int incx);

QBLAS_API void cblas_qcopy(int n,
                           const Sleef_quad *x, int incx,
                           Sleef_quad *y,       int incy);

QBLAS_API void cblas_qswap(int n,
                           Sleef_quad *x, int incx,
                           Sleef_quad *y, int incy);

#ifdef __cplusplus
}
#endif

#endif
