/* QBLAS kernel dispatch tables.
 *
 * Each Level 1/2/3 routine that has per-ISA specialisations declares a
 * function pointer here.  At library init `qblas_dispatch_init()` populates
 * each pointer with the best implementation available on the host CPU,
 * falling back to the generic version when no SIMD-capable kernel exists.
 *
 * Pointers are written exactly once (during init) and read lock-free
 * thereafter, so there is no synchronisation cost on the hot path.
 */
#ifndef QBLAS_DISPATCH_H
#define QBLAS_DISPATCH_H

#include "qblas_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----- Level 1 ----- */
typedef Sleef_quad (*qdot_fn)(size_t n, const Sleef_quad *x, ptrdiff_t incx,
                              const Sleef_quad *y, ptrdiff_t incy);
typedef void (*qaxpy_fn)(size_t n, Sleef_quad alpha,
                         const Sleef_quad *x, ptrdiff_t incx,
                         Sleef_quad *y,       ptrdiff_t incy);
typedef void (*qscal_fn)(size_t n, Sleef_quad alpha,
                         Sleef_quad *x, ptrdiff_t incx);
typedef Sleef_quad (*qasum_fn)(size_t n, const Sleef_quad *x, ptrdiff_t incx);
typedef size_t     (*qiamax_fn)(size_t n, const Sleef_quad *x, ptrdiff_t incx);

extern qdot_fn   qblas_dispatch_qdot;
extern qaxpy_fn  qblas_dispatch_qaxpy;
extern qscal_fn  qblas_dispatch_qscal;
extern qasum_fn  qblas_dispatch_qasum;
extern qiamax_fn qblas_dispatch_qiamax;

/* ----- Level 2 ----- */
/* gemv_n: y := alpha * A * x + beta * y, A is m x k, contiguous-row stride lda. */
typedef void (*qgemv_n_fn)(size_t m, size_t k,
                           Sleef_quad alpha,
                           const Sleef_quad *A, size_t lda,
                           const Sleef_quad *x, ptrdiff_t incx,
                           Sleef_quad beta,
                           Sleef_quad *y, ptrdiff_t incy);
/* gemv_t: y := alpha * A^T * x + beta * y, A is m x k stored row-stride lda. */
typedef void (*qgemv_t_fn)(size_t m, size_t k,
                           Sleef_quad alpha,
                           const Sleef_quad *A, size_t lda,
                           const Sleef_quad *x, ptrdiff_t incx,
                           Sleef_quad beta,
                           Sleef_quad *y, ptrdiff_t incy);

extern qgemv_n_fn qblas_dispatch_qgemv_n;
extern qgemv_t_fn qblas_dispatch_qgemv_t;

/* ----- Level 3 (GEMM packed micro-kernel) ----- */
/* qgemm_kernel: accumulate C(mr x nr) += alpha * A_packed(mr x kc) * B_packed(kc x nr)
 * with beta-scaling of C handled by caller (kernel writes via FMA). */
typedef void (*qgemm_kernel_fn)(size_t kc,
                                Sleef_quad alpha,
                                const Sleef_quad *A_packed,
                                const Sleef_quad *B_packed,
                                Sleef_quad *C, size_t ldc);

extern qgemm_kernel_fn qblas_dispatch_qgemm_kernel; /* MR x NR fixed by tier */
extern size_t qblas_dispatch_qgemm_MR;
extern size_t qblas_dispatch_qgemm_NR;

void qblas_dispatch_init(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* QBLAS_DISPATCH_H */
