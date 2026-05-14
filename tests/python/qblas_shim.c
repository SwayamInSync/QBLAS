/* Pointer-based shim around the by-value-Sleef_quad qblas entry points.
 *
 * Python's ctypes doesn't reliably honour SysV AMD64's rule that a
 * 16-byte struct made of two 8-byte integer fields is passed in two
 * integer registers.  In practice we see bus errors when ctypes pushes
 * Sleef_quad scalars on the stack mis-aligned.
 *
 * Each shim just dereferences the Quad-scalar pointers and forwards.
 * Public signature stays C ABI standard and only takes pointers / ints /
 * void pointers, which ctypes handles correctly across libffi versions.
 */
#include <qblas/qblas.h>

#define EXP __attribute__((visibility("default")))

EXP void shim_qaxpy(int n, const Sleef_quad *alpha,
                    const void *x, int incx, void *y, int incy) {
    cblas_qaxpy(n, *alpha, (const Sleef_quad *)x, incx,
                (Sleef_quad *)y, incy);
}
EXP void shim_qscal(int n, const Sleef_quad *alpha, void *x, int incx) {
    cblas_qscal(n, *alpha, (Sleef_quad *)x, incx);
}

EXP void shim_qgemv(int layout, int trans, int m, int n,
                    const Sleef_quad *alpha,
                    const void *A, int lda, const void *x, int incx,
                    const Sleef_quad *beta, void *y, int incy) {
    cblas_qgemv((QBLAS_LAYOUT)layout, (QBLAS_TRANSPOSE)trans, m, n,
                *alpha, (const Sleef_quad *)A, lda,
                (const Sleef_quad *)x, incx, *beta,
                (Sleef_quad *)y, incy);
}

EXP void shim_qger(int layout, int m, int n,
                   const Sleef_quad *alpha,
                   const void *x, int incx, const void *y, int incy,
                   void *A, int lda) {
    cblas_qger((QBLAS_LAYOUT)layout, m, n, *alpha,
               (const Sleef_quad *)x, incx,
               (const Sleef_quad *)y, incy,
               (Sleef_quad *)A, lda);
}

EXP void shim_qgemm(int layout, int ta, int tb, int m, int n, int k,
                    const Sleef_quad *alpha,
                    const void *A, int lda, const void *B, int ldb,
                    const Sleef_quad *beta, void *C, int ldc) {
    cblas_qgemm((QBLAS_LAYOUT)layout, (QBLAS_TRANSPOSE)ta,
                (QBLAS_TRANSPOSE)tb, m, n, k,
                *alpha, (const Sleef_quad *)A, lda,
                (const Sleef_quad *)B, ldb, *beta,
                (Sleef_quad *)C, ldc);
}

EXP void shim_qsyrk(int layout, int uplo, int trans, int n, int k,
                    const Sleef_quad *alpha,
                    const void *A, int lda,
                    const Sleef_quad *beta, void *C, int ldc) {
    cblas_qsyrk((QBLAS_LAYOUT)layout, (QBLAS_UPLO)uplo,
                (QBLAS_TRANSPOSE)trans, n, k,
                *alpha, (const Sleef_quad *)A, lda,
                *beta, (Sleef_quad *)C, ldc);
}

EXP void shim_qtrmm(int layout, int side, int uplo, int trans, int diag,
                    int m, int n, const Sleef_quad *alpha,
                    const void *A, int lda, void *B, int ldb) {
    cblas_qtrmm((QBLAS_LAYOUT)layout, (QBLAS_SIDE)side, (QBLAS_UPLO)uplo,
                (QBLAS_TRANSPOSE)trans, (QBLAS_DIAG)diag, m, n,
                *alpha, (const Sleef_quad *)A, lda,
                (Sleef_quad *)B, ldb);
}

EXP void shim_qtrsm(int layout, int side, int uplo, int trans, int diag,
                    int m, int n, const Sleef_quad *alpha,
                    const void *A, int lda, void *B, int ldb) {
    cblas_qtrsm((QBLAS_LAYOUT)layout, (QBLAS_SIDE)side, (QBLAS_UPLO)uplo,
                (QBLAS_TRANSPOSE)trans, (QBLAS_DIAG)diag, m, n,
                *alpha, (const Sleef_quad *)A, lda,
                (Sleef_quad *)B, ldb);
}

/* qdot returns Sleef_quad by value; write to an out pointer instead. */
EXP void shim_qdot(int n, const void *x, int incx,
                   const void *y, int incy, Sleef_quad *out) {
    *out = cblas_qdot(n, (const Sleef_quad *)x, incx,
                          (const Sleef_quad *)y, incy);
}
EXP void shim_qnrm2(int n, const void *x, int incx, Sleef_quad *out) {
    *out = cblas_qnrm2(n, (const Sleef_quad *)x, incx);
}
EXP void shim_qasum(int n, const void *x, int incx, Sleef_quad *out) {
    *out = cblas_qasum(n, (const Sleef_quad *)x, incx);
}
EXP size_t shim_iqamax(int n, const void *x, int incx) {
    return cblas_iqamax(n, (const Sleef_quad *)x, incx);
}

/* Constructors / casts to populate Sleef_quad without going through ctypes
 * by-value returns either. */
EXP void shim_d2q(double v, Sleef_quad *out) {
    *out = Sleef_cast_from_doubleq1(v);
}
EXP double shim_q2d(const Sleef_quad *in) {
    return (double)Sleef_cast_to_doubleq1(*in);
}
