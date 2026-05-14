/* Kernel template instantiated once per ISA tier.  The including TU sets:
 *   QV_WIDTH         1, 2, 4, or 8         vector width in quads
 *   QV_SUFFIX        suffix for symbol names (generic, sse2, avx2, ...)
 *   QV_ISA_SUFFIX    SLEEF ISA suffix for width-2 (sse2 or advsimd) */

#ifndef QV_WIDTH
#  error "kernels_template.h: define QV_WIDTH (1|2|4|8) and QV_SUFFIX first"
#endif
#ifndef QV_SUFFIX
#  error "kernels_template.h: define QV_SUFFIX before include"
#endif

#include "common/qblas_internal.h"
#include "common/qblas_dispatch.h"

#include <sleefquad.h>
#include <stddef.h>

#define QV_PASTE_(a, b) a##_##b
#define QV_PASTE(a, b)  QV_PASTE_(a, b)
#define QV_FN(name)     QV_PASTE(QV_PASTE(qblas, name), QV_SUFFIX)

#if QV_WIDTH == 1
   typedef Sleef_quad qv_t;
#  define QV_ADD(a,b)     Sleef_addq1_u05((a),(b))
#  define QV_FMA(a,b,c)   Sleef_fmaq1_u05((a),(b),(c))
#  define QV_MUL(a,b)     Sleef_mulq1_u05((a),(b))
#  define QV_SPLAT(s)     (s)
#  define QV_LOADU(p)     (*(p))
#  define QV_STOREU(p,v)  (*(p) = (v))
#  define QV_FABS(a)      Sleef_fabsq1((a))
#  define QV_LANE(v, i)   (v)
#elif QV_WIDTH == 2
#  ifndef QV_ISA_SUFFIX
#    error "width-2 kernels must define QV_ISA_SUFFIX (sse2 or advsimd)"
#  endif
#  define QV_NAME_(op,ulp,isa) Sleef_ ## op ## q2 ## ulp ## isa
#  define QV_NAME(op,ulp,isa)  QV_NAME_(op,ulp,isa)
   typedef Sleef_quadx2 qv_t;
#  define QV_ADD(a,b)     QV_NAME(add, _u05, QV_ISA_SUFFIX)((a),(b))
#  define QV_FMA(a,b,c)   QV_NAME(fma, _u05, QV_ISA_SUFFIX)((a),(b),(c))
#  define QV_MUL(a,b)     QV_NAME(mul, _u05, QV_ISA_SUFFIX)((a),(b))
#  define QV_SPLAT(s)     QV_NAME(splat, _, QV_ISA_SUFFIX)((s))
#  define QV_LOADU(p)     QV_NAME(load,  _, QV_ISA_SUFFIX)((Sleef_quad *)(p))
#  define QV_STOREU(p,v)  QV_NAME(store, _, QV_ISA_SUFFIX)((Sleef_quad *)(p), (v))
#  define QV_FABS(a)      QV_NAME(fabs,  _, QV_ISA_SUFFIX)((a))
#  define QV_LANE(v, i)   QV_NAME(get,   _, QV_ISA_SUFFIX)((v), (i))
#elif QV_WIDTH == 4
   typedef Sleef_quadx4 qv_t;
#  define QV_ADD(a,b)     Sleef_addq4_u05avx2((a),(b))
#  define QV_FMA(a,b,c)   Sleef_fmaq4_u05avx2((a),(b),(c))
#  define QV_MUL(a,b)     Sleef_mulq4_u05avx2((a),(b))
#  define QV_SPLAT(s)     Sleef_splatq4_avx2((s))
#  define QV_LOADU(p)     Sleef_loadq4_avx2((Sleef_quad *)(p))
#  define QV_STOREU(p,v)  Sleef_storeq4_avx2((Sleef_quad *)(p), (v))
#  define QV_FABS(a)      Sleef_fabsq4_avx2((a))
#  define QV_LANE(v, i)   Sleef_getq4_avx2((v), (i))
#elif QV_WIDTH == 8
   typedef Sleef_quadx8 qv_t;
#  define QV_ADD(a,b)     Sleef_addq8_u05avx512f((a),(b))
#  define QV_FMA(a,b,c)   Sleef_fmaq8_u05avx512f((a),(b),(c))
#  define QV_MUL(a,b)     Sleef_mulq8_u05avx512f((a),(b))
#  define QV_SPLAT(s)     Sleef_splatq8_avx512f((s))
#  define QV_LOADU(p)     Sleef_loadq8_avx512f((Sleef_quad *)(p))
#  define QV_STOREU(p,v)  Sleef_storeq8_avx512f((Sleef_quad *)(p), (v))
#  define QV_FABS(a)      Sleef_fabsq8_avx512f((a))
#  define QV_LANE(v, i)   Sleef_getq8_avx512f((v), (i))
#else
#  error "QV_WIDTH must be 1, 2, 4, or 8"
#endif

/* dot */
static Sleef_quad QV_FN(qdot)(size_t n,
                              const Sleef_quad *x, ptrdiff_t incx,
                              const Sleef_quad *y, ptrdiff_t incy) {
    Sleef_quad acc = QBLAS_ZERO;

    if (incx == 1 && incy == 1) {
#if QV_WIDTH > 1
        /* Four independent accumulators to overlap quad FMA latency. */
        const size_t W = (size_t)QV_WIDTH;
        const size_t UNROLL = 4;
        const size_t step = W * UNROLL;
        qv_t a0 = QV_SPLAT(QBLAS_ZERO);
        qv_t a1 = QV_SPLAT(QBLAS_ZERO);
        qv_t a2 = QV_SPLAT(QBLAS_ZERO);
        qv_t a3 = QV_SPLAT(QBLAS_ZERO);

        size_t i = 0;
        for (; i + step <= n; i += step) {
            qv_t xv0 = QV_LOADU(x + i + 0 * W);
            qv_t yv0 = QV_LOADU(y + i + 0 * W);
            qv_t xv1 = QV_LOADU(x + i + 1 * W);
            qv_t yv1 = QV_LOADU(y + i + 1 * W);
            qv_t xv2 = QV_LOADU(x + i + 2 * W);
            qv_t yv2 = QV_LOADU(y + i + 2 * W);
            qv_t xv3 = QV_LOADU(x + i + 3 * W);
            qv_t yv3 = QV_LOADU(y + i + 3 * W);
            a0 = QV_FMA(xv0, yv0, a0);
            a1 = QV_FMA(xv1, yv1, a1);
            a2 = QV_FMA(xv2, yv2, a2);
            a3 = QV_FMA(xv3, yv3, a3);
        }
        for (; i + W <= n; i += W) {
            qv_t xv = QV_LOADU(x + i);
            qv_t yv = QV_LOADU(y + i);
            a0 = QV_FMA(xv, yv, a0);
        }
        a0 = QV_ADD(a0, a2);
        a1 = QV_ADD(a1, a3);
        a0 = QV_ADD(a0, a1);
        for (size_t lane = 0; lane < W; ++lane)
            acc = qadd(acc, QV_LANE(a0, (int)lane));
        for (; i < n; ++i)
            acc = qfma(x[i], y[i], acc);
#else
        for (size_t i = 0; i < n; ++i)
            acc = qfma(x[i], y[i], acc);
#endif
    } else {
        /* Caller pre-shifts x/y to the logical first element so signed
         * strides can start at zero here. */
        ptrdiff_t ix = 0, iy = 0;
        for (size_t i = 0; i < n; ++i) {
            acc = qfma(x[ix], y[iy], acc);
            ix += incx;
            iy += incy;
        }
    }
    return acc;
}

/* axpy */
static void QV_FN(qaxpy)(size_t n, Sleef_quad alpha,
                         const Sleef_quad *x, ptrdiff_t incx,
                         Sleef_quad *y,       ptrdiff_t incy) {
    if (qiszero(alpha)) return;

    if (incx == 1 && incy == 1) {
#if QV_WIDTH > 1
        const size_t W = (size_t)QV_WIDTH;
        const qv_t va = QV_SPLAT(alpha);
        const size_t UNROLL = 4;
        const size_t step = W * UNROLL;
        size_t i = 0;
        for (; i + step <= n; i += step) {
            qv_t x0 = QV_LOADU(x + i + 0 * W);
            qv_t x1 = QV_LOADU(x + i + 1 * W);
            qv_t x2 = QV_LOADU(x + i + 2 * W);
            qv_t x3 = QV_LOADU(x + i + 3 * W);
            qv_t y0 = QV_LOADU(y + i + 0 * W);
            qv_t y1 = QV_LOADU(y + i + 1 * W);
            qv_t y2 = QV_LOADU(y + i + 2 * W);
            qv_t y3 = QV_LOADU(y + i + 3 * W);
            y0 = QV_FMA(va, x0, y0);
            y1 = QV_FMA(va, x1, y1);
            y2 = QV_FMA(va, x2, y2);
            y3 = QV_FMA(va, x3, y3);
            QV_STOREU(y + i + 0 * W, y0);
            QV_STOREU(y + i + 1 * W, y1);
            QV_STOREU(y + i + 2 * W, y2);
            QV_STOREU(y + i + 3 * W, y3);
        }
        for (; i + W <= n; i += W) {
            qv_t xv = QV_LOADU(x + i);
            qv_t yv = QV_LOADU(y + i);
            yv = QV_FMA(va, xv, yv);
            QV_STOREU(y + i, yv);
        }
        for (; i < n; ++i)
            y[i] = qfma(alpha, x[i], y[i]);
#else
        for (size_t i = 0; i < n; ++i)
            y[i] = qfma(alpha, x[i], y[i]);
#endif
    } else {
        ptrdiff_t ix = 0, iy = 0;
        for (size_t i = 0; i < n; ++i) {
            y[iy] = qfma(alpha, x[ix], y[iy]);
            ix += incx;
            iy += incy;
        }
    }
}

/* scal */
static void QV_FN(qscal)(size_t n, Sleef_quad alpha, Sleef_quad *x, ptrdiff_t incx) {
    if (qisone(alpha)) return;
    if (qiszero(alpha)) {
        if (incx == 1) {
            for (size_t i = 0; i < n; ++i) x[i] = QBLAS_ZERO;
        } else {
            ptrdiff_t ix = 0;
            for (size_t i = 0; i < n; ++i) { x[ix] = QBLAS_ZERO; ix += incx; }
        }
        return;
    }

    if (incx == 1) {
#if QV_WIDTH > 1
        const size_t W = (size_t)QV_WIDTH;
        const qv_t va = QV_SPLAT(alpha);
        size_t i = 0;
        const size_t UNROLL = 4;
        const size_t step = W * UNROLL;
        for (; i + step <= n; i += step) {
            qv_t x0 = QV_LOADU(x + i + 0 * W);
            qv_t x1 = QV_LOADU(x + i + 1 * W);
            qv_t x2 = QV_LOADU(x + i + 2 * W);
            qv_t x3 = QV_LOADU(x + i + 3 * W);
            QV_STOREU(x + i + 0 * W, QV_MUL(va, x0));
            QV_STOREU(x + i + 1 * W, QV_MUL(va, x1));
            QV_STOREU(x + i + 2 * W, QV_MUL(va, x2));
            QV_STOREU(x + i + 3 * W, QV_MUL(va, x3));
        }
        for (; i + W <= n; i += W) {
            qv_t xv = QV_LOADU(x + i);
            QV_STOREU(x + i, QV_MUL(va, xv));
        }
        for (; i < n; ++i) x[i] = qmul(alpha, x[i]);
#else
        for (size_t i = 0; i < n; ++i) x[i] = qmul(alpha, x[i]);
#endif
    } else {
        ptrdiff_t ix = 0;
        for (size_t i = 0; i < n; ++i) {
            x[ix] = qmul(alpha, x[ix]);
            ix += incx;
        }
    }
}

/* asum: sum of |x_i| */
static Sleef_quad QV_FN(qasum)(size_t n, const Sleef_quad *x, ptrdiff_t incx) {
    Sleef_quad acc = QBLAS_ZERO;
    if (incx == 1) {
#if QV_WIDTH > 1
        const size_t W = (size_t)QV_WIDTH;
        qv_t a0 = QV_SPLAT(QBLAS_ZERO);
        qv_t a1 = QV_SPLAT(QBLAS_ZERO);
        size_t i = 0;
        for (; i + 2 * W <= n; i += 2 * W) {
            qv_t v0 = QV_FABS(QV_LOADU(x + i));
            qv_t v1 = QV_FABS(QV_LOADU(x + i + W));
            a0 = QV_ADD(a0, v0);
            a1 = QV_ADD(a1, v1);
        }
        for (; i + W <= n; i += W) {
            qv_t v = QV_FABS(QV_LOADU(x + i));
            a0 = QV_ADD(a0, v);
        }
        a0 = QV_ADD(a0, a1);
        for (size_t lane = 0; lane < W; ++lane)
            acc = qadd(acc, QV_LANE(a0, (int)lane));
        for (; i < n; ++i) acc = qadd(acc, qfabs(x[i]));
#else
        for (size_t i = 0; i < n; ++i) acc = qadd(acc, qfabs(x[i]));
#endif
    } else {
        ptrdiff_t ix = 0;
        for (size_t i = 0; i < n; ++i) {
            acc = qadd(acc, qfabs(x[ix]));
            ix += incx;
        }
    }
    return acc;
}

/* iamax: scalar walk; memory-bound at the sizes we ship for. */
static size_t QV_FN(qiamax)(size_t n, const Sleef_quad *x, ptrdiff_t incx) {
    if (n == 0) return 0;
    Sleef_quad best = qfabs(x[0]);
    size_t arg = 0;
    if (incx == 1) {
        for (size_t i = 1; i < n; ++i) {
            Sleef_quad v = qfabs(x[i]);
            if (Sleef_icmpgtq1(v, best)) { best = v; arg = i; }
        }
    } else {
        ptrdiff_t ix = 0;
        best = qfabs(x[ix]); arg = 0;
        ix += incx;
        for (size_t i = 1; i < n; ++i) {
            Sleef_quad v = qfabs(x[ix]);
            if (Sleef_icmpgtq1(v, best)) { best = v; arg = i; }
            ix += incx;
        }
    }
    return arg;
}

/* gemv with no transpose: each output row is a dot of one A row with x. */
static void QV_FN(qgemv_n)(size_t m, size_t k,
                           Sleef_quad alpha,
                           const Sleef_quad *A, size_t lda,
                           const Sleef_quad *x, ptrdiff_t incx,
                           Sleef_quad beta,
                           Sleef_quad *y, ptrdiff_t incy) {
    for (size_t i = 0; i < m; ++i) {
        Sleef_quad s = QV_FN(qdot)(k, A + i * lda, 1, x, incx);
        if (qiszero(beta)) {
            y[(ptrdiff_t)i * incy] = qmul(alpha, s);
        } else {
            Sleef_quad prev = y[(ptrdiff_t)i * incy];
            y[(ptrdiff_t)i * incy] = qfma(alpha, s, qmul(beta, prev));
        }
    }
}

/* gemv with transpose: scale y by beta, then accumulate alpha*x[i]*A[i,:]. */
static void QV_FN(qgemv_t)(size_t m, size_t k,
                           Sleef_quad alpha,
                           const Sleef_quad *A, size_t lda,
                           const Sleef_quad *x, ptrdiff_t incx,
                           Sleef_quad beta,
                           Sleef_quad *y, ptrdiff_t incy) {
    if (qiszero(beta)) {
        for (size_t j = 0; j < k; ++j) y[(ptrdiff_t)j * incy] = QBLAS_ZERO;
    } else if (!qisone(beta)) {
        for (size_t j = 0; j < k; ++j)
            y[(ptrdiff_t)j * incy] = qmul(beta, y[(ptrdiff_t)j * incy]);
    }

    for (size_t i = 0; i < m; ++i) {
        Sleef_quad ax = qmul(alpha, x[(ptrdiff_t)i * incx]);
        if (qiszero(ax)) continue;
        if (incy == 1) {
            QV_FN(qaxpy)(k, ax, A + i * lda, 1, y, 1);
        } else {
            const Sleef_quad *Arow = A + i * lda;
            for (size_t j = 0; j < k; ++j)
                y[(ptrdiff_t)j * incy] = qfma(ax, Arow[j], y[(ptrdiff_t)j * incy]);
        }
    }
}

/* Packed GEMM micro-kernel: C(MR×NR) += alpha * A_packed(MR×kc) * B_packed(kc×NR).
 * A_packed has MR scalars per k step; B_packed has NR scalars per k step. */
#if QV_WIDTH == 1
#  define QV_MR 4
#  define QV_NR 4
#elif QV_WIDTH == 2
#  define QV_MR 4
#  define QV_NR 4
#elif QV_WIDTH == 4
#  define QV_MR 4
#  define QV_NR 4
#elif QV_WIDTH == 8
#  define QV_MR 4
#  define QV_NR 8
#endif

static void QV_FN(qgemm_kernel)(size_t kc,
                                Sleef_quad alpha,
                                const Sleef_quad *A_packed,
                                const Sleef_quad *B_packed,
                                Sleef_quad *C, size_t ldc) {
#if QV_WIDTH > 1
    enum { W = QV_WIDTH, NR_VEC = QV_NR / QV_WIDTH };
    qv_t c00 = QV_SPLAT(QBLAS_ZERO), c01 = QV_SPLAT(QBLAS_ZERO);
    qv_t c10 = QV_SPLAT(QBLAS_ZERO), c11 = QV_SPLAT(QBLAS_ZERO);
    qv_t c20 = QV_SPLAT(QBLAS_ZERO), c21 = QV_SPLAT(QBLAS_ZERO);
    qv_t c30 = QV_SPLAT(QBLAS_ZERO), c31 = QV_SPLAT(QBLAS_ZERO);
#if QV_MR >= 8
    qv_t c40 = QV_SPLAT(QBLAS_ZERO), c41 = QV_SPLAT(QBLAS_ZERO);
    qv_t c50 = QV_SPLAT(QBLAS_ZERO), c51 = QV_SPLAT(QBLAS_ZERO);
    qv_t c60 = QV_SPLAT(QBLAS_ZERO), c61 = QV_SPLAT(QBLAS_ZERO);
    qv_t c70 = QV_SPLAT(QBLAS_ZERO), c71 = QV_SPLAT(QBLAS_ZERO);
    (void)c41; (void)c51; (void)c61; (void)c71;
#endif
    (void)c01; (void)c11; (void)c21; (void)c31;

    for (size_t p = 0; p < kc; ++p) {
        qv_t b0 = QV_LOADU(B_packed + p * QV_NR + 0 * W);
        qv_t va0 = QV_SPLAT(A_packed[p * QV_MR + 0]);
        qv_t va1 = QV_SPLAT(A_packed[p * QV_MR + 1]);
        qv_t va2 = QV_SPLAT(A_packed[p * QV_MR + 2]);
        qv_t va3 = QV_SPLAT(A_packed[p * QV_MR + 3]);
        c00 = QV_FMA(va0, b0, c00);
        c10 = QV_FMA(va1, b0, c10);
        c20 = QV_FMA(va2, b0, c20);
        c30 = QV_FMA(va3, b0, c30);
#if QV_MR >= 8
        qv_t va4 = QV_SPLAT(A_packed[p * QV_MR + 4]);
        qv_t va5 = QV_SPLAT(A_packed[p * QV_MR + 5]);
        qv_t va6 = QV_SPLAT(A_packed[p * QV_MR + 6]);
        qv_t va7 = QV_SPLAT(A_packed[p * QV_MR + 7]);
        c40 = QV_FMA(va4, b0, c40);
        c50 = QV_FMA(va5, b0, c50);
        c60 = QV_FMA(va6, b0, c60);
        c70 = QV_FMA(va7, b0, c70);
#endif
        if (NR_VEC > 1) {
            qv_t b1 = QV_LOADU(B_packed + p * QV_NR + 1 * W);
            c01 = QV_FMA(va0, b1, c01);
            c11 = QV_FMA(va1, b1, c11);
            c21 = QV_FMA(va2, b1, c21);
            c31 = QV_FMA(va3, b1, c31);
        }
    }

    qv_t va = QV_SPLAT(alpha);
    QV_STOREU(C + 0 * ldc + 0 * W, QV_FMA(va, c00, QV_LOADU(C + 0 * ldc + 0 * W)));
    QV_STOREU(C + 1 * ldc + 0 * W, QV_FMA(va, c10, QV_LOADU(C + 1 * ldc + 0 * W)));
    QV_STOREU(C + 2 * ldc + 0 * W, QV_FMA(va, c20, QV_LOADU(C + 2 * ldc + 0 * W)));
    QV_STOREU(C + 3 * ldc + 0 * W, QV_FMA(va, c30, QV_LOADU(C + 3 * ldc + 0 * W)));
#if QV_MR >= 8
    QV_STOREU(C + 4 * ldc + 0 * W, QV_FMA(va, c40, QV_LOADU(C + 4 * ldc + 0 * W)));
    QV_STOREU(C + 5 * ldc + 0 * W, QV_FMA(va, c50, QV_LOADU(C + 5 * ldc + 0 * W)));
    QV_STOREU(C + 6 * ldc + 0 * W, QV_FMA(va, c60, QV_LOADU(C + 6 * ldc + 0 * W)));
    QV_STOREU(C + 7 * ldc + 0 * W, QV_FMA(va, c70, QV_LOADU(C + 7 * ldc + 0 * W)));
#endif
    if (NR_VEC > 1) {
        QV_STOREU(C + 0 * ldc + 1 * W, QV_FMA(va, c01, QV_LOADU(C + 0 * ldc + 1 * W)));
        QV_STOREU(C + 1 * ldc + 1 * W, QV_FMA(va, c11, QV_LOADU(C + 1 * ldc + 1 * W)));
        QV_STOREU(C + 2 * ldc + 1 * W, QV_FMA(va, c21, QV_LOADU(C + 2 * ldc + 1 * W)));
        QV_STOREU(C + 3 * ldc + 1 * W, QV_FMA(va, c31, QV_LOADU(C + 3 * ldc + 1 * W)));
    }
#else
    Sleef_quad c[QV_MR][QV_NR];
    for (size_t i = 0; i < QV_MR; ++i)
        for (size_t j = 0; j < QV_NR; ++j)
            c[i][j] = QBLAS_ZERO;
    for (size_t p = 0; p < kc; ++p) {
        for (size_t i = 0; i < QV_MR; ++i) {
            Sleef_quad a = A_packed[p * QV_MR + i];
            for (size_t j = 0; j < QV_NR; ++j)
                c[i][j] = qfma(a, B_packed[p * QV_NR + j], c[i][j]);
        }
    }
    for (size_t i = 0; i < QV_MR; ++i)
        for (size_t j = 0; j < QV_NR; ++j)
            C[i * ldc + j] = qfma(alpha, c[i][j], C[i * ldc + j]);
#endif
}

void QV_FN(register_kernels)(void);
void QV_FN(register_kernels)(void) {
    qblas_dispatch_qdot          = QV_FN(qdot);
    qblas_dispatch_qaxpy         = QV_FN(qaxpy);
    qblas_dispatch_qscal         = QV_FN(qscal);
    qblas_dispatch_qasum         = QV_FN(qasum);
    qblas_dispatch_qiamax        = QV_FN(qiamax);
    qblas_dispatch_qgemv_n       = QV_FN(qgemv_n);
    qblas_dispatch_qgemv_t       = QV_FN(qgemv_t);
    qblas_dispatch_qgemm_kernel  = QV_FN(qgemm_kernel);
    qblas_dispatch_qgemm_MR      = QV_MR;
    qblas_dispatch_qgemm_NR      = QV_NR;
}
