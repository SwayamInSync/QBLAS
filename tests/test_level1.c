#include "test_helpers.h"

/* Naive scalar references. */
static Sleef_quad ref_dot(int n, const Sleef_quad *x, int ix,
                          const Sleef_quad *y, int iy) {
    int ox = (ix < 0) ? (n - 1) * (-ix) : 0;
    int oy = (iy < 0) ? (n - 1) * (-iy) : 0;
    Sleef_quad s = qd(0.0);
    for (int i = 0; i < n; ++i) {
        s = q_fma(x[ox + i * ix], y[oy + i * iy], s);
    }
    return s;
}
static Sleef_quad ref_asum(int n, const Sleef_quad *x, int ix) {
    int ox = (ix < 0) ? (n - 1) * (-ix) : 0;
    Sleef_quad s = qd(0.0);
    for (int i = 0; i < n; ++i) s = q_add(s, q_abs(x[ox + i * ix]));
    return s;
}
static int ref_iamax(int n, const Sleef_quad *x, int ix) {
    int ox = (ix < 0) ? (n - 1) * (-ix) : 0;
    Sleef_quad best = q_abs(x[ox]);
    int arg = 0;
    for (int i = 1; i < n; ++i) {
        Sleef_quad v = q_abs(x[ox + i * ix]);
        if (q_lt(best, v)) { best = v; arg = i; }
    }
    return arg;
}
static void ref_axpy(int n, Sleef_quad a,
                     const Sleef_quad *x, int ix,
                     Sleef_quad *y, int iy) {
    int ox = (ix < 0) ? (n - 1) * (-ix) : 0;
    int oy = (iy < 0) ? (n - 1) * (-iy) : 0;
    for (int i = 0; i < n; ++i)
        y[oy + i * iy] = q_fma(a, x[ox + i * ix], y[oy + i * iy]);
}
static void ref_scal(int n, Sleef_quad a, Sleef_quad *x, int ix) {
    int ox = (ix < 0) ? (n - 1) * (-ix) : 0;
    for (int i = 0; i < n; ++i) x[ox + i * ix] = q_mul(a, x[ox + i * ix]);
}

static void test_dot(int n, int incx, int incy) {
    int len_x = n * (incx < 0 ? -incx : incx);
    int len_y = n * (incy < 0 ? -incy : incy);
    Sleef_quad *x = (Sleef_quad *)malloc(len_x * sizeof(Sleef_quad));
    Sleef_quad *y = (Sleef_quad *)malloc(len_y * sizeof(Sleef_quad));
    fill_vec(x, len_x, 1);
    fill_vec(y, len_y, 1);

    Sleef_quad got  = cblas_qdot(n, x, incx, y, incy);
    Sleef_quad want = ref_dot(n, x, incx, y, incy);
    char tag[64]; snprintf(tag, sizeof tag, "dot n=%d incx=%d incy=%d", n, incx, incy);
    CHECK_NEAR(got, want, 1e-30, tag);

    free(x); free(y);
}

static void test_asum(int n, int incx) {
    int len_x = n * (incx < 0 ? -incx : incx);
    Sleef_quad *x = (Sleef_quad *)malloc(len_x * sizeof(Sleef_quad));
    fill_vec(x, len_x, 1);
    Sleef_quad got  = cblas_qasum(n, x, incx);
    Sleef_quad want = ref_asum(n, x, incx);
    char tag[64]; snprintf(tag, sizeof tag, "asum n=%d incx=%d", n, incx);
    CHECK_NEAR(got, want, 1e-30, tag);
    free(x);
}

static void test_iamax(int n, int incx) {
    int len_x = n * (incx < 0 ? -incx : incx);
    Sleef_quad *x = (Sleef_quad *)malloc(len_x * sizeof(Sleef_quad));
    fill_vec(x, len_x, 1);
    int got_ = (int)cblas_iqamax(n, x, incx);
    int want = ref_iamax(n, x, incx);
    char tag[64]; snprintf(tag, sizeof tag, "iamax n=%d incx=%d (got=%d want=%d)", n, incx, got_, want);
    CHECK(got_ == want, "%s", tag);
    free(x);
}

static void test_axpy(int n, int incx, int incy) {
    int len_x = n * (incx < 0 ? -incx : incx);
    int len_y = n * (incy < 0 ? -incy : incy);
    Sleef_quad *x = (Sleef_quad *)malloc(len_x * sizeof(Sleef_quad));
    Sleef_quad *y1 = (Sleef_quad *)malloc(len_y * sizeof(Sleef_quad));
    Sleef_quad *y2 = (Sleef_quad *)malloc(len_y * sizeof(Sleef_quad));
    fill_vec(x, len_x, 1);
    fill_vec(y1, len_y, 1);
    memcpy(y2, y1, len_y * sizeof(Sleef_quad));

    Sleef_quad alpha = qd(0.75);
    cblas_qaxpy(n, alpha, x, incx, y1, incy);
    ref_axpy   (n, alpha, x, incx, y2, incy);
    char tag[64]; snprintf(tag, sizeof tag, "axpy n=%d incx=%d incy=%d", n, incx, incy);
    CHECK_ARR(y1, y2, len_y, 1e-30, tag);

    free(x); free(y1); free(y2);
}

static void test_scal(int n, int incx) {
    int len_x = n * (incx < 0 ? -incx : incx);
    Sleef_quad *x1 = (Sleef_quad *)malloc(len_x * sizeof(Sleef_quad));
    Sleef_quad *x2 = (Sleef_quad *)malloc(len_x * sizeof(Sleef_quad));
    fill_vec(x1, len_x, 1);
    memcpy(x2, x1, len_x * sizeof(Sleef_quad));
    cblas_qscal(n, qd(-2.5), x1, incx);
    ref_scal   (n, qd(-2.5), x2, incx);
    char tag[64]; snprintf(tag, sizeof tag, "scal n=%d incx=%d", n, incx);
    CHECK_ARR(x1, x2, len_x, 1e-30, tag);
    free(x1); free(x2);
}

static void test_copy_swap(int n) {
    Sleef_quad *x  = (Sleef_quad *)malloc(n * sizeof(Sleef_quad));
    Sleef_quad *y  = (Sleef_quad *)malloc(n * sizeof(Sleef_quad));
    Sleef_quad *xs = (Sleef_quad *)malloc(n * sizeof(Sleef_quad));
    fill_vec(x, n, 1);
    fill_vec(y, n, 1);
    memcpy(xs, x, n * sizeof(Sleef_quad));

    Sleef_quad *yc = (Sleef_quad *)malloc(n * sizeof(Sleef_quad));
    cblas_qcopy(n, x, 1, yc, 1);
    CHECK_ARR(yc, x, n, 1e-30, "copy");

    cblas_qswap(n, x, 1, y, 1);
    Sleef_quad *yorig = (Sleef_quad *)malloc(n * sizeof(Sleef_quad));
    memcpy(yorig, yc, n * sizeof(Sleef_quad));
    CHECK_ARR(y, xs, n, 1e-30, "swap-y");

    free(yorig); free(yc); free(xs); free(x); free(y);
}

int main(void) {
    static int sizes[]   = { 1, 7, 16, 31, 64, 100, 1023, 8192, 100000 };
    static int strides[] = { 1, 2, 3, -1, -2 };

    for (size_t si = 0; si < sizeof sizes / sizeof sizes[0]; ++si) {
        int n = sizes[si];
        for (size_t a = 0; a < sizeof strides / sizeof strides[0]; ++a) {
            for (size_t b = 0; b < sizeof strides / sizeof strides[0]; ++b) {
                test_dot (n, strides[a], strides[b]);
                test_axpy(n, strides[a], strides[b]);
            }
            test_asum (n, strides[a]);
            test_scal (n, strides[a]);
            test_iamax(n, strides[a]);
        }
        test_copy_swap(n);
    }

    REPORT();
}
