/* Shared helpers for the qblas test suite. */
#ifndef QBLAS_TEST_HELPERS_H
#define QBLAS_TEST_HELPERS_H

#include <qblas/qblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned int g_seed = 12345u;

static inline unsigned int rnd(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed;
}
static inline double frand(double lo, double hi) {
    return lo + (hi - lo) * (rnd() / (double)0xFFFFFFFFu);
}

static inline Sleef_quad qd(double x) { return Sleef_cast_from_doubleq1(x); }
static inline double     dq(Sleef_quad x) { return (double)Sleef_cast_to_doubleq1(x); }

static inline Sleef_quad q_add(Sleef_quad a, Sleef_quad b) { return Sleef_addq1_u05(a, b); }
static inline Sleef_quad q_sub(Sleef_quad a, Sleef_quad b) { return Sleef_subq1_u05(a, b); }
static inline Sleef_quad q_mul(Sleef_quad a, Sleef_quad b) { return Sleef_mulq1_u05(a, b); }
static inline Sleef_quad q_fma(Sleef_quad a, Sleef_quad b, Sleef_quad c) {
    return Sleef_fmaq1_u05(a, b, c);
}
static inline Sleef_quad q_abs(Sleef_quad a) { return Sleef_fabsq1(a); }
static inline int        q_lt (Sleef_quad a, Sleef_quad b) { return Sleef_icmpltq1(a, b); }
static inline int        q_eq (Sleef_quad a, Sleef_quad b) { return Sleef_icmpeqq1(a, b); }

static inline void fill_vec(Sleef_quad *v, int n, int stride) {
    for (int i = 0; i < n; ++i) v[(size_t)i * stride] = qd(frand(-1.0, 1.0));
}
static inline void fill_mat(Sleef_quad *A, int rows, int cols, int ld) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A[(size_t)i * ld + j] = qd(frand(-1.0, 1.0));
}

/* Max relative error across two quad arrays, falling back to absolute
 * diff when both values are essentially zero. */
static inline double max_rel_err(const Sleef_quad *a, const Sleef_quad *b,
                                 int n, int astride, int bstride) {
    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
        Sleef_quad av = a[(size_t)i * astride];
        Sleef_quad bv = b[(size_t)i * bstride];
        Sleef_quad d  = q_abs(q_sub(av, bv));
        Sleef_quad s  = q_add(q_abs(av), q_abs(bv));
        double dd = dq(d);
        double ss = dq(s);
        double r = (ss > 1e-20) ? (dd / ss) : dd;
        if (r > worst) worst = r;
    }
    return worst;
}

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond, ...) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fputc('\n', stderr); \
    } \
} while (0)

#define CHECK_NEAR(got, want, tol, what) do { \
    ++g_checks; \
    double _g = dq(got), _w = dq(want); \
    double _e = fabs(_g - _w); \
    if (_e > (tol) * (fabs(_w) + 1e-30)) { \
        ++g_failures; \
        fprintf(stderr, "FAIL %s:%d %s: got %.17g want %.17g err %.3e\n", \
                __FILE__, __LINE__, what, _g, _w, _e); \
    } \
} while (0)

#define CHECK_ARR(actual, expected, n, tol, what) do { \
    double _rel = max_rel_err((actual), (expected), (n), 1, 1); \
    ++g_checks; \
    if (_rel > (tol)) { \
        ++g_failures; \
        fprintf(stderr, "FAIL %s:%d %s: rel_err=%.3e (n=%d)\n", \
                __FILE__, __LINE__, what, _rel, (int)(n)); \
    } \
} while (0)

#define REPORT() do { \
    if (g_failures) { \
        fprintf(stderr, "%d of %d checks FAILED\n", g_failures, g_checks); \
        return 1; \
    } \
    printf("%d checks PASSED  (tier=%s, threads=%d)\n", \
           g_checks, qblas_get_dispatch_tier(), qblas_get_num_threads()); \
    return 0; \
} while (0)

#endif
