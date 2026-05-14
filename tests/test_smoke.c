#include <qblas/qblas.h>
#include <stdio.h>
#include <stdlib.h>

static Sleef_quad qd(double x) { return Sleef_cast_from_doubleq1(x); }
static double     dq(Sleef_quad x) { return (double)Sleef_cast_to_doubleq1(x); }

int main(void) {
    printf("QBLAS smoke test\n");
    printf("  version : %s\n", qblas_get_version());
    printf("  tier    : %s\n", qblas_get_dispatch_tier());
    printf("  threads : %d\n", qblas_get_num_threads());

    enum { N = 8 };
    Sleef_quad x[N], y[N];
    for (int i = 0; i < N; ++i) {
        x[i] = qd((double)(i + 1));
        y[i] = qd(2.0);
    }
    Sleef_quad dot = cblas_qdot(N, x, 1, y, 1);
    printf("  dot     : %f (expected 72)\n", dq(dot));
    if (dq(dot) != 72.0) { fprintf(stderr, "FAIL: dot\n"); return 1; }

    cblas_qaxpy(N, qd(0.5), x, 1, y, 1);
    if (dq(y[0]) != 2.5 || dq(y[7]) != 6.0) {
        fprintf(stderr, "FAIL: axpy y[0]=%f y[7]=%f\n", dq(y[0]), dq(y[7]));
        return 1;
    }

    printf("  smoke   : OK\n");
    return 0;
}
