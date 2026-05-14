
#include "test_helpers.h"

static void ref_syrk_row(int doT, int n, int k,
                         Sleef_quad alpha,
                         const Sleef_quad *A, int lda,
                         Sleef_quad beta, Sleef_quad *C, int ldc) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Sleef_quad s = qd(0.0);
            for (int p = 0; p < k; ++p) {
                Sleef_quad aip = doT ? A[(size_t)p * lda + i] : A[(size_t)i * lda + p];
                Sleef_quad ajp = doT ? A[(size_t)p * lda + j] : A[(size_t)j * lda + p];
                s = q_fma(aip, ajp, s);
            }
            C[(size_t)i * ldc + j] = q_fma(beta, C[(size_t)i * ldc + j], q_mul(alpha, s));
        }
    }
}

static void test_syrk(int n, int k, QBLAS_LAYOUT layout, QBLAS_UPLO uplo, QBLAS_TRANSPOSE trans) {
    int doT = (trans == QblasTrans);
    int Arows = doT ? k : n, Acols = doT ? n : k;
    Sleef_quad *A_row = malloc((size_t)Arows * Acols * sizeof(Sleef_quad));
    Sleef_quad *C_row = malloc((size_t)n * n * sizeof(Sleef_quad));
    fill_mat(A_row, Arows, Acols, Acols);
    fill_mat(C_row, n, n, n);

    Sleef_quad alpha = qd(0.7), beta = qd(0.3);

    Sleef_quad *C_ref = malloc((size_t)n * n * sizeof(Sleef_quad));
    memcpy(C_ref, C_row, (size_t)n * n * sizeof(Sleef_quad));
    ref_syrk_row(doT, n, k, alpha, A_row, Acols, beta, C_ref, n);

    Sleef_quad *A_use, *C_use; int lda, ldc;
    Sleef_quad *A_col=NULL, *C_col=NULL;
    if (layout == QblasRowMajor) {
        A_use = A_row; lda = Acols;
        C_use = malloc((size_t)n * n * sizeof(Sleef_quad));
        memcpy(C_use, C_row, (size_t)n * n * sizeof(Sleef_quad));
        ldc = n;
    } else {
        A_col = malloc((size_t)Arows * Acols * sizeof(Sleef_quad));
        C_col = malloc((size_t)n * n * sizeof(Sleef_quad));
        for (int i = 0; i < Arows; ++i) for (int j = 0; j < Acols; ++j)
            A_col[(size_t)j * Arows + i] = A_row[(size_t)i * Acols + j];
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
            C_col[(size_t)j * n + i] = C_row[(size_t)i * n + j];
        A_use = A_col; lda = Arows;
        C_use = C_col; ldc = n;
    }

    cblas_qsyrk(layout, uplo, trans, n, k, alpha, A_use, lda, beta, C_use, ldc);

    Sleef_quad *C_got = malloc((size_t)n * n * sizeof(Sleef_quad));
    if (layout == QblasRowMajor) memcpy(C_got, C_use, (size_t)n * n * sizeof(Sleef_quad));
    else for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
        C_got[(size_t)i * n + j] = C_use[(size_t)j * n + i];

    int fails = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int in_tri = (uplo == QblasUpper) ? (j >= i) : (j <= i);
            if (!in_tri) continue;
            Sleef_quad d = q_abs(q_sub(C_got[(size_t)i*n+j], C_ref[(size_t)i*n+j]));
            Sleef_quad s = q_add(q_abs(C_got[(size_t)i*n+j]), q_abs(C_ref[(size_t)i*n+j]));
            double dd = dq(d), ss = dq(s);
            double r = (ss > 1e-20) ? dd/ss : dd;
            if (r > 1e-28) ++fails;
        }
    }
    char tag[128];
    snprintf(tag, sizeof tag, "syrk n=%d k=%d layout=%d uplo=%d trans=%d", n, k, layout, uplo, trans);
    CHECK(fails == 0, "%s (%d cells off)", tag, fails);

    free(C_got);
    if (layout == QblasRowMajor) free(C_use);
    if (A_col) free(A_col);
    if (C_col) free(C_col);
    free(C_ref); free(C_row); free(A_row);
}

/* Verify trsm by multiplying op(A)*X and comparing to alpha*B_orig.
 * A and B are kept row-major; col-major paths transpose into a scratch
 * buffer before being passed to qblas. */
static inline Sleef_quad rowmaj_A(const Sleef_quad *A_row, int m, int i, int j) {
    return A_row[(size_t)i * m + j];
}
static inline Sleef_quad rowmaj_B(const Sleef_quad *B_row, int n, int i, int j) {
    return B_row[(size_t)i * n + j];
}

static void test_trsm_left(int m, int n, QBLAS_LAYOUT layout,
                           QBLAS_UPLO uplo, QBLAS_TRANSPOSE transa, QBLAS_DIAG diag) {
    Sleef_quad *A_row  = malloc((size_t)m * m * sizeof(Sleef_quad));
    Sleef_quad *B_row  = malloc((size_t)m * n * sizeof(Sleef_quad));
    Sleef_quad *B0_row = malloc((size_t)m * n * sizeof(Sleef_quad));
    fill_mat(A_row, m, m, m);
    for (int i = 0; i < m; ++i) A_row[(size_t)i * m + i] = qd(frand(2.0, 4.0));
    fill_mat(B_row, m, n, n);
    memcpy(B0_row, B_row, (size_t)m * n * sizeof(Sleef_quad));

    Sleef_quad *A_use, *B_use; int lda, ldb;
    Sleef_quad *A_col = NULL, *B_col = NULL;
    if (layout == QblasRowMajor) {
        A_use = A_row; lda = m;
        B_use = B_row; ldb = n;
    } else {
        A_col = malloc((size_t)m * m * sizeof(Sleef_quad));
        B_col = malloc((size_t)m * n * sizeof(Sleef_quad));
        for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j)
            A_col[(size_t)j * m + i] = A_row[(size_t)i * m + j];
        for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j)
            B_col[(size_t)j * m + i] = B_row[(size_t)i * n + j];
        A_use = A_col; lda = m;
        B_use = B_col; ldb = m;
    }

    Sleef_quad alpha = qd(0.5);
    cblas_qtrsm(layout, QblasLeft, uplo, transa, diag, m, n,
                alpha, A_use, lda, B_use, ldb);

    Sleef_quad *X_row = malloc((size_t)m * n * sizeof(Sleef_quad));
    if (layout == QblasRowMajor) memcpy(X_row, B_use, (size_t)m * n * sizeof(Sleef_quad));
    else for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j)
        X_row[(size_t)i * n + j] = B_use[(size_t)j * m + i];

    /* Backward error: ‖op(A)·X − alpha·B0‖_F / (‖A‖_F·‖X‖_F + ‖alpha·B0‖_F)
     * — the standard stability check for triangular solves. */
    int doT = (transa == QblasTrans);
    int upper = (uplo == QblasUpper);
    double res_sq = 0.0, denom_sq = 0.0, A_fro = 0.0, X_fro = 0.0, B_fro = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < m; ++p) {
            int in_tri_orig = upper ? (p >= i) : (p <= i);
            if (!in_tri_orig) continue;
            double a = (diag == QblasUnit && p == i) ? 1.0
                                                     : dq(rowmaj_A(A_row, m, i, p));
            A_fro += a * a;
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            X_fro += dq(rowmaj_B(X_row, n, i, j)) * dq(rowmaj_B(X_row, n, i, j));
            B_fro += dq(q_mul(alpha, rowmaj_B(B0_row, n, i, j))) * dq(q_mul(alpha, rowmaj_B(B0_row, n, i, j)));
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            Sleef_quad s = qd(0.0);
            for (int p = 0; p < m; ++p) {
                int ii = doT ? p : i;
                int pp = doT ? i : p;
                int in_tri = upper ? (pp >= ii) : (pp <= ii);
                if (!in_tri) continue;
                Sleef_quad aip;
                if (diag == QblasUnit && ii == pp) aip = qd(1.0);
                else aip = rowmaj_A(A_row, m, ii, pp);
                s = q_fma(aip, rowmaj_B(X_row, n, p, j), s);
            }
            Sleef_quad rhs = q_mul(alpha, rowmaj_B(B0_row, n, i, j));
            double r = dq(q_sub(s, rhs));
            res_sq += r * r;
        }
    }
    A_fro = sqrt(A_fro);
    X_fro = sqrt(X_fro);
    B_fro = sqrt(B_fro);
    denom_sq = A_fro * X_fro + B_fro + 1e-300;
    double backward_err = sqrt(res_sq) / denom_sq;
    char tag[128];
    snprintf(tag, sizeof tag, "trsm m=%d n=%d layout=%d uplo=%d ta=%d diag=%d",
             m, n, layout, uplo, transa, diag);
    /* 1e-25 ceiling is ~10¹⁰ tighter than double-precision rounding. */
    CHECK(backward_err < 1e-25, "%s backward_err=%.3e", tag, backward_err);
    free(X_row);
    if (A_col) free(A_col);
    if (B_col) free(B_col);
    free(B0_row); free(B_row); free(A_row);
}

int main(void) {
    int dims[][2] = { {3,3}, {8,5}, {17,11}, {32,32} };
    for (size_t d = 0; d < sizeof dims / sizeof dims[0]; ++d) {
        for (int layout = QblasRowMajor; layout <= QblasColMajor; ++layout)
            for (int uplo = QblasUpper; uplo <= QblasLower; ++uplo)
                for (int trans = QblasNoTrans; trans <= QblasTrans; ++trans) {
                    test_syrk(dims[d][0], dims[d][1], (QBLAS_LAYOUT)layout,
                              (QBLAS_UPLO)uplo, (QBLAS_TRANSPOSE)trans);
                }
    }

    /* trsm size grid spans the NB=64 block-size boundary. */
    int td[][2] = { {3,3}, {8,5}, {17,11}, {63,17}, {64,33}, {129,11}, {200,40} };
    for (size_t d = 0; d < sizeof td / sizeof td[0]; ++d) {
        for (int layout = QblasRowMajor; layout <= QblasColMajor; ++layout)
            for (int uplo = QblasUpper; uplo <= QblasLower; ++uplo)
                for (int trans = QblasNoTrans; trans <= QblasTrans; ++trans)
                    for (int diag = QblasNonUnit; diag <= QblasUnit; ++diag)
                        test_trsm_left(td[d][0], td[d][1], (QBLAS_LAYOUT)layout,
                                       (QBLAS_UPLO)uplo, (QBLAS_TRANSPOSE)trans,
                                       (QBLAS_DIAG)diag);
    }
    REPORT();
}
