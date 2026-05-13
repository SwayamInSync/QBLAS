/* Level 2 BLAS correctness suite — gemv, ger, symv, trsv.
 *
 * Strategy: build a row-major naive reference for each, then translate the
 * test matrices/vectors to both layouts and all transpose options, calling
 * cblas_qgemv etc. and checking the output against the reference. */
#include "test_helpers.h"

/* Reference gemv: y = alpha * op(A) * x + beta * y, row-major. */
static void ref_gemv_row(int doT, int m, int n,
                         Sleef_quad alpha,
                         const Sleef_quad *A, int lda,
                         const Sleef_quad *x, int incx,
                         Sleef_quad beta,
                         Sleef_quad *y, int incy) {
    int rows = doT ? n : m;
    int cols = doT ? m : n;
    int ox = (incx < 0) ? (cols - 1) * (-incx) : 0;
    int oy = (incy < 0) ? (rows - 1) * (-incy) : 0;
    for (int i = 0; i < rows; ++i) {
        Sleef_quad s = qd(0.0);
        for (int j = 0; j < cols; ++j) {
            Sleef_quad aij = doT ? A[(size_t)j * lda + i] : A[(size_t)i * lda + j];
            s = q_fma(aij, x[ox + j * incx], s);
        }
        Sleef_quad scaled = q_mul(alpha, s);
        Sleef_quad oldy = y[oy + i * incy];
        y[oy + i * incy] = q_fma(beta, oldy, scaled);
    }
}

/* Take a row-major M x N matrix and produce its column-major copy in `B`. */
static void to_colmajor(const Sleef_quad *Arow, int M, int N, int lda_row,
                        Sleef_quad *Bcol, int ldb_col) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            Bcol[(size_t)j * ldb_col + i] = Arow[(size_t)i * lda_row + j];
}

static void test_gemv_case(int m, int n, QBLAS_LAYOUT layout, QBLAS_TRANSPOSE trans,
                           int incx, int incy) {
    /* Build matrix in row-major, reference uses row-major directly. */
    Sleef_quad *Arow = (Sleef_quad *)malloc((size_t)m * n * sizeof(Sleef_quad));
    fill_mat(Arow, m, n, n);

    int doT = (trans == QblasTrans || trans == QblasConjTrans);
    int rows_out = doT ? n : m;
    int cols_in  = doT ? m : n;

    int len_x = cols_in  * (incx < 0 ? -incx : incx);
    int len_y = rows_out * (incy < 0 ? -incy : incy);

    Sleef_quad *x  = (Sleef_quad *)malloc(len_x * sizeof(Sleef_quad));
    Sleef_quad *y0 = (Sleef_quad *)malloc(len_y * sizeof(Sleef_quad));
    fill_vec(x, len_x, 1);
    fill_vec(y0, len_y, 1);

    Sleef_quad alpha = qd(0.75), beta = qd(-0.3);
    Sleef_quad *y_ref = (Sleef_quad *)malloc(len_y * sizeof(Sleef_quad));
    memcpy(y_ref, y0, len_y * sizeof(Sleef_quad));
    ref_gemv_row(doT, m, n, alpha, Arow, n, x, incx, beta, y_ref, incy);

    Sleef_quad *A_use;
    int lda_use;
    Sleef_quad *Acol = NULL;
    if (layout == QblasRowMajor) {
        A_use = Arow; lda_use = n;
    } else {
        Acol = (Sleef_quad *)malloc((size_t)m * n * sizeof(Sleef_quad));
        to_colmajor(Arow, m, n, n, Acol, m);
        A_use = Acol; lda_use = m;
    }

    Sleef_quad *y_got = (Sleef_quad *)malloc(len_y * sizeof(Sleef_quad));
    memcpy(y_got, y0, len_y * sizeof(Sleef_quad));
    cblas_qgemv(layout, trans, m, n, alpha, A_use, lda_use, x, incx, beta, y_got, incy);

    char tag[128];
    snprintf(tag, sizeof tag, "gemv m=%d n=%d layout=%d trans=%d incx=%d incy=%d",
             m, n, layout, trans, incx, incy);
    CHECK_ARR(y_got, y_ref, len_y, 1e-29, tag);

    free(y_got); free(y_ref); free(y0); free(x); free(Arow);
    if (Acol) free(Acol);
}

static void test_gemv_all(void) {
    int dims[][2] = { {1,1}, {3,5}, {17,1}, {1,32}, {64,33}, {7, 11}, {32, 32}, {127, 65} };
    int strides[] = { 1, 2, -1 };
    for (size_t d = 0; d < sizeof dims / sizeof dims[0]; ++d) {
        for (size_t s1 = 0; s1 < sizeof strides / sizeof strides[0]; ++s1) {
            for (size_t s2 = 0; s2 < sizeof strides / sizeof strides[0]; ++s2) {
                for (int layout = QblasRowMajor; layout <= QblasColMajor; ++layout) {
                    for (int t = QblasNoTrans; t <= QblasTrans; ++t) {
                        test_gemv_case(dims[d][0], dims[d][1],
                                       (QBLAS_LAYOUT)layout, (QBLAS_TRANSPOSE)t,
                                       strides[s1], strides[s2]);
                    }
                }
            }
        }
    }
}

/* ----- ger ----- */
static void ref_ger_row(int m, int n,
                        Sleef_quad alpha,
                        const Sleef_quad *x, int incx,
                        const Sleef_quad *y, int incy,
                        Sleef_quad *A, int lda) {
    int ox = (incx < 0) ? (m - 1) * (-incx) : 0;
    int oy = (incy < 0) ? (n - 1) * (-incy) : 0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A[(size_t)i * lda + j] = q_fma(q_mul(alpha, x[ox + i * incx]),
                                           y[oy + j * incy],
                                           A[(size_t)i * lda + j]);
}

static void test_ger_case(int m, int n, QBLAS_LAYOUT layout, int incx, int incy) {
    int len_x = m * (incx < 0 ? -incx : incx);
    int len_y = n * (incy < 0 ? -incy : incy);
    Sleef_quad *x = (Sleef_quad *)malloc(len_x * sizeof(Sleef_quad));
    Sleef_quad *y = (Sleef_quad *)malloc(len_y * sizeof(Sleef_quad));
    Sleef_quad *Arow = (Sleef_quad *)malloc((size_t)m * n * sizeof(Sleef_quad));
    Sleef_quad *Arow0 = (Sleef_quad *)malloc((size_t)m * n * sizeof(Sleef_quad));
    fill_vec(x, len_x, 1);
    fill_vec(y, len_y, 1);
    fill_mat(Arow, m, n, n);
    memcpy(Arow0, Arow, (size_t)m * n * sizeof(Sleef_quad));

    Sleef_quad alpha = qd(0.5);
    ref_ger_row(m, n, alpha, x, incx, y, incy, Arow0, n);

    Sleef_quad *A_use; int lda_use; Sleef_quad *Acol = NULL;
    if (layout == QblasRowMajor) {
        A_use = Arow; lda_use = n;
    } else {
        Acol = (Sleef_quad *)malloc((size_t)m * n * sizeof(Sleef_quad));
        to_colmajor(Arow, m, n, n, Acol, m);
        A_use = Acol; lda_use = m;
    }
    cblas_qger(layout, m, n, alpha, x, incx, y, incy, A_use, lda_use);

    /* Bring back to row-major for comparison. */
    Sleef_quad *Arow_after = (Sleef_quad *)malloc((size_t)m * n * sizeof(Sleef_quad));
    if (layout == QblasRowMajor) {
        memcpy(Arow_after, A_use, (size_t)m * n * sizeof(Sleef_quad));
    } else {
        /* Acol back to row major. */
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                Arow_after[(size_t)i * n + j] = A_use[(size_t)j * lda_use + i];
    }

    char tag[128];
    snprintf(tag, sizeof tag, "ger m=%d n=%d layout=%d incx=%d incy=%d", m, n, layout, incx, incy);
    CHECK_ARR(Arow_after, Arow0, m * n, 1e-29, tag);

    free(Arow_after); free(Arow0); free(Arow); free(x); free(y);
    if (Acol) free(Acol);
}

static void test_ger_all(void) {
    int dims[][2] = { {1,1}, {3,5}, {17,8}, {32,32}, {64,17}, {7,11} };
    int strides[] = { 1, 2, -1 };
    for (size_t d = 0; d < sizeof dims / sizeof dims[0]; ++d) {
        for (size_t s1 = 0; s1 < sizeof strides / sizeof strides[0]; ++s1) {
            for (size_t s2 = 0; s2 < sizeof strides / sizeof strides[0]; ++s2) {
                for (int layout = QblasRowMajor; layout <= QblasColMajor; ++layout) {
                    test_ger_case(dims[d][0], dims[d][1],
                                  (QBLAS_LAYOUT)layout, strides[s1], strides[s2]);
                }
            }
        }
    }
}

/* ----- trsv: solve op(A) x = b, A triangular ----- */
static void test_trsv_case(int n, QBLAS_LAYOUT layout, QBLAS_UPLO uplo,
                           QBLAS_TRANSPOSE trans, QBLAS_DIAG diag) {
    Sleef_quad *Arow = (Sleef_quad *)malloc((size_t)n * n * sizeof(Sleef_quad));
    fill_mat(Arow, n, n, n);
    /* Make A diagonally dominant so the triangular solve is well-conditioned
     * and we don't see catastrophic cancellation. */
    for (int i = 0; i < n; ++i)
        Arow[(size_t)i * n + i] = qd(frand(2.0, 4.0));

    Sleef_quad *x_orig = (Sleef_quad *)malloc(n * sizeof(Sleef_quad));
    fill_vec(x_orig, n, 1);

    /* Build b = A * x_orig (using the triangle we'll actually use). */
    int doT = (trans == QblasTrans || trans == QblasConjTrans);
    int upper = (uplo == QblasUpper);
    Sleef_quad *b = (Sleef_quad *)malloc(n * sizeof(Sleef_quad));
    for (int i = 0; i < n; ++i) {
        Sleef_quad s = qd(0.0);
        for (int j = 0; j < n; ++j) {
            int ii = doT ? j : i;
            int jj = doT ? i : j;
            int in_triangle = upper ? (jj >= ii) : (jj <= ii);
            if (!in_triangle) continue;
            Sleef_quad aij;
            if (diag == QblasUnit && ii == jj) aij = qd(1.0);
            else aij = Arow[(size_t)ii * n + jj];
            s = q_fma(aij, x_orig[j], s);
        }
        b[i] = s;
    }

    /* Translate A to the requested layout. */
    Sleef_quad *A_use; int lda_use; Sleef_quad *Acol = NULL;
    if (layout == QblasRowMajor) { A_use = Arow; lda_use = n; }
    else { Acol = malloc((size_t)n*n*sizeof(Sleef_quad)); to_colmajor(Arow, n, n, n, Acol, n); A_use = Acol; lda_use = n; }

    Sleef_quad *x = (Sleef_quad *)malloc(n * sizeof(Sleef_quad));
    memcpy(x, b, n * sizeof(Sleef_quad));
    cblas_qtrsv(layout, uplo, trans, diag, n, A_use, lda_use, x, 1);

    char tag[128];
    snprintf(tag, sizeof tag, "trsv n=%d layout=%d uplo=%d trans=%d diag=%d", n, layout, uplo, trans, diag);
    CHECK_ARR(x, x_orig, n, 1e-25, tag);

    free(x); free(b); free(x_orig); free(Arow);
    if (Acol) free(Acol);
}

static void test_trsv_all(void) {
    int sizes[] = { 1, 5, 16, 33, 64 };
    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; ++s) {
        int n = sizes[s];
        for (int layout = QblasRowMajor; layout <= QblasColMajor; ++layout)
            for (int uplo = QblasUpper; uplo <= QblasLower; ++uplo)
                for (int trans = QblasNoTrans; trans <= QblasTrans; ++trans)
                    for (int diag = QblasNonUnit; diag <= QblasUnit; ++diag)
                        test_trsv_case(n, layout, uplo, trans, diag);
    }
}

int main(void) {
    test_gemv_all();
    test_ger_all();
    test_trsv_all();
    REPORT();
}
