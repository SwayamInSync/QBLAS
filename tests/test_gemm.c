/* GEMM correctness: all combinations of layout x transA x transB x sizes. */
#include "test_helpers.h"

static void ref_gemm_row(int doTa, int doTb, int m, int n, int k,
                         Sleef_quad alpha,
                         const Sleef_quad *A, int lda,
                         const Sleef_quad *B, int ldb,
                         Sleef_quad beta,
                         Sleef_quad *C, int ldc) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            Sleef_quad s = qd(0.0);
            for (int p = 0; p < k; ++p) {
                Sleef_quad a = doTa ? A[(size_t)p * lda + i] : A[(size_t)i * lda + p];
                Sleef_quad b = doTb ? B[(size_t)j * ldb + p] : B[(size_t)p * ldb + j];
                s = q_fma(a, b, s);
            }
            C[(size_t)i * ldc + j] = q_fma(beta, C[(size_t)i * ldc + j],
                                            q_mul(alpha, s));
        }
    }
}

/* Convert A row-major (rows x cols, lda=cols) to col-major (lda=rows). */
static void rowmajor_to_colmajor(const Sleef_quad *A, int rows, int cols,
                                 Sleef_quad *B) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            B[(size_t)j * rows + i] = A[(size_t)i * cols + j];
}

static void test_gemm_case(int m, int n, int k,
                           QBLAS_LAYOUT layout,
                           QBLAS_TRANSPOSE ta, QBLAS_TRANSPOSE tb) {
    int doTa = (ta == QblasTrans || ta == QblasConjTrans);
    int doTb = (tb == QblasTrans || tb == QblasConjTrans);

    /* Logical sizes (always row-major reference) */
    int Arows = doTa ? k : m, Acols = doTa ? m : k;
    int Brows = doTb ? n : k, Bcols = doTb ? k : n;

    Sleef_quad *A_row = malloc((size_t)Arows * Acols * sizeof(Sleef_quad));
    Sleef_quad *B_row = malloc((size_t)Brows * Bcols * sizeof(Sleef_quad));
    Sleef_quad *C0    = malloc((size_t)m * n * sizeof(Sleef_quad));
    fill_mat(A_row, Arows, Acols, Acols);
    fill_mat(B_row, Brows, Bcols, Bcols);
    fill_mat(C0,    m,     n,     n);

    Sleef_quad alpha = qd(0.6), beta = qd(-0.4);

    /* Reference: compute on row-major copy. */
    Sleef_quad *C_ref = malloc((size_t)m * n * sizeof(Sleef_quad));
    memcpy(C_ref, C0, (size_t)m * n * sizeof(Sleef_quad));
    ref_gemm_row(doTa, doTb, m, n, k, alpha,
                 A_row, Acols, B_row, Bcols, beta, C_ref, n);

    /* Build inputs in requested layout. */
    Sleef_quad *A_use, *B_use, *C_use;
    int lda, ldb, ldc;
    Sleef_quad *A_col=NULL, *B_col=NULL, *C_col=NULL;
    if (layout == QblasRowMajor) {
        A_use = A_row; lda = Acols;
        B_use = B_row; ldb = Bcols;
        C_use = malloc((size_t)m * n * sizeof(Sleef_quad));
        memcpy(C_use, C0, (size_t)m * n * sizeof(Sleef_quad));
        ldc = n;
    } else {
        A_col = malloc((size_t)Arows * Acols * sizeof(Sleef_quad));
        B_col = malloc((size_t)Brows * Bcols * sizeof(Sleef_quad));
        C_col = malloc((size_t)m * n * sizeof(Sleef_quad));
        rowmajor_to_colmajor(A_row, Arows, Acols, A_col);
        rowmajor_to_colmajor(B_row, Brows, Bcols, B_col);
        rowmajor_to_colmajor(C0,    m,     n,     C_col);
        A_use = A_col; lda = Arows;
        B_use = B_col; ldb = Brows;
        C_use = C_col; ldc = m;
    }

    cblas_qgemm(layout, ta, tb, m, n, k, alpha,
                A_use, lda, B_use, ldb, beta, C_use, ldc);

    /* Bring C_use back to row-major for comparison. */
    Sleef_quad *C_got = malloc((size_t)m * n * sizeof(Sleef_quad));
    if (layout == QblasRowMajor) {
        memcpy(C_got, C_use, (size_t)m * n * sizeof(Sleef_quad));
    } else {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                C_got[(size_t)i * n + j] = C_use[(size_t)j * m + i];
    }

    char tag[160];
    snprintf(tag, sizeof tag,
             "gemm m=%d n=%d k=%d layout=%d ta=%d tb=%d",
             m, n, k, layout, ta, tb);
    CHECK_ARR(C_got, C_ref, m * n, 1e-29, tag);

    free(C_got); free(C_ref);
    if (layout == QblasRowMajor) free(C_use);
    if (A_col) free(A_col);
    if (B_col) free(B_col);
    if (C_col) free(C_col);
    free(C0); free(B_row); free(A_row);
}

int main(void) {
    /* Small (edge-tile) and larger (multi-block) sizes. */
    int sizes[][3] = {
        {1, 1, 1},     {1, 3, 1},   {3, 1, 1},
        {4, 4, 4},     {5, 5, 5},   {7, 11, 13},
        {16, 16, 16},  {17, 19, 23},
        {32, 32, 32},  {64, 64, 64},
        {65, 65, 65},  {129, 33, 17},
        {128, 128, 128}
    };
    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; ++s) {
        int m = sizes[s][0], n = sizes[s][1], k = sizes[s][2];
        for (int layout = QblasRowMajor; layout <= QblasColMajor; ++layout)
            for (int ta = QblasNoTrans; ta <= QblasTrans; ++ta)
                for (int tb = QblasNoTrans; tb <= QblasTrans; ++tb)
                    test_gemm_case(m, n, k, (QBLAS_LAYOUT)layout,
                                   (QBLAS_TRANSPOSE)ta, (QBLAS_TRANSPOSE)tb);
    }
    REPORT();
}
