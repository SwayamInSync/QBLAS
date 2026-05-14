/* Goto / van de Geijn 5-loop blocked GEMM around the dispatch tier's
 * packed micro-kernel.  Loop order: jc(L3) -> pc(K) -> ic(L2) ->
 * jr,ir(MRxNR register tile).  A and B are packed into contiguous
 * buffers per pc panel so the kernel sees a single uniform layout
 * regardless of user-side transposes / leading dims.  C is pre-scaled by
 * beta once up front; the kernel only does additive FMAs. */

#include "common/qblas_internal.h"
#include "common/qblas_dispatch.h"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

typedef struct {
    const Sleef_quad *data;
    size_t row_stride;
    size_t col_stride;
} qmat_t;

static inline Sleef_quad qmat_get(const qmat_t *M, size_t i, size_t j) {
    return M->data[i * M->row_stride + j * M->col_stride];
}

/* Build a row-major virtual view of A under (layout, trans). */
static qmat_t make_view(const Sleef_quad *A, int lda,
                        QBLAS_LAYOUT layout, QBLAS_TRANSPOSE trans) {
    qmat_t v;
    v.data = A;
    int doT = (trans == QblasTrans || trans == QblasConjTrans);
    if (layout == QblasRowMajor) {
        if (!doT) { v.row_stride = (size_t)lda; v.col_stride = 1; }
        else      { v.row_stride = 1;           v.col_stride = (size_t)lda; }
    } else {
        if (!doT) { v.row_stride = 1;           v.col_stride = (size_t)lda; }
        else      { v.row_stride = (size_t)lda; v.col_stride = 1; }
    }
    return v;
}

/* Pack into [mc/MR][kc][MR] panels.  Tail rows zero-padded so the kernel
 * always sees MR rows. */
static void pack_A(size_t mc, size_t kc,
                   const qmat_t *A, size_t i0, size_t p0,
                   size_t MR,
                   Sleef_quad *restrict Ap) {
    size_t pos = 0;
    size_t i = 0;
    for (; i + MR <= mc; i += MR) {
        for (size_t p = 0; p < kc; ++p) {
            for (size_t r = 0; r < MR; ++r) {
                Ap[pos++] = qmat_get(A, i0 + i + r, p0 + p);
            }
        }
    }
    if (i < mc) {
        size_t left = mc - i;
        for (size_t p = 0; p < kc; ++p) {
            for (size_t r = 0; r < MR; ++r) {
                Ap[pos++] = (r < left) ? qmat_get(A, i0 + i + r, p0 + p) : QBLAS_ZERO;
            }
        }
    }
}

/* Pack into [nc/NR][kc][NR] panels.  Tail cols zero-padded. */
static void pack_B(size_t kc, size_t nc,
                   const qmat_t *B, size_t p0, size_t j0,
                   size_t NR,
                   Sleef_quad *restrict Bp) {
    size_t pos = 0;
    size_t j = 0;
    for (; j + NR <= nc; j += NR) {
        for (size_t p = 0; p < kc; ++p) {
            for (size_t c = 0; c < NR; ++c) {
                Bp[pos++] = qmat_get(B, p0 + p, j0 + j + c);
            }
        }
    }
    if (j < nc) {
        size_t left = nc - j;
        for (size_t p = 0; p < kc; ++p) {
            for (size_t c = 0; c < NR; ++c) {
                Bp[pos++] = (c < left) ? qmat_get(B, p0 + p, j0 + j + c) : QBLAS_ZERO;
            }
        }
    }
}

static void scale_C(int row_major, size_t m, size_t n,
                    Sleef_quad beta, Sleef_quad *C, size_t ldc) {
    if (qisone(beta)) return;
    if (qiszero(beta)) {
        if (row_major) {
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < n; ++j) C[i * ldc + j] = QBLAS_ZERO;
        } else {
            for (size_t j = 0; j < n; ++j)
                for (size_t i = 0; i < m; ++i) C[j * ldc + i] = QBLAS_ZERO;
        }
        return;
    }
    if (row_major) {
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                C[i * ldc + j] = qmul(beta, C[i * ldc + j]);
    } else {
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m; ++i)
                C[j * ldc + i] = qmul(beta, C[j * ldc + i]);
    }
}

/* For a full MR×NR tile we call the kernel against C directly.  For an
 * edge tile the kernel writes the zero-padded MR×NR result into scratch
 * and we add only the live m_real × n_real region into C. */
static void apply_tile(size_t m_real, size_t n_real,
                       size_t MR, size_t NR,
                       size_t kc,
                       Sleef_quad alpha,
                       const Sleef_quad *Ap, const Sleef_quad *Bp,
                       Sleef_quad *C, size_t ldc) {
    if (m_real == MR && n_real == NR) {
        qblas_dispatch_qgemm_kernel(kc, alpha, Ap, Bp, C, ldc);
        return;
    }
    Sleef_quad scratch[16 * 16];           /* MR * NR <= 256 for shipped tiers */
    for (size_t i = 0; i < MR * NR; ++i) scratch[i] = QBLAS_ZERO;
    qblas_dispatch_qgemm_kernel(kc, alpha, Ap, Bp, scratch, NR);
    for (size_t i = 0; i < m_real; ++i)
        for (size_t j = 0; j < n_real; ++j)
            C[i * ldc + j] = qadd(C[i * ldc + j], scratch[i * NR + j]);
}

void cblas_qgemm(QBLAS_LAYOUT layout,
                 QBLAS_TRANSPOSE transa, QBLAS_TRANSPOSE transb,
                 int m_, int n_, int k_,
                 Sleef_quad alpha,
                 const Sleef_quad *A, int lda,
                 const Sleef_quad *B, int ldb,
                 Sleef_quad beta,
                 Sleef_quad *C, int ldc) {
    if (m_ <= 0 || n_ <= 0) return;
    size_t m = (size_t)m_, n = (size_t)n_, k = (size_t)k_;

    int row_major = (layout == QblasRowMajor);

    /* Compute in a row-major C view.  Col-major C is the same buffer
     * reinterpreted: addresses match because additions commute, so we
     * just swap M<->N and A<->B (keeping their transpose flags). */
    qmat_t Av, Bv;
    size_t mm, nn, kk = k;
    size_t ldc_eff = (size_t)ldc;
    if (row_major) {
        mm = m; nn = n;
        Av = make_view(A, lda, layout, transa);
        Bv = make_view(B, ldb, layout, transb);
    } else {
        mm = n; nn = m;
        Av = make_view(B, ldb, QblasRowMajor, transb);
        Bv = make_view(A, lda, QblasRowMajor, transa);
    }

    scale_C(1, mm, nn, beta, C, ldc_eff);

    if (qiszero(alpha) || kk == 0) return;

    qblas_dispatch_init();
    const size_t MR = qblas_dispatch_qgemm_MR;
    const size_t NR = qblas_dispatch_qgemm_NR;

    const qblas_tune_t *tune = qblas_tune();
    size_t kc = tune->gemm_kc;
    size_t mc = tune->gemm_mc;
    size_t nc = tune->gemm_nc;
    if (kc > kk) kc = kk;
    if (mc > mm) mc = mm;

    int nthreads = qblas_resolve_threads(mm * nn, kk);
    if (nthreads < 1) nthreads = 1;

    /* Shrink nc so every thread gets at least ~2 jc-blocks of work. */
    if (nthreads > 1) {
        size_t target_blocks = (size_t)nthreads * 2;
        size_t max_nc_per_block = (nn + target_blocks - 1) / target_blocks;
        if (max_nc_per_block < nc) nc = max_nc_per_block;
        if (nc < NR) nc = NR;
    }
    if (nc > nn) nc = nn;
    size_t mc_pad = ((mc + MR - 1) / MR) * MR;
    size_t nc_pad = ((nc + NR - 1) / NR) * NR;

    size_t Bp_size = kc * nc_pad;
    size_t Ap_size = mc_pad * kc;

    size_t njc = (nn + nc - 1) / nc;

#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
        Sleef_quad *Ap = (Sleef_quad *)qblas_aligned_alloc(Ap_size * sizeof(Sleef_quad));
        Sleef_quad *Bp = (Sleef_quad *)qblas_aligned_alloc(Bp_size * sizeof(Sleef_quad));

#ifdef _OPENMP
        #pragma omp for schedule(dynamic)
#endif
        for (size_t b = 0; b < njc; ++b) {
            size_t jc = b * nc;
            size_t nc_use = (jc + nc <= nn) ? nc : (nn - jc);
            size_t nc_use_pad = ((nc_use + NR - 1) / NR) * NR;

            for (size_t pc = 0; pc < kk; pc += kc) {
                size_t kc_use = (pc + kc <= kk) ? kc : (kk - pc);

                pack_B(kc_use, nc_use, &Bv, pc, jc, NR, Bp);

                for (size_t ic = 0; ic < mm; ic += mc) {
                    size_t mc_use = (ic + mc <= mm) ? mc : (mm - ic);
                    size_t mc_use_pad = ((mc_use + MR - 1) / MR) * MR;

                    pack_A(mc_use, kc_use, &Av, ic, pc, MR, Ap);

                    for (size_t jr = 0; jr < nc_use_pad; jr += NR) {
                        size_t n_real = (jc + jr + NR <= jc + nc_use) ? NR : (nc_use - jr);
                        const Sleef_quad *Bp_panel = Bp + (jr / NR) * (kc_use * NR);
                        for (size_t ir = 0; ir < mc_use_pad; ir += MR) {
                            size_t m_real = (ic + ir + MR <= ic + mc_use) ? MR : (mc_use - ir);
                            const Sleef_quad *Ap_panel = Ap + (ir / MR) * (kc_use * MR);
                            Sleef_quad *C_tile = C + (ic + ir) * ldc_eff + (jc + jr);
                            apply_tile(m_real, n_real, MR, NR, kc_use,
                                       alpha, Ap_panel, Bp_panel,
                                       C_tile, ldc_eff);
                        }
                    }
                }
            }
        }

        qblas_aligned_free(Ap);
        qblas_aligned_free(Bp);
    }
}
