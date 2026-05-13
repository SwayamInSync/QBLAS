/* qgemm: high-performance quad-precision matrix-matrix multiply.
 *
 * Algorithm: Goto / van de Geijn 5-loop blocking around a register tile.
 *
 *     for jc in 0..n step nc:                      // L3 panel of B,C
 *       for pc in 0..k step kc:                    // K panel
 *         pack B[pc:pc+kc, jc:jc+nc] -> Bp         //   contiguous (kc x nc)
 *         for ic in 0..m step mc:                  // L2 panel of A
 *           pack A[ic:ic+mc, pc:pc+kc] -> Ap       //   contiguous (mc x kc)
 *           for jr in 0..nc step NR:               // register tile col
 *             for ir in 0..mc step MR:             // register tile row
 *               kernel(Ap[ir:ir+MR, :],
 *                      Bp[:, jr:jr+NR],
 *                      C [ic+ir:ic+ir+MR, jc+jr:jc+jr+NR])
 *
 * MR / NR come from the dispatch tier (kernel publishes them at init).
 * mc, kc, nc are chosen here so each level fits its target cache:
 *   - Ap   (mc * kc * 16 bytes) ≤ L2 / 2
 *   - Bp slice (kc * NR * 16 bytes) ≤ L1 / 2
 *   - Bp panel (kc * nc * 16 bytes) ≤ L3 / cores
 *
 * Quad is 16 bytes per element. Defaults below target a modern x86 with
 * ~32 KB L1 / 256 KB L2 / 8-16 MB L3, then we round to multiples of MR/NR.
 *
 * Threading: parallel over jc (outer loop).  Each thread gets its own
 * Bp buffer; Ap is shared per-pc.  In practice we keep Ap thread-local too
 * because it's small.
 *
 * Transpose / layout handling: we normalise to row-major no-trans semantics
 * by adjusting how we *read* A and B during packing.  The micro-kernel is
 * always called with packed buffers, so it sees a uniform layout regardless
 * of user-side transposes.
 *
 * Beta scaling: C is pre-scaled by beta in one sweep before any kernel
 * touches it.  After that, the kernel only does additive FMAs.
 */

#include "common/qblas_internal.h"
#include "common/qblas_dispatch.h"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

/* ---- Default cache-blocking parameters.
 *      Tuned for: 32KB L1d, 256KB-1MB L2, 8MB+ L3.  Multiples of MR/NR. */
#define QBLAS_KC_DEFAULT  128
#define QBLAS_MC_DEFAULT  128
#define QBLAS_NC_DEFAULT  1024

/* ---------------- Row-major view helpers ----------------------------
 * We model A and B as virtual row-major matrices once transposes are
 * applied.  The "virtual" element at (i, j) maps onto the physical buffer
 * via a strided index.  `qmat_t` captures everything the packer needs. */
typedef struct {
    const Sleef_quad *data;
    size_t row_stride;    /* between consecutive *virtual* rows */
    size_t col_stride;    /* between consecutive *virtual* cols */
} qmat_t;

static inline Sleef_quad qmat_get(const qmat_t *M, size_t i, size_t j) {
    return M->data[i * M->row_stride + j * M->col_stride];
}

/* Convert (layout, trans, lda) into a row-major virtual view. */
static qmat_t make_view(const Sleef_quad *A, int lda,
                        QBLAS_LAYOUT layout, QBLAS_TRANSPOSE trans) {
    qmat_t v;
    v.data = A;
    int doT = (trans == QblasTrans || trans == QblasConjTrans);
    if (layout == QblasRowMajor) {
        if (!doT) { v.row_stride = (size_t)lda; v.col_stride = 1; }
        else      { v.row_stride = 1;           v.col_stride = (size_t)lda; }
    } else { /* col-major */
        if (!doT) { v.row_stride = 1;           v.col_stride = (size_t)lda; }
        else      { v.row_stride = (size_t)lda; v.col_stride = 1; }
    }
    return v;
}

/* ---------------- Packing ----------------------------------------- */
/* Pack mc x kc rows of A into a contiguous buffer in "panel of MR" layout:
 *   block layout: [mc/MR][kc][MR]  (then row-tail flushed at end).
 * I.e. for each MR-row block, k iterations laid out as MR scalars each.
 */
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
    /* Edge block: pad with zeros so the kernel still sees MR rows. */
    if (i < mc) {
        size_t left = mc - i;
        for (size_t p = 0; p < kc; ++p) {
            for (size_t r = 0; r < MR; ++r) {
                Ap[pos++] = (r < left) ? qmat_get(A, i0 + i + r, p0 + p) : QBLAS_ZERO;
            }
        }
    }
}

/* Pack kc x nc cols of B into a contiguous buffer in "panel of NR" layout:
 *   block layout: [nc/NR][kc][NR].
 */
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

/* ---------------- Beta scaling ----------------------------------- */
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

/* Edge-case kernel: when an mc/nc block isn't a multiple of MR/NR, the last
 * row/col tile gets zero-padded in the packed buffer.  We need to write only
 * the *real* (m_left x n_left) portion back into C.  We accumulate into a
 * local MR x NR scratch and copy the live submatrix.
 *
 * Layout note: C is in user's layout (row or col major).  The dispatch
 * kernel only knows ldc — its stride between consecutive *rows* (it iterates
 * "down the m direction").  For column-major C, the caller passes the
 * transposed virtual view (we do that by swapping M↔N etc. — see gemm()). */
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
    /* Edge: compute into scratch, then copy. */
    Sleef_quad scratch[16 * 16]; /* MR*NR ≤ 256 - safe for tiers we ship */
    for (size_t i = 0; i < MR * NR; ++i) scratch[i] = QBLAS_ZERO;
    qblas_dispatch_qgemm_kernel(kc, alpha, Ap, Bp, scratch, NR);
    for (size_t i = 0; i < m_real; ++i)
        for (size_t j = 0; j < n_real; ++j)
            C[i * ldc + j] = qadd(C[i * ldc + j], scratch[i * NR + j]);
}

/* Note: the dispatch kernel's contract is "C += alpha * Ap * Bp".  But we
 * also support edge tiles via apply_tile.  For the FULL tile case, the
 * kernel reads C and writes back alpha*acc + C (it folds the add into the
 * final FMA).  For the partial-tile case, apply_tile computes alpha*acc
 * into the scratch (zero-init) and then *adds* into the real C.
 *
 * Annoying mismatch in the current kernel template: the QV>1 kernel
 * computes "y = alpha*acc + y_loaded", so the in-place add is done.  But
 * for the scratch path we initialise to zero, call the kernel writing into
 * scratch — the kernel does "scratch = alpha*acc + 0" which is correct.
 * Then apply_tile adds scratch into C. */

/* ---------------- Driver ----------------------------------------- */
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

    /* Strategy: always compute in *row-major-of-C* view.  If C is column
     * major we instead compute (alpha * op(B)^T * op(A)^T + beta * C^T)
     * which is the same buffer, just transposed.  Net effect: swap m↔n,
     * swap A↔B, flip both transposes. */
    qmat_t Av, Bv;
    Sleef_quad *Crm;
    size_t mm, nn, kk;
    size_t ldc_eff;
    if (row_major) {
        mm = m; nn = n; kk = k;
        Av = make_view(A, lda, layout, transa);
        Bv = make_view(B, ldb, layout, transb);
        Crm = C;
        ldc_eff = (size_t)ldc;
    } else {
        /* Reinterpret as row-major by flipping. */
        mm = n; nn = m; kk = k;
        /* In the new view: A_new = op(B)^T, B_new = op(A)^T. */
        QBLAS_TRANSPOSE new_ta = (transb == QblasNoTrans) ? QblasTrans : QblasNoTrans;
        QBLAS_TRANSPOSE new_tb = (transa == QblasNoTrans) ? QblasTrans : QblasNoTrans;
        Av = make_view(B, ldb, QblasRowMajor, new_ta);
        Bv = make_view(A, lda, QblasRowMajor, new_tb);
        Crm = C;
        ldc_eff = (size_t)ldc;
    }

    /* Pre-scale C. */
    if (row_major) {
        scale_C(1, mm, nn, beta, Crm, ldc_eff);
    } else {
        /* C is column-major: we wrote everything in transposed (row-major)
         * coords, so iterate accordingly. */
        scale_C(0, n, m, beta, Crm, ldc_eff);
    }

    if (qiszero(alpha) || kk == 0) return;

    /* Make sure dispatch is initialised. */
    qblas_dispatch_init();
    const size_t MR = qblas_dispatch_qgemm_MR;
    const size_t NR = qblas_dispatch_qgemm_NR;

    size_t kc = QBLAS_KC_DEFAULT;
    size_t mc = QBLAS_MC_DEFAULT;
    size_t nc = QBLAS_NC_DEFAULT;
    if (kc > kk) kc = kk;
    if (mc > mm) mc = mm;
    if (nc > nn) nc = nn;
    /* Round mc / nc up to MR / NR multiples so the inner loops are clean. */
    size_t mc_pad = ((mc + MR - 1) / MR) * MR;
    size_t nc_pad = ((nc + NR - 1) / NR) * NR;

    /* For row-major Crm we walk Crm[i, j] = Crm + i*ldc_eff + j; for the
     * column-major rewrite, Crm[i_new, j_new] (in our virtual view) lives
     * at Crm + i_new + j_new * ldc_eff — i.e. row stride 1, col stride
     * ldc_eff.  We pass either form into the kernels by adjusting the
     * effective ldc and pointer per tile. */
    int col_view_for_C = !row_major;  /* if true, virtual rows of C have stride 1 */

    size_t Bp_size = kc * nc_pad;
    size_t Ap_size = mc_pad * kc;

    int nthreads = qblas_resolve_threads(mm * nn, kk);
    if (nthreads < 1) nthreads = 1;

#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
        Sleef_quad *Ap = (Sleef_quad *)qblas_aligned_alloc(Ap_size * sizeof(Sleef_quad));
        Sleef_quad *Bp = (Sleef_quad *)qblas_aligned_alloc(Bp_size * sizeof(Sleef_quad));

        /* Each thread owns a strided slice of jc.  Static schedule over
         * jc-chunks of size nc keeps each thread on contiguous C columns. */
#ifdef _OPENMP
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();
#else
        int tid = 0, nt = 1;
#endif

        for (size_t jc = (size_t)tid * nc; jc < nn; jc += (size_t)nt * nc) {
            size_t nc_use = (jc + nc <= nn) ? nc : (nn - jc);
            size_t nc_use_pad = ((nc_use + NR - 1) / NR) * NR;

            for (size_t pc = 0; pc < kk; pc += kc) {
                size_t kc_use = (pc + kc <= kk) ? kc : (kk - pc);

                pack_B(kc_use, nc_use, &Bv, pc, jc, NR, Bp);

                for (size_t ic = 0; ic < mm; ic += mc) {
                    size_t mc_use = (ic + mc <= mm) ? mc : (mm - ic);
                    size_t mc_use_pad = ((mc_use + MR - 1) / MR) * MR;

                    pack_A(mc_use, kc_use, &Av, ic, pc, MR, Ap);

                    /* Now iterate over the (mc_use_pad x nc_use_pad) tile of
                     * the packed buffers, calling the micro-kernel. */
                    for (size_t jr = 0; jr < nc_use_pad; jr += NR) {
                        size_t n_real = (jc + jr + NR <= jc + nc_use) ? NR : (nc_use - jr);
                        const Sleef_quad *Bp_panel = Bp + (jr / NR) * (kc_use * NR);
                        for (size_t ir = 0; ir < mc_use_pad; ir += MR) {
                            size_t m_real = (ic + ir + MR <= ic + mc_use) ? MR : (mc_use - ir);
                            const Sleef_quad *Ap_panel = Ap + (ir / MR) * (kc_use * MR);

                            /* Position in C for this tile. */
                            Sleef_quad *C_tile;
                            size_t C_row_stride; /* stride between micro-kernel "rows" */
                            if (!col_view_for_C) {
                                /* row-major C: C[i*ldc + j], rows stride ldc */
                                C_tile = Crm + (ic + ir) * ldc_eff + (jc + jr);
                                C_row_stride = ldc_eff;
                            } else {
                                /* col-major C: virtual row i_new, col j_new
                                 * lives at C + i_new + j_new * ldc.  But we
                                 * computed the gemm in the transposed view
                                 * where i_new=jc+jr direction, j_new=ic+ir
                                 * direction — wait, no. We swapped m↔n.
                                 *
                                 * Actually the col-major rewrite means the
                                 * kernel's (i, j) corresponds to original
                                 * (j, i) in user space.  Original C[j, i] in
                                 * column-major = C + i*ldc + j.  So in the
                                 * virtual view, "row i" of kernel = original
                                 * column i; stride between virtual rows of C
                                 * equals stride between original cols = ldc.
                                 *
                                 * Hmm — that's actually the same memory
                                 * stride as the row-major case.  The only
                                 * difference is what we pass for the tile
                                 * base. */
                                C_tile = Crm + (jc + jr) + (ic + ir) * ldc_eff;
                                /* Now the micro-kernel "moves down one row"
                                 * by +1 in memory; "moves right one column"
                                 * by +ldc_eff.  But it expects rows of stride
                                 * ldc and cols of stride 1.  So this is
                                 * effectively the *transpose* of what we
                                 * want — the kernel can't handle that.
                                 *
                                 * Easiest correct fix: write into a scratch
                                 * MR x NR (row stride NR) and transpose-copy
                                 * back. */
                                Sleef_quad scratch[16 * 16];
                                size_t i, j;
                                for (i = 0; i < MR * NR; ++i) scratch[i] = QBLAS_ZERO;
                                qblas_dispatch_qgemm_kernel(kc_use, alpha,
                                                            Ap_panel, Bp_panel,
                                                            scratch, NR);
                                for (i = 0; i < m_real; ++i)
                                    for (j = 0; j < n_real; ++j)
                                        C_tile[j + i * ldc_eff] =
                                            qadd(C_tile[j + i * ldc_eff],
                                                 scratch[i * NR + j]);
                                continue; /* skip the normal kernel path */
                            }
                            apply_tile(m_real, n_real, MR, NR, kc_use,
                                       alpha, Ap_panel, Bp_panel,
                                       C_tile, C_row_stride);
                        }
                    }
                }
            }
        }

        qblas_aligned_free(Ap);
        qblas_aligned_free(Bp);
    } /* omp parallel */
}
