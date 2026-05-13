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
 *      kc: chosen so that the MR x kc panel of A_packed + kc x NR slice of B
 *          fit in L1 (with room for the C tile).
 *      mc: chosen so that mc x kc panel of A_packed fits in L2.
 *      nc: chosen as a function of the number of threads — see qgemm() — so
 *          every thread gets ≥ 1 jc-block.  This constant is a *ceiling*.
 *
 *      Quad is 16 bytes per element so memory pressure is 2× double.
 *
 *      Concrete kc/mc/nc come from `qblas_tune()`, which derives them at
 *      library init from CPUID cache descriptors. */

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

    /* Strategy: always compute in a row-major C view.  Column-major C is
     * the same buffer reinterpreted: col-major C[i,j] at C + i + j*ldc is
     * the same address as row-major C^T[j,i] at C + j*ldc + i (additions
     * commute).  So a col-major gemm
     *   C := alpha op(A) op(B) + beta C    (C is m x n)
     * becomes the row-major gemm
     *   C^T := alpha op(B)^T op(A)^T + beta C^T   (C^T is n x m)
     * The C buffer doesn't move; we just swap M↔N, swap A↔B, and flip
     * both transposes.  The leading dim of the (new) row-major C is the
     * user's ldc (unchanged). */
    qmat_t Av, Bv;
    size_t mm, nn, kk = k;
    size_t ldc_eff = (size_t)ldc;
    if (row_major) {
        mm = m; nn = n;
        Av = make_view(A, lda, layout, transa);
        Bv = make_view(B, ldb, layout, transb);
    } else {
        /* For column-major C we instead compute C^T = α·op(B)^T·op(A)^T + β·C^T
         * which is the same buffer.  Derivation of strides:
         *   col-major A, transa=N: op(A)[i,j] addr = base + i + j*lda.
         *     op(A)^T[i,j] = op(A)[j,i] = base + j + i*lda → (rs=lda, cs=1).
         *     That matches make_view(RowMajor, NoTrans).
         *   col-major A, transa=T: op(A)[i,j] = A[j,i] = base + j + i*lda.
         *     op(A)^T[i,j] = base + i + j*lda → (rs=1, cs=lda).
         *     That matches make_view(RowMajor, Trans).
         * So for the rewrite we keep the original transpose flags; only the
         * A↔B and M↔N swap is needed. */
        mm = n; nn = m;
        Av = make_view(B, ldb, QblasRowMajor, transb);
        Bv = make_view(A, lda, QblasRowMajor, transa);
    }

    /* Pre-scale C (row-major view).  The same memory accesses cover the
     * col-major buffer because we've swapped m↔n. */
    scale_C(1, mm, nn, beta, C, ldc_eff);

    if (qiszero(alpha) || kk == 0) return;

    /* Make sure dispatch is initialised. */
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

    /* Scale nc down so every thread gets ≥ 1 jc-block.  We aim for at least
     * 2 blocks per thread to allow scheduling slack. */
    if (nthreads > 1) {
        size_t target_blocks = (size_t)nthreads * 2;
        size_t max_nc_per_block = (nn + target_blocks - 1) / target_blocks;
        if (max_nc_per_block < nc) nc = max_nc_per_block;
        if (nc < NR) nc = NR;  /* never smaller than one register tile */
    }
    if (nc > nn) nc = nn;
    size_t mc_pad = ((mc + MR - 1) / MR) * MR;
    size_t nc_pad = ((nc + NR - 1) / NR) * NR;

    size_t Bp_size = kc * nc_pad;
    size_t Ap_size = mc_pad * kc;

    /* Enumerate jc-blocks so we can hand each one to a thread.  Each
     * block i covers [i*nc, min((i+1)*nc, nn)). */
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

                    /* Now iterate over the (mc_use_pad x nc_use_pad) tile of
                     * the packed buffers, calling the micro-kernel. */
                    for (size_t jr = 0; jr < nc_use_pad; jr += NR) {
                        size_t n_real = (jc + jr + NR <= jc + nc_use) ? NR : (nc_use - jr);
                        const Sleef_quad *Bp_panel = Bp + (jr / NR) * (kc_use * NR);
                        for (size_t ir = 0; ir < mc_use_pad; ir += MR) {
                            size_t m_real = (ic + ir + MR <= ic + mc_use) ? MR : (mc_use - ir);
                            const Sleef_quad *Ap_panel = Ap + (ir / MR) * (kc_use * MR);

                            /* Same address formula in either layout: a
                             * col-major buffer with leading dim ldc, when
                             * viewed as row-major (n x m) with leading dim
                             * ldc, has identical element addresses. */
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
    } /* omp parallel */
}
