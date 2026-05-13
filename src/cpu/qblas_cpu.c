/* CPU feature detection + global library init (dispatch table population).
 *
 * On x86_64 we use the CPUID instruction (via <cpuid.h>).  We never assume
 * runtime support for an ISA that the binary wasn't compiled to contain,
 * so each kernel TU is opt-in: a tier's kernel symbols are only registered
 * if its preprocessor guard is enabled (QBLAS_HAS_AVX2, etc).
 */

#include "common/qblas_internal.h"
#include "common/qblas_dispatch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
#  include <cpuid.h>
#  define QBLAS_X86_64 1
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#  define QBLAS_AARCH64 1
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

static qblas_cpu_tier_t g_tier = QBLAS_TIER_GENERIC;
static int g_initialized = 0;
static qblas_tune_t g_tune = {
    /* Conservative defaults; overwritten at init. */
    .l1_data = 32 * 1024,
    .l2      = 256 * 1024,
    .l3      = 8  * 1024 * 1024,
    .cores   = 1,
    .l1_thread_threshold   = QBLAS_PARALLEL_THRESHOLD_L1_DEFAULT,
    .gemv_thread_threshold = QBLAS_PARALLEL_THRESHOLD_GEMV_DEFAULT,
    .gemm_mc = 128,
    .gemm_kc = 128,
    .gemm_nc = 512,
    .omp_overhead_cycles = 8192,
};
const qblas_tune_t *qblas_tune(void) { return &g_tune; }

/* ---------- CPUID helpers (x86 only) ---------- */
#ifdef QBLAS_X86_64
static int has_xgetbv_avx_state(void) {
    /* Need OS to enable XMM/YMM state for AVX, ZMM/K state for AVX-512. */
    unsigned int eax, edx;
    /* xgetbv 0 */
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    /* bit 1 = XMM, bit 2 = YMM */
    return (eax & 0x6) == 0x6;
}
static int has_xgetbv_avx512_state(void) {
    unsigned int eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    /* bits 5,6,7 = opmask, ZMM_Hi256, Hi16_ZMM */
    return (eax & 0xE6) == 0xE6;
}
#endif

static qblas_cpu_tier_t detect_tier(void) {
#ifdef QBLAS_X86_64
    unsigned int eax, ebx, ecx, edx;
    int has_sse2 = 0, has_avx = 0, has_avx2 = 0, has_fma = 0;
    int has_avx512f = 0, has_avx512dq = 0;
    int has_osxsave = 0;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        has_sse2    = (edx & (1u << 26)) != 0;
        has_fma     = (ecx & (1u << 12)) != 0;
        has_osxsave = (ecx & (1u << 27)) != 0;
        has_avx     = (ecx & (1u << 28)) != 0;
    }
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        has_avx2     = (ebx & (1u << 5))  != 0;
        has_avx512f  = (ebx & (1u << 16)) != 0;
        has_avx512dq = (ebx & (1u << 17)) != 0;
    }

    /* OS support check via XGETBV - only valid if OSXSAVE is set. */
    int avx_ok = has_osxsave && has_avx && has_xgetbv_avx_state();
    int avx512_ok = avx_ok && has_xgetbv_avx512_state();

#ifdef QBLAS_HAS_AVX512
    if (avx512_ok && has_avx512f && has_avx512dq) return QBLAS_TIER_AVX512;
#endif
#ifdef QBLAS_HAS_AVX2
    if (avx_ok && has_avx2 && has_fma) return QBLAS_TIER_AVX2;
#endif
#ifdef QBLAS_HAS_SSE2
    if (has_sse2) return QBLAS_TIER_SSE2;
#endif
    (void)avx_ok;
    return QBLAS_TIER_GENERIC;
#elif defined(QBLAS_AARCH64)
#  ifdef QBLAS_HAS_NEON
    return QBLAS_TIER_NEON;
#  else
    return QBLAS_TIER_GENERIC;
#  endif
#else
    return QBLAS_TIER_GENERIC;
#endif
}

/* ---------- Cache hierarchy detection ---------- */
/* On x86 we read CPUID leaf 4 (deterministic cache parameters).
 * On POSIX we fall back to sysconf which works on Linux for L1 size at least.
 * On failure we keep the static defaults. */
static void detect_caches(qblas_tune_t *t) {
#ifdef QBLAS_X86_64
    /* Leaf 4 iterates over caches; index 0 = L1d, 1 = L1i, 2 = L2, 3 = L3 typically. */
    for (unsigned i = 0; i < 8; ++i) {
        unsigned int a, b, c, d;
        if (!__get_cpuid_count(4, i, &a, &b, &c, &d)) break;
        unsigned type   = a & 0x1F;
        if (type == 0) break;                /* no more caches */
        unsigned level  = (a >> 5) & 0x7;
        /* Cache size = (ways+1)*(partitions+1)*(line_size+1)*(sets+1) */
        unsigned ways       = ((b >> 22) & 0x3FF) + 1;
        unsigned partitions = ((b >> 12) & 0x3FF) + 1;
        unsigned line_size  = (b & 0xFFF) + 1;
        unsigned sets       = c + 1;
        size_t cache_bytes  = (size_t)ways * partitions * line_size * sets;
        /* type 1 = data, type 2 = instruction, type 3 = unified */
        if (level == 1 && type != 2) t->l1_data = cache_bytes;
        else if (level == 2 && type != 2) t->l2 = cache_bytes;
        else if (level == 3 && type != 2) t->l3 = cache_bytes;
    }
#else
    /* POSIX fallback: only L1 is reliably exposed. */
#  ifdef _SC_LEVEL1_DCACHE_SIZE
    long v = sysconf(_SC_LEVEL1_DCACHE_SIZE); if (v > 0) t->l1_data = (size_t)v;
#  endif
#  ifdef _SC_LEVEL2_CACHE_SIZE
    v = sysconf(_SC_LEVEL2_CACHE_SIZE); if (v > 0) t->l2 = (size_t)v;
#  endif
#  ifdef _SC_LEVEL3_CACHE_SIZE
    v = sysconf(_SC_LEVEL3_CACHE_SIZE); if (v > 0) t->l3 = (size_t)v;
#  endif
#endif
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    t->cores = (ncpu > 0) ? (int)ncpu : 1;
}

/* Time one CLOCK_MONOTONIC nanosecond. */
static inline uint64_t ts_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Measure the per-spawn cost of an empty OpenMP parallel region.  This gives
 * us a real fork/join number for the host machine instead of guessing — on a
 * tiny laptop with libgomp it's a few µs; on libomp tuned servers it can be
 * under a µs.  We then convert to "cycles" using a rough 3 GHz estimate so
 * the cost units match qblas_resolve_threads's quad-FMA-cycle accounting. */
static size_t measure_omp_overhead_cycles(int max_threads) {
#ifdef _OPENMP
    if (max_threads <= 1) return 1; /* irrelevant */
    /* Warm up the thread pool. */
    int warm = 0;
    for (int i = 0; i < 4; ++i) {
        #pragma omp parallel num_threads(max_threads) reduction(+:warm)
        warm += 1;
    }
    (void)warm;
    /* Time many empty parallel regions. */
    const int iters = 256;
    uint64_t t0 = ts_ns();
    int sink = 0;
    for (int i = 0; i < iters; ++i) {
        #pragma omp parallel num_threads(max_threads) reduction(+:sink)
        sink += 1;
    }
    uint64_t t1 = ts_ns();
    (void)sink;
    double ns_per_spawn = (double)(t1 - t0) / iters;
    /* Convert at ~3 GHz; not exact but the absolute number isn't critical —
     * what matters is the *ratio* to per-op cost in qblas_resolve_threads. */
    size_t cycles = (size_t)(ns_per_spawn * 3.0);
    if (cycles < 1024) cycles = 1024;
    if (cycles > 1u << 20) cycles = 1u << 20;
    return cycles;
#else
    (void)max_threads;
    return 1;
#endif
}

/* Derive runtime tunables once caches + thread count + omp overhead are known. */
static void derive_tunables(qblas_tune_t *t) {
    /* L1 threshold: switch to threaded when total quad-op cycles exceed
     * fork-cost.  Each quad op is ~32 cycles, so threshold in elements is
     * ~ omp_overhead / 32.  Floor 1024 — below that even threading overhead
     * dominates regardless of CPU.  The ratio /16 instead of /32 gives ~2×
     * headroom on the break-even point, matching empirical observations. */
    size_t l1_thr = t->omp_overhead_cycles / 16;
    if (l1_thr < 1024) l1_thr = 1024;
    if (l1_thr > 16384) l1_thr = 16384;
    t->l1_thread_threshold = l1_thr;
    t->gemv_thread_threshold = l1_thr * 2;

    /* GEMM blocking from cache sizes.
     *   kc panel size: keep MR*kc + kc*NR + MR*NR (the live micro-kernel
     *   working set) under L1/2.  MR=NR=4, quad=16 bytes:
     *     2*4*kc*16 + 4*4*16 = 128*kc + 256  ≤ L1/2
     *     kc ≤ (L1/2 - 256) / 128
     *   mc: A panel mc*kc must fit L2/2 → mc ≤ L2/(2*kc*16).
     *   nc: B panel kc*nc must fit L3/threads → nc ≤ L3/(cores*kc*16). */
    const size_t MR = 4, NR = 4, quad_bytes = 16;
    size_t kc = (t->l1_data / 2 - MR*NR*quad_bytes) / ((MR + NR) * quad_bytes);
    if (kc > 256) kc = 256;
    if (kc < 32)  kc = 32;
    /* Round down to multiple of 8 for clean unroll. */
    kc &= ~(size_t)7;
    t->gemm_kc = kc;

    /* OpenBLAS sizes its A panel close to the full L2 (the B panel slice and
     * C tile are streamed, not retained), so we use ~80 % of L2 for A. */
    size_t mc = (t->l2 * 4) / (5 * kc * quad_bytes);
    if (mc > 512) mc = 512;
    if (mc < MR)  mc = MR;
    mc -= mc % MR;
    t->gemm_mc = mc;

    /* nc is the ceiling on a B-panel.  OpenBLAS uses nc ~ L3/(kc*sizeof),
     * NOT L3/cores — per-thread parallelism is handled by the caller, which
     * subdivides nc-blocks across threads.  Make this a generous ceiling
     * (whole-L3 / kc) and let qblas_gemm()'s nc-scaling do the right thing
     * for small problems with many threads. */
    size_t nc = t->l3 / (kc * quad_bytes);
    if (nc > 4096) nc = 4096;
    if (nc < NR)   nc = NR;
    nc -= nc % NR;
    t->gemm_nc = nc;
}

const char *qblas_get_dispatch_tier(void) {
    switch (qblas_cpu_tier()) {
    case QBLAS_TIER_AVX512:  return "avx512";
    case QBLAS_TIER_AVX2:    return "avx2";
    case QBLAS_TIER_SSE2:    return "sse2";
    case QBLAS_TIER_NEON:    return "neon";
    default:                 return "generic";
    }
}

qblas_cpu_tier_t qblas_cpu_tier(void) {
    if (!g_initialized) qblas_dispatch_init();
    return g_tier;
}

/* ---------- Dispatch table definitions ---------- */
qdot_fn   qblas_dispatch_qdot   = NULL;
qaxpy_fn  qblas_dispatch_qaxpy  = NULL;
qscal_fn  qblas_dispatch_qscal  = NULL;
qasum_fn  qblas_dispatch_qasum  = NULL;
qiamax_fn qblas_dispatch_qiamax = NULL;
qgemv_n_fn qblas_dispatch_qgemv_n = NULL;
qgemv_t_fn qblas_dispatch_qgemv_t = NULL;
qgemm_kernel_fn qblas_dispatch_qgemm_kernel = NULL;
size_t qblas_dispatch_qgemm_MR = 0;
size_t qblas_dispatch_qgemm_NR = 0;

/* Forward decls for kernel registries.  Each tier registers its kernels
 * by name; the registration symbol exists only when that tier is built. */
void qblas_register_generic(void);
#ifdef QBLAS_HAS_SSE2
void qblas_register_sse2(void);
#endif
#ifdef QBLAS_HAS_AVX2
void qblas_register_avx2(void);
#endif
#ifdef QBLAS_HAS_AVX512
void qblas_register_avx512(void);
#endif
#ifdef QBLAS_HAS_NEON
void qblas_register_neon(void);
#endif

void qblas_dispatch_init(void) {
    if (g_initialized) return;

    /* Always start with generic so every pointer is non-NULL. */
    qblas_register_generic();

    g_tier = detect_tier();

    /* Discover the host's cache hierarchy and thread count, then time the
     * cost of an empty OpenMP region.  Everything downstream (parallel
     * thresholds, GEMM blocking) reads from g_tune. */
    detect_caches(&g_tune);
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
#else
    int max_threads = 1;
#endif
    g_tune.omp_overhead_cycles = measure_omp_overhead_cycles(max_threads);
    derive_tunables(&g_tune);

    /* Upgrade to richer tiers in order; later overrides win. */
#ifdef QBLAS_HAS_SSE2
    if (g_tier >= QBLAS_TIER_SSE2) qblas_register_sse2();
#endif
#ifdef QBLAS_HAS_NEON
    if (g_tier >= QBLAS_TIER_NEON) qblas_register_neon();
#endif
#ifdef QBLAS_HAS_AVX2
    if (g_tier >= QBLAS_TIER_AVX2) qblas_register_avx2();
#endif
#ifdef QBLAS_HAS_AVX512
    if (g_tier >= QBLAS_TIER_AVX512) qblas_register_avx512();
#endif

    /* Honour env override: QBLAS_DISPATCH=generic|sse2|avx2|avx512|neon */
    const char *env = getenv("QBLAS_DISPATCH");
    if (env) {
        qblas_register_generic();
        if (0) {}
#ifdef QBLAS_HAS_SSE2
        else if (strcmp(env, "sse2") == 0)   { qblas_register_sse2();   g_tier = QBLAS_TIER_SSE2;   }
#endif
#ifdef QBLAS_HAS_NEON
        else if (strcmp(env, "neon") == 0)   { qblas_register_neon();   g_tier = QBLAS_TIER_NEON;   }
#endif
#ifdef QBLAS_HAS_AVX2
        else if (strcmp(env, "avx2") == 0)   { qblas_register_avx2();   g_tier = QBLAS_TIER_AVX2;   }
#endif
#ifdef QBLAS_HAS_AVX512
        else if (strcmp(env, "avx512") == 0) { qblas_register_avx512(); g_tier = QBLAS_TIER_AVX512; }
#endif
        else if (strcmp(env, "generic") == 0) { g_tier = QBLAS_TIER_GENERIC; }
    }

    g_initialized = 1;
}

/* Library constructor: detect once and populate. */
__attribute__((constructor))
static void qblas_lib_init(void) { qblas_dispatch_init(); }

/* ---------- Public library control ---------- */
const char *qblas_get_version(void) { return "QBLAS 0.1.0"; }

/* ---------- Aligned alloc / free ---------- */
void *qblas_aligned_alloc(size_t bytes) {
    /* Round up to multiple of alignment to satisfy posix_memalign. */
    size_t rounded = (bytes + QBLAS_ALIGN - 1) & ~(size_t)(QBLAS_ALIGN - 1);
    void *p = NULL;
    if (posix_memalign(&p, QBLAS_ALIGN, rounded == 0 ? QBLAS_ALIGN : rounded) != 0) return NULL;
    return p;
}
void qblas_aligned_free(void *p) { free(p); }

/* ---------- Threading ---------- */
#ifdef _OPENMP
static int g_user_thread_cap = 0; /* 0 = follow omp default */
#endif

void qblas_set_num_threads(int n) {
#ifdef _OPENMP
    g_user_thread_cap = n > 0 ? n : 0;
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}
int qblas_get_num_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
int qblas_get_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* Decide how many threads to actually use for a piece of work.  Avoids
 * paying parallel-region overhead when the work is too small.
 *
 * Quad ops are heavy: each SLEEF quad FMA is ~30 cycles of software DD
 * math, so even short loops have non-trivial total cycle counts and
 * benefit from threading.  We multiply the caller's abstract "ops" by
 * an expected cycles-per-quad-op estimate before comparing to the
 * fork/join overhead. */
int qblas_resolve_threads(size_t work_units, size_t per_unit_cost) {
#ifdef _OPENMP
    int max = omp_get_max_threads();
    if (max <= 1) return 1;
    if (per_unit_cost == 0) per_unit_cost = 1;
    const size_t quad_cycles_per_op = 32;     /* SLEEF q4 FMA cost */
    const size_t fork_join_cycles   = 8192;   /* approx OMP parallel-region */
    size_t total_cycles = work_units * per_unit_cost * quad_cycles_per_op;
    if (total_cycles < fork_join_cycles * 2) return 1;
    size_t t = total_cycles / fork_join_cycles;
    if ((int)t > max) t = (size_t)max;
    if (g_user_thread_cap && (int)t > g_user_thread_cap) t = (size_t)g_user_thread_cap;
    if (t < 1) t = 1;
    return (int)t;
#else
    (void)work_units; (void)per_unit_cost;
    return 1;
#endif
}
