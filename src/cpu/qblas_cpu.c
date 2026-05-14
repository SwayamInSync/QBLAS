#include "common/qblas_internal.h"
#include "common/qblas_dispatch.h"

#ifndef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 200809L
#endif
/* macOS hides _SC_NPROCESSORS_ONLN behind _DARWIN_C_SOURCE when
 * _POSIX_C_SOURCE is defined (it is a BSD extension, not in POSIX). */
#ifdef __APPLE__
#  ifndef _DARWIN_C_SOURCE
#    define _DARWIN_C_SOURCE 1
#  endif
#endif

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

#ifdef QBLAS_X86_64
static int has_xgetbv_avx_state(void) {
    unsigned int eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (eax & 0x6) == 0x6;             /* XMM | YMM */
}
static int has_xgetbv_avx512_state(void) {
    unsigned int eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (eax & 0xE6) == 0xE6;           /* + opmask | ZMM_Hi256 | Hi16_ZMM */
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

    int avx_ok    = has_osxsave && has_avx && has_xgetbv_avx_state();
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

static void detect_caches(qblas_tune_t *t) {
#ifdef QBLAS_X86_64
    for (unsigned i = 0; i < 8; ++i) {
        unsigned int a, b, c, d;
        if (!__get_cpuid_count(4, i, &a, &b, &c, &d)) break;
        unsigned type   = a & 0x1F;
        if (type == 0) break;
        unsigned level  = (a >> 5) & 0x7;
        /* cache size = (ways+1) * (partitions+1) * (line_size+1) * (sets+1) */
        unsigned ways       = ((b >> 22) & 0x3FF) + 1;
        unsigned partitions = ((b >> 12) & 0x3FF) + 1;
        unsigned line_size  = (b & 0xFFF) + 1;
        unsigned sets       = c + 1;
        size_t cache_bytes  = (size_t)ways * partitions * line_size * sets;
        if      (level == 1 && type != 2) t->l1_data = cache_bytes;
        else if (level == 2 && type != 2) t->l2      = cache_bytes;
        else if (level == 3 && type != 2) t->l3      = cache_bytes;
    }
#else
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
#ifdef _SC_NPROCESSORS_ONLN
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    t->cores = (ncpu > 0) ? (int)ncpu : 1;
#else
    t->cores = 1;
#endif
}

#ifdef _OPENMP
static inline uint64_t ts_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
#endif

/* Measures the cost of an empty parallel region so qblas_resolve_threads
 * can decide if a piece of work justifies forking. */
static size_t measure_omp_overhead_cycles(int max_threads) {
#ifdef _OPENMP
    if (max_threads <= 1) return 1;
    int warm = 0;
    for (int i = 0; i < 4; ++i) {
        #pragma omp parallel num_threads(max_threads) reduction(+:warm)
        warm += 1;
    }
    (void)warm;
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
    size_t cycles = (size_t)(ns_per_spawn * 3.0);   /* approx at 3 GHz */
    if (cycles < 1024) cycles = 1024;
    if (cycles > 1u << 20) cycles = 1u << 20;
    return cycles;
#else
    (void)max_threads;
    return 1;
#endif
}

static void derive_tunables(qblas_tune_t *t) {
    size_t l1_thr = t->omp_overhead_cycles / 16;
    if (l1_thr < 1024)  l1_thr = 1024;
    if (l1_thr > 16384) l1_thr = 16384;
    t->l1_thread_threshold   = l1_thr;
    t->gemv_thread_threshold = l1_thr * 2;

    /* GEMM blocking sized to fit the live micro-kernel set in L1, A panel
     * in ~80% of L2, B panel in L3. quad = 16 bytes per element. */
    const size_t MR = 4, NR = 4, quad_bytes = 16;
    size_t kc = (t->l1_data / 2 - MR*NR*quad_bytes) / ((MR + NR) * quad_bytes);
    if (kc > 256) kc = 256;
    if (kc < 32)  kc = 32;
    kc &= ~(size_t)7;
    t->gemm_kc = kc;

    size_t mc = (t->l2 * 4) / (5 * kc * quad_bytes);
    if (mc > 512) mc = 512;
    if (mc < MR)  mc = MR;
    mc -= mc % MR;
    t->gemm_mc = mc;

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

    qblas_register_generic();

    g_tier = detect_tier();

    detect_caches(&g_tune);
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
#else
    int max_threads = 1;
#endif
    g_tune.omp_overhead_cycles = measure_omp_overhead_cycles(max_threads);
    derive_tunables(&g_tune);

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

    /* QBLAS_DISPATCH=tier_name forces a tier the binary contains *and* the
     * CPU supports; otherwise the override is reported and dropped. */
    qblas_cpu_tier_t auto_tier = g_tier;
    const char *env = getenv("QBLAS_DISPATCH");
    if (env) {
        qblas_cpu_tier_t want = (qblas_cpu_tier_t)(-1);
        if      (strcmp(env, "generic") == 0) want = QBLAS_TIER_GENERIC;
        else if (strcmp(env, "sse2")    == 0) want = QBLAS_TIER_SSE2;
        else if (strcmp(env, "neon")    == 0) want = QBLAS_TIER_NEON;
        else if (strcmp(env, "avx2")    == 0) want = QBLAS_TIER_AVX2;
        else if (strcmp(env, "avx512")  == 0) want = QBLAS_TIER_AVX512;

        if ((int)want >= 0 && (int)want > (int)auto_tier) {
            const char *auto_name;
            switch (auto_tier) {
            case QBLAS_TIER_AVX512: auto_name = "avx512"; break;
            case QBLAS_TIER_AVX2:   auto_name = "avx2";   break;
            case QBLAS_TIER_NEON:   auto_name = "neon";   break;
            case QBLAS_TIER_SSE2:   auto_name = "sse2";   break;
            default:                auto_name = "generic"; break;
            }
            fprintf(stderr,
                "qblas: QBLAS_DISPATCH=%s is higher than the CPU's detected "
                "tier (%s); ignoring.\n",
                env, auto_name);
            want = (qblas_cpu_tier_t)(-1);
        }

        if ((int)want >= 0) {
            qblas_register_generic();
            switch (want) {
            case QBLAS_TIER_GENERIC: g_tier = QBLAS_TIER_GENERIC; break;
#ifdef QBLAS_HAS_SSE2
            case QBLAS_TIER_SSE2:    qblas_register_sse2();   g_tier = QBLAS_TIER_SSE2;   break;
#endif
#ifdef QBLAS_HAS_NEON
            case QBLAS_TIER_NEON:    qblas_register_neon();   g_tier = QBLAS_TIER_NEON;   break;
#endif
#ifdef QBLAS_HAS_AVX2
            case QBLAS_TIER_AVX2:    qblas_register_avx2();   g_tier = QBLAS_TIER_AVX2;   break;
#endif
#ifdef QBLAS_HAS_AVX512
            case QBLAS_TIER_AVX512:  qblas_register_avx512(); g_tier = QBLAS_TIER_AVX512; break;
#endif
            default: break;
            }
        }
    }

    g_initialized = 1;
}

__attribute__((constructor))
static void qblas_lib_init(void) { qblas_dispatch_init(); }

const char *qblas_get_version(void) { return "QBLAS 1.5.0"; }

void *qblas_aligned_alloc(size_t bytes) {
    size_t rounded = (bytes + QBLAS_ALIGN - 1) & ~(size_t)(QBLAS_ALIGN - 1);
    void *p = NULL;
    if (posix_memalign(&p, QBLAS_ALIGN, rounded == 0 ? QBLAS_ALIGN : rounded) != 0) return NULL;
    return p;
}
void qblas_aligned_free(void *p) { free(p); }

#ifdef _OPENMP
static int g_user_thread_cap = 0;
#endif

/* qblas_set_num_threads only caps qblas-internal work distribution; it
 * never calls omp_set_num_threads, so the host application's OpenMP
 * state is left alone. */
void qblas_set_num_threads(int n) {
#ifdef _OPENMP
    g_user_thread_cap = n > 0 ? n : 0;
#else
    (void)n;
#endif
}
int qblas_get_num_threads(void) {
#ifdef _OPENMP
    int omp_max = omp_get_max_threads();
    if (g_user_thread_cap > 0 && g_user_thread_cap < omp_max)
        return g_user_thread_cap;
    return omp_max;
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

/* Returns a thread count where each thread has at least one omp_overhead
 * worth of work; small problems and big machines naturally pick small
 * thread counts instead of fanning out and paying for fork/join. */
int qblas_resolve_threads(size_t work_units, size_t per_unit_cost) {
#ifdef _OPENMP
    int max = omp_get_max_threads();
    if (max <= 1) return 1;
    if (per_unit_cost == 0) per_unit_cost = 1;
    const size_t quad_cycles_per_op = 32;
    size_t overhead = g_tune.omp_overhead_cycles;
    if (overhead < 1024) overhead = 1024;
    size_t total_cycles = work_units * per_unit_cost * quad_cycles_per_op;
    if (total_cycles < overhead * 2) return 1;
    size_t t = total_cycles / overhead;
    if ((int)t > max) t = (size_t)max;
    if (g_user_thread_cap && (int)t > g_user_thread_cap) t = (size_t)g_user_thread_cap;
    if (t < 1) t = 1;
    return (int)t;
#else
    (void)work_units; (void)per_unit_cost;
    return 1;
#endif
}
