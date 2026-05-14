#ifndef QBLAS_INTERNAL_H
#define QBLAS_INTERNAL_H

#include <qblas/qblas.h>
#include <sleefquad.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QBLAS_ZERO   Sleef_cast_from_doubleq1(0.0)
#define QBLAS_ONE    Sleef_cast_from_doubleq1(1.0)
#define QBLAS_NEGONE Sleef_cast_from_doubleq1(-1.0)

static inline Sleef_quad qadd(Sleef_quad a, Sleef_quad b) { return Sleef_addq1_u05(a, b); }
static inline Sleef_quad qsub(Sleef_quad a, Sleef_quad b) { return Sleef_subq1_u05(a, b); }
static inline Sleef_quad qmul(Sleef_quad a, Sleef_quad b) { return Sleef_mulq1_u05(a, b); }
static inline Sleef_quad qdiv(Sleef_quad a, Sleef_quad b) { return Sleef_divq1_u05(a, b); }
static inline Sleef_quad qfma(Sleef_quad a, Sleef_quad b, Sleef_quad c) {
    return Sleef_fmaq1_u05(a, b, c);
}
static inline Sleef_quad qneg(Sleef_quad a)  { return Sleef_negq1(a); }
static inline Sleef_quad qfabs(Sleef_quad a) { return Sleef_fabsq1(a); }
static inline Sleef_quad qsqrt(Sleef_quad a) { return Sleef_sqrtq1_u05(a); }

static inline bool qiszero(Sleef_quad a) { return Sleef_icmpeqq1(a, QBLAS_ZERO) != 0; }
static inline bool qisone(Sleef_quad a)  { return Sleef_icmpeqq1(a, QBLAS_ONE)  != 0; }

/* For incx<0 the vector starts at the high end and walks back. */
static inline ptrdiff_t qblas_stride_offset(int incx, int n) {
    return (incx < 0) ? (ptrdiff_t)(n - 1) * (ptrdiff_t)(-incx) : 0;
}

#define QBLAS_ALIGN 64

void *qblas_aligned_alloc(size_t bytes);
void  qblas_aligned_free(void *p);

typedef enum {
    QBLAS_TIER_GENERIC = 0,
    QBLAS_TIER_SSE2    = 1,
    QBLAS_TIER_NEON    = 2,
    QBLAS_TIER_AVX2    = 3,
    QBLAS_TIER_AVX512  = 4
} qblas_cpu_tier_t;

QBLAS_API qblas_cpu_tier_t qblas_cpu_tier(void);

int qblas_resolve_threads(size_t work_units, size_t per_unit_cost);

/* Runtime tunables populated once by qblas_dispatch_init() from CPUID /
 * sysconf / a timed empty OpenMP region. */
typedef struct {
    size_t l1_data;
    size_t l2;
    size_t l3;
    int    cores;
    size_t l1_thread_threshold;     /* elements */
    size_t gemv_thread_threshold;   /* m*n elements */
    size_t gemm_mc;
    size_t gemm_kc;
    size_t gemm_nc;
    size_t omp_overhead_cycles;
} qblas_tune_t;

QBLAS_API const qblas_tune_t *qblas_tune(void);

#define QBLAS_PARALLEL_THRESHOLD_L1_DEFAULT   16384
#define QBLAS_PARALLEL_THRESHOLD_GEMV_DEFAULT 16384
#define QBLAS_PARALLEL_THRESHOLD_GEMM         64

#if defined(__GNUC__) || defined(__clang__)
#  define QBLAS_LIKELY(x)   __builtin_expect(!!(x), 1)
#  define QBLAS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#  define QBLAS_INLINE     static inline __attribute__((always_inline))
#  define QBLAS_NOINLINE   __attribute__((noinline))
#else
#  define QBLAS_LIKELY(x)   (x)
#  define QBLAS_UNLIKELY(x) (x)
#  define QBLAS_INLINE     static inline
#  define QBLAS_NOINLINE
#endif

#ifdef __cplusplus
}
#endif

#endif
