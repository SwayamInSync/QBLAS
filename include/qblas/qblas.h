/* QBLAS - High-performance quad-precision BLAS on top of SLEEF.
 *
 * This is the umbrella C header. It pulls in the CBLAS-style API plus
 * library-wide control routines (threading, dispatch info, version).
 *
 * The quad scalar type is `Sleef_quad` (defined by SLEEF). Memory layouts
 * follow CBLAS conventions: `QblasRowMajor` / `QblasColMajor`, and transpose
 * specifiers `QblasNoTrans` / `QblasTrans` / `QblasConjTrans` (the conjugate
 * variant is accepted for source-compat with CBLAS — quad is real, so it
 * behaves identically to `QblasTrans`).
 */
#ifndef QBLAS_H
#define QBLAS_H

#include <sleefquad.h>
#include <stddef.h>

#if defined(_WIN32) && !defined(QBLAS_STATIC)
#  ifdef QBLAS_BUILDING
#    define QBLAS_API __declspec(dllexport)
#  else
#    define QBLAS_API __declspec(dllimport)
#  endif
#else
#  define QBLAS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* CBLAS-style enums (kept binary-compatible with cblas.h ints) */
typedef enum {
    QblasRowMajor = 101,
    QblasColMajor = 102
} QBLAS_LAYOUT;

typedef enum {
    QblasNoTrans   = 111,
    QblasTrans     = 112,
    QblasConjTrans = 113   /* treated as Trans (quad is real) */
} QBLAS_TRANSPOSE;

typedef enum {
    QblasUpper = 121,
    QblasLower = 122
} QBLAS_UPLO;

typedef enum {
    QblasNonUnit = 131,
    QblasUnit    = 132
} QBLAS_DIAG;

typedef enum {
    QblasLeft  = 141,
    QblasRight = 142
} QBLAS_SIDE;

/* Library control */
QBLAS_API const char *qblas_get_version(void);
QBLAS_API const char *qblas_get_dispatch_tier(void); /* "avx512" | "avx2" | "sse2" | "neon" | "generic" */

QBLAS_API void qblas_set_num_threads(int n);
QBLAS_API int  qblas_get_num_threads(void);
QBLAS_API int  qblas_get_max_threads(void);

#include "qblas_level1.h"
#include "qblas_level2.h"
#include "qblas_level3.h"

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* QBLAS_H */
