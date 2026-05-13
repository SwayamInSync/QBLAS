/* NEON kernel tier — width-2 SLEEF quad vectors on AArch64. */
#define QBLAS_HAS_NEON 1
#define QV_WIDTH 2
#define QV_SUFFIX neon
#define QV_ISA_SUFFIX advsimd
#include "../kernels_template.h"

void qblas_register_neon(void) { qblas_register_kernels_neon(); }
