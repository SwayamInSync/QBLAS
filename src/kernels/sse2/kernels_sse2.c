/* SSE2 kernel tier — width-2 SLEEF quad vectors. */
#define QBLAS_HAS_SSE2 1
#define QV_WIDTH 2
#define QV_SUFFIX sse2
#define QV_ISA_SUFFIX sse2
#include "../kernels_template.h"

void qblas_register_sse2(void) { qblas_register_kernels_sse2(); }
