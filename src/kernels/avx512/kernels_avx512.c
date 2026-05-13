/* AVX-512F kernel tier — width-8 SLEEF quad vectors (Sleef_quadx8). */
#define QBLAS_HAS_AVX512 1
#define QV_WIDTH 8
#define QV_SUFFIX avx512
#include "../kernels_template.h"

void qblas_register_avx512(void) { qblas_register_kernels_avx512(); }
