#define QBLAS_HAS_AVX2 1
#define QV_WIDTH 4
#define QV_SUFFIX avx2
#include "../kernels_template.h"

void qblas_register_avx2(void) { qblas_register_kernels_avx2(); }
