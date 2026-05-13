/* Generic (scalar) kernel tier — always built. */
#define QV_WIDTH 1
#define QV_SUFFIX generic
#include "../kernels_template.h"

void qblas_register_generic(void) { qblas_register_kernels_generic(); }
