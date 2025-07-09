#ifndef QUADBLAS_CORE_PLATFORM_HPP
#define QUADBLAS_CORE_PLATFORM_HPP

// Platform detection
#if defined(__x86_64__) || defined(_M_X64)
    #define QUADBLAS_X86_64
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define QUADBLAS_AARCH64
    #include <arm_neon.h>
#endif

// OpenMP detection
#ifdef _OPENMP
#include <omp.h>
#endif

// SLEEF includes
#include <sleefquad.h>

// Compatibility for SLEEF_QUAD_C macro
#ifndef SLEEF_QUAD_C
    #define SLEEF_QUAD_C(x) Sleef_cast_from_doubleq1(x)
#endif

#endif // QUADBLAS_CORE_PLATFORM_HPP