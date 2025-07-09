#ifndef QUADBLAS_HPP
#define QUADBLAS_HPP

/**
 * QuadBLAS - High Performance Quad Precision BLAS Library
 * 
 * A header-only library providing optimized linear algebra operations
 * for IEEE 754 quadruple precision (128-bit) floating point numbers.
 * 
 * Features:
 * - SIMD vectorization (x86-64 SSE/AVX, ARM64 NEON)
 * - OpenMP multi-threading
 * - Cache-optimized algorithms
 * - Both C and C++ interfaces
 * - Header-only design for easy integration
 * 
 * Built on top of the SLEEF mathematical library.
 * 
 * Copyright (c) 2025 QuadBLAS Project
 * Licensed under the Apache License, Version 2.0
 * See LICENSE file in the root directory or visit:
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Include all public headers in dependency order

// Core fundamentals
#include "core/platform.hpp"
#include "core/constants.hpp" 
#include "core/types.hpp"

// Memory management
#include "memory/allocation.hpp"

// SIMD abstractions
#include "simd/quad_vector.hpp"

// Threading utilities
#include "threading/openmp_utils.hpp"

// Internal details
#include "detail/blocking.hpp"

// Algorithm implementations (Level 1, 2, 3 BLAS)
#include "algorithms/level1.hpp"
#include "algorithms/level2.hpp"
#include "algorithms/level3.hpp"

// Public interfaces
#include "interface/c_interface.hpp"
#include "interface/cpp_classes.hpp"

// Version information
namespace QuadBLAS {
    constexpr const char* VERSION = "1.0.0";
    constexpr int VERSION_MAJOR = 1;
    constexpr int VERSION_MINOR = 0;
    constexpr int VERSION_PATCH = 0;
}

#endif // QUADBLAS_HPP