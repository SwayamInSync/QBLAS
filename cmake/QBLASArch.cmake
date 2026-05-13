# Detect target architecture and the set of ISA tiers we will build kernels for.
#
# Sets the cache variables:
#   QBLAS_ARCH       — x86_64 | aarch64 | generic
#   QBLAS_ISA_TIERS  — semicolon list of tier ids the build system should compile
#                      e.g. "generic;sse2;avx2;avx512" on x86_64
#                           "generic;neon" on aarch64
#
# Also exposes per-tier compile flag variables QBLAS_<TIER>_FLAGS for use when
# building the per-ISA kernel translation units.

include_guard()

include(CheckCCompilerFlag)

function(qblas_detect_arch)
    set(arch "generic")
    set(tiers "generic")

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(arch "x86_64")
        set(tiers "generic" "sse2" "avx2" "avx512")

        set(QBLAS_GENERIC_FLAGS ""                             CACHE INTERNAL "")
        set(QBLAS_SSE2_FLAGS    "-msse2"                       CACHE INTERNAL "")
        set(QBLAS_AVX2_FLAGS    "-mavx2;-mfma;-mbmi2"          CACHE INTERNAL "")
        set(QBLAS_AVX512_FLAGS  "-mavx512f;-mavx512dq;-mavx512bw;-mfma;-mbmi2" CACHE INTERNAL "")

    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(arch "aarch64")
        set(tiers "generic" "neon")

        set(QBLAS_GENERIC_FLAGS ""                             CACHE INTERNAL "")
        set(QBLAS_NEON_FLAGS    "" CACHE INTERNAL "")  # NEON is baseline on aarch64

    else()
        message(STATUS "QBLAS: unknown CMAKE_SYSTEM_PROCESSOR='${CMAKE_SYSTEM_PROCESSOR}', falling back to generic only")
    endif()

    set(QBLAS_ARCH      "${arch}"  CACHE INTERNAL "QBLAS target arch")
    set(QBLAS_ISA_TIERS "${tiers}" CACHE INTERNAL "QBLAS ISA tiers")
endfunction()
