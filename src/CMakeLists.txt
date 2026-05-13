# QBLAS library — built as a single shared (or static) library with multiple
# internal "kernel tiers" compiled with their own SIMD flags.  Runtime
# dispatch selects the best tier at library init.

# ---------------------------------------------------------------------------
# Core sources (always built)
# ---------------------------------------------------------------------------
set(QBLAS_CORE_SOURCES
    cpu/qblas_cpu.c
    level1/qblas_level1.c
    level2/qblas_level2.c
    level3/qblas_gemm.c
    level3/qblas_level3_extra.c)

if(QBLAS_BUILD_SHARED)
    add_library(qblas SHARED ${QBLAS_CORE_SOURCES})
else()
    add_library(qblas STATIC ${QBLAS_CORE_SOURCES})
endif()

target_compile_definitions(qblas PRIVATE QBLAS_BUILDING)
target_include_directories(qblas
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(qblas PUBLIC qblas::sleef)
if(QBLAS_USE_OPENMP)
    target_link_libraries(qblas PUBLIC OpenMP::OpenMP_C)
endif()

set_target_properties(qblas PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})


# Baseline flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(qblas PRIVATE
        -Wall -Wextra -Wno-unused-parameter
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O3 -g>
        $<$<CONFIG:Debug>:-O0 -g>)
    if(QBLAS_NATIVE)
        target_compile_options(qblas PRIVATE -march=native)
    endif()
endif()

# ---------------------------------------------------------------------------
# Per-tier kernel libraries
#
# Each tier is built as an OBJECT library with its own -march flags, then
# linked into the main qblas library.  The registration symbol from each
# tier is referenced by qblas_cpu.c; with --whole-archive (or visibility
# tricks) we ensure none of them are stripped.
# ---------------------------------------------------------------------------
function(qblas_add_tier name width flags has_define)
    set(target qblas_kernels_${name})
    add_library(${target} OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${name}/kernels_${name}.c)
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE qblas::sleef)
    target_compile_definitions(${target} PRIVATE
        QBLAS_BUILDING
        ${has_define}=1)
    if(flags)
        target_compile_options(${target} PRIVATE ${flags})
    endif()
    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_STANDARD 11)
    # Hidden visibility so only the registration function is exported
    # through qblas itself (we use it locally).
endfunction()

# Always-on generic tier.
qblas_add_tier(generic 1 "" QBLAS_HAS_GENERIC)

# x86_64 tiers.
if(QBLAS_ARCH STREQUAL "x86_64")
    qblas_add_tier(sse2   2 "${QBLAS_SSE2_FLAGS}"   QBLAS_HAS_SSE2)
    qblas_add_tier(avx2   4 "${QBLAS_AVX2_FLAGS}"   QBLAS_HAS_AVX2)
    qblas_add_tier(avx512 8 "${QBLAS_AVX512_FLAGS}" QBLAS_HAS_AVX512)

    target_link_libraries(qblas PRIVATE
        qblas_kernels_generic
        qblas_kernels_sse2
        qblas_kernels_avx2
        qblas_kernels_avx512)
    target_compile_definitions(qblas PRIVATE
        QBLAS_HAS_GENERIC=1
        QBLAS_HAS_SSE2=1
        QBLAS_HAS_AVX2=1
        QBLAS_HAS_AVX512=1)
elseif(QBLAS_ARCH STREQUAL "aarch64")
    qblas_add_tier(neon 2 "${QBLAS_NEON_FLAGS}" QBLAS_HAS_NEON)
    target_link_libraries(qblas PRIVATE qblas_kernels_generic qblas_kernels_neon)
    target_compile_definitions(qblas PRIVATE
        QBLAS_HAS_GENERIC=1
        QBLAS_HAS_NEON=1)
else()
    target_link_libraries(qblas PRIVATE qblas_kernels_generic)
    target_compile_definitions(qblas PRIVATE QBLAS_HAS_GENERIC=1)
endif()

# Export the library name as `qblas::qblas` for downstream consumers.
add_library(qblas::qblas ALIAS qblas)

# Install rules (basic — enough for `cmake --install`).
include(GNUInstallDirs)
install(TARGETS qblas
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/qblas
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
