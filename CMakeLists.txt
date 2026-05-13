cmake_minimum_required(VERSION 3.16)
project(qblas
    VERSION 0.1.0
    LANGUAGES C CXX
    DESCRIPTION "QBLAS: high-performance BLAS for IEEE-754 binary128 (quad precision) on top of SLEEF")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(QBLAS_BUILD_TESTS    "Build correctness tests"      ON)
option(QBLAS_BUILD_BENCH    "Build Google Benchmark suite" ON)
option(QBLAS_BUILD_SHARED   "Build shared library"         ON)
option(QBLAS_USE_OPENMP     "Enable OpenMP threading"      ON)
option(QBLAS_NATIVE         "Tune top-level code for the host CPU" OFF)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------
include(CheckCSourceCompiles)
include(CheckCCompilerFlag)
include(QBLASArch)
qblas_detect_arch()
message(STATUS "QBLAS target arch: ${QBLAS_ARCH}")
message(STATUS "QBLAS ISA tiers: ${QBLAS_ISA_TIERS}")

# ---------------------------------------------------------------------------
# Dependencies: SLEEF
# ---------------------------------------------------------------------------
# Allow user override via SLEEF_ROOT. Otherwise we look for a local build under
# .sleef-prefix (populated by scripts/bootstrap.sh) before falling back to system.
if(DEFINED ENV{SLEEF_ROOT} AND NOT SLEEF_ROOT)
    set(SLEEF_ROOT "$ENV{SLEEF_ROOT}")
endif()
if(NOT SLEEF_ROOT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.sleef-prefix/include/sleefquad.h")
    set(SLEEF_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.sleef-prefix")
endif()

find_path(SLEEF_INCLUDE_DIR sleefquad.h
    HINTS "${SLEEF_ROOT}/include" /usr/local/include /usr/include)
find_library(SLEEF_LIBRARY      NAMES sleef     HINTS "${SLEEF_ROOT}/lib" /usr/local/lib /usr/lib)
find_library(SLEEFQUAD_LIBRARY  NAMES sleefquad HINTS "${SLEEF_ROOT}/lib" /usr/local/lib /usr/lib)

if(NOT SLEEF_INCLUDE_DIR OR NOT SLEEF_LIBRARY OR NOT SLEEFQUAD_LIBRARY)
    message(FATAL_ERROR
        "SLEEF (with quad support) not found. "
        "Set SLEEF_ROOT or run scripts/bootstrap_sleef.sh.")
endif()
message(STATUS "SLEEF: ${SLEEFQUAD_LIBRARY}")

find_library(TLFLOAT_LIBRARY NAMES tlfloat HINTS "${SLEEF_ROOT}/lib" /usr/local/lib /usr/lib)

add_library(qblas::sleef INTERFACE IMPORTED)
target_include_directories(qblas::sleef INTERFACE "${SLEEF_INCLUDE_DIR}")
target_link_libraries(qblas::sleef INTERFACE "${SLEEFQUAD_LIBRARY}" "${SLEEF_LIBRARY}")
if(TLFLOAT_LIBRARY)
    target_link_libraries(qblas::sleef INTERFACE "${TLFLOAT_LIBRARY}")
endif()
# Embed RPATH so tests/benches find the SLEEF .sleef-prefix libs (and
# transitively libtlfloat) without LD_LIBRARY_PATH gymnastics. Use the
# old DT_RPATH tag (--disable-new-dtags) because DT_RUNPATH is not
# inherited from dependent libraries on glibc, which would cause
# libsleefquad's NEEDED libtlfloat to fail to resolve in the executable.
set(CMAKE_INSTALL_RPATH "${SLEEF_ROOT}/lib")
set(CMAKE_BUILD_RPATH   "${SLEEF_ROOT}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_link_options(-Wl,--disable-new-dtags)
endif()

# ---------------------------------------------------------------------------
# Dependencies: OpenMP
# ---------------------------------------------------------------------------
if(QBLAS_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        message(STATUS "OpenMP: enabled (${OpenMP_C_VERSION})")
    else()
        message(WARNING "OpenMP not found - building single-threaded")
        set(QBLAS_USE_OPENMP OFF)
    endif()
endif()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_subdirectory(src)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(QBLAS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(QBLAS_BUILD_BENCH)
    add_subdirectory(bench)
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== QBLAS configuration ===")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Arch / ISAs     : ${QBLAS_ARCH} / ${QBLAS_ISA_TIERS}")
message(STATUS "  OpenMP          : ${QBLAS_USE_OPENMP}")
message(STATUS "  Tests           : ${QBLAS_BUILD_TESTS}")
message(STATUS "  Benchmarks      : ${QBLAS_BUILD_BENCH}")
message(STATUS "  SLEEF root      : ${SLEEF_ROOT}")
message(STATUS "")
