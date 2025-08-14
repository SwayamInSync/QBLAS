cmake_minimum_required(VERSION 3.15)
project(QuadBLAS)

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Set C++ standard globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP (your existing code)
if(APPLE)
    find_library(OpenMP_LIBRARY
        NAMES omp gomp iomp5
        PATHS /opt/homebrew/lib /usr/local/lib /opt/homebrew/opt/libomp/lib
        NO_DEFAULT_PATH
    )
    find_path(OpenMP_INCLUDE_DIR
        NAMES omp.h
        PATHS /opt/homebrew/include /usr/local/include /opt/homebrew/opt/libomp/include
        NO_DEFAULT_PATH
    )
    
    if(OpenMP_LIBRARY AND OpenMP_INCLUDE_DIR)
        set(OpenMP_FOUND TRUE)
        add_library(OpenMP::OpenMP_CXX UNKNOWN IMPORTED)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            IMPORTED_LOCATION "${OpenMP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OpenMP_INCLUDE_DIR}"
            INTERFACE_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp"
        )
        message(STATUS "Found OpenMP: ${OpenMP_LIBRARY}")
    else()
        set(OpenMP_FOUND FALSE)
        message(WARNING "OpenMP not found. Install with: brew install libomp")
    endif()
else()
    find_package(OpenMP)
endif()

# Handle SLEEF library (fixed for your environment)
set(SLEEF_CUSTOM_ROOT "$ENV{SLEEF_ROOT}")
if(SLEEF_CUSTOM_ROOT)
    set(SLEEF_INCLUDE_DIR "${SLEEF_CUSTOM_ROOT}/include")
    set(SLEEF_LIB_DIR "${SLEEF_CUSTOM_ROOT}/lib")
    if(NOT EXISTS ${SLEEF_INCLUDE_DIR} OR NOT EXISTS ${SLEEF_LIB_DIR})
        message(FATAL_ERROR "SLEEF installation not found at: ${SLEEF_CUSTOM_ROOT}")
    endif()
    # Use dynamic libraries (.dylib) instead of static (.a)
    set(SLEEF_LIBRARIES "${SLEEF_LIB_DIR}/libsleef.dylib" "${SLEEF_LIB_DIR}/libsleefquad.dylib")
    message(STATUS "Using SLEEF from: ${SLEEF_CUSTOM_ROOT}")
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SLEEF REQUIRED sleef)
endif()

# Add Google Benchmark with proper Release configuration
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/benchmark/CMakeLists.txt")
    # Configure Google Benchmark for Release mode and disable testing
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Enable testing of the benchmark library.")
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Enable Google Test in benchmark.")
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "Allow benchmark to download dependencies.")
    
    # Ensure benchmark is built in Release mode
    set(CMAKE_BUILD_TYPE_BACKUP ${CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE Release)
    
    add_subdirectory(third_party/benchmark)
    
    # Restore original build type
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_BACKUP})
    
    set(GOOGLE_BENCHMARK_AVAILABLE TRUE)
    message(STATUS "Google Benchmark added (Release mode)")
else()
    set(GOOGLE_BENCHMARK_AVAILABLE FALSE)
    message(STATUS "Google Benchmark not found (run: git submodule add https://github.com/google/benchmark.git third_party/benchmark)")
endif()

# Create executables
add_executable(quadblas_test test_quadblas.cpp)
add_executable(quadblas_benchmark_legacy benchmarks/legacy_benchmark.cpp)

# Add Google Benchmark executable if available
if(GOOGLE_BENCHMARK_AVAILABLE)
    add_executable(quadblas_benchmark benchmarks/benchmark.cpp)
endif()

# Common configuration function
function(configure_target target_name)
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    
    # Set optimization flags based on build type
    target_compile_options(${target_name} PRIVATE 
        $<$<CONFIG:Release>:-O3 -march=native -ffast-math -DNDEBUG>
        $<$<CONFIG:Debug>:-g -O0 -fsanitize=address>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g -march=native -ffast-math>
    )
    
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    
    if(SLEEF_CUSTOM_ROOT)
        target_include_directories(${target_name} PRIVATE ${SLEEF_INCLUDE_DIR})
    else()
        target_include_directories(${target_name} PRIVATE ${SLEEF_INCLUDE_DIRS})
        target_link_directories(${target_name} PRIVATE ${SLEEF_LIBRARY_DIRS})
        target_compile_definitions(${target_name} PRIVATE ${SLEEF_CFLAGS_OTHER})
    endif()
    
    target_link_libraries(${target_name} 
        ${SLEEF_LIBRARIES}
        $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>
        $<$<CONFIG:Debug>:-fsanitize=address>
    )
endfunction()

# Configure targets
configure_target(quadblas_test)
configure_target(quadblas_benchmark_legacy)

# Configure Google Benchmark target with proper linking
if(GOOGLE_BENCHMARK_AVAILABLE)
    configure_target(quadblas_benchmark)
    target_link_libraries(quadblas_benchmark benchmark::benchmark benchmark::benchmark_main)
endif()

message(STATUS "=== Configuration ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenMP: ${OpenMP_FOUND}")
message(STATUS "SLEEF: ${SLEEF_LIBRARIES}")
message(STATUS "Google Benchmark: ${GOOGLE_BENCHMARK_AVAILABLE}")