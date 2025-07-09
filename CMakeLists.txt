cmake_minimum_required(VERSION 3.15)
project(QuadBLAS)

# Check for custom SLEEF installation via environment variable
set(SLEEF_CUSTOM_ROOT "$ENV{SLEEF_ROOT}")

if(SLEEF_CUSTOM_ROOT)
    # Use custom SLEEF installation
    message(STATUS "Using custom SLEEF installation: ${SLEEF_CUSTOM_ROOT}")
    set(SLEEF_INCLUDE_DIR "${SLEEF_CUSTOM_ROOT}/include")
    set(SLEEF_LIB_DIR "${SLEEF_CUSTOM_ROOT}/lib")
    
    # Verify paths exist
    if(NOT EXISTS ${SLEEF_INCLUDE_DIR})
        message(FATAL_ERROR "SLEEF include directory not found: ${SLEEF_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS ${SLEEF_LIB_DIR})
        message(FATAL_ERROR "SLEEF library directory not found: ${SLEEF_LIB_DIR}")
    endif()
    
else()
    # Use system SLEEF via pkg-config
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SLEEF REQUIRED sleef)
endif()

# Find OpenMP
find_package(OpenMP)

# Create executables
add_executable(quadblas_test test_quadblas.cpp)
add_executable(quadblas_benchmark benchmark.cpp)

# Set C++ standard
target_compile_features(quadblas_test PRIVATE cxx_std_17)
target_compile_features(quadblas_benchmark PRIVATE cxx_std_17)

# Add compiler flags for optimization
target_compile_options(quadblas_test PRIVATE 
    -O3 -march=native -ffast-math
    $<$<CONFIG:DEBUG>:-g -fsanitize=address>
)
target_compile_options(quadblas_benchmark PRIVATE 
    -O3 -march=native -ffast-math
    $<$<CONFIG:DEBUG>:-g -fsanitize=address>
)

# Include directories
if(SLEEF_CUSTOM_ROOT)
    target_include_directories(quadblas_test PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}  # For QuadBLAS headers
        ${SLEEF_INCLUDE_DIR}
    )
    target_include_directories(quadblas_benchmark PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}  # For QuadBLAS headers
        ${SLEEF_INCLUDE_DIR}
    )
    
    # Link directories and libraries for custom SLEEF
    target_link_directories(quadblas_test PRIVATE ${SLEEF_LIB_DIR})
    target_link_directories(quadblas_benchmark PRIVATE ${SLEEF_LIB_DIR})
    
    target_link_libraries(quadblas_test 
        sleef sleefquad
        $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
        $<$<CONFIG:DEBUG>:-fsanitize=address>
    )
    target_link_libraries(quadblas_benchmark 
        sleef sleefquad
        $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
        $<$<CONFIG:DEBUG>:-fsanitize=address>
    )
else()
    # Use pkg-config provided paths and libraries
    target_include_directories(quadblas_test PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}  # For QuadBLAS headers
        ${SLEEF_INCLUDE_DIRS}
    )
    target_include_directories(quadblas_benchmark PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}  # For QuadBLAS headers
        ${SLEEF_INCLUDE_DIRS}
    )
    
    target_link_directories(quadblas_test PRIVATE ${SLEEF_LIBRARY_DIRS})
    target_link_directories(quadblas_benchmark PRIVATE ${SLEEF_LIBRARY_DIRS})
    
    target_compile_definitions(quadblas_test PRIVATE ${SLEEF_CFLAGS_OTHER})
    target_compile_definitions(quadblas_benchmark PRIVATE ${SLEEF_CFLAGS_OTHER})
    
    target_link_libraries(quadblas_test 
        ${SLEEF_LIBRARIES}
        $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
        $<$<CONFIG:DEBUG>:-fsanitize=address>
    )
    target_link_libraries(quadblas_benchmark 
        ${SLEEF_LIBRARIES}
        $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
        $<$<CONFIG:DEBUG>:-fsanitize=address>
    )
endif()

# Print configuration info
message(STATUS "SLEEF configuration:")
if(SLEEF_CUSTOM_ROOT)
    message(STATUS "  Custom installation: ${SLEEF_CUSTOM_ROOT}")
else()
    message(STATUS "  System installation via pkg-config")
    message(STATUS "  Include dirs: ${SLEEF_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${SLEEF_LIBRARIES}")
endif()
message(STATUS "OpenMP found: ${OpenMP_FOUND}")