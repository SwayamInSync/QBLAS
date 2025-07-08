cmake_minimum_required(VERSION 3.15)
project(QuadBLAS_Example)

# Set SLEEF paths manually
set(SLEEF_ROOT "/datadrive/guestuser/t-swsingh/temp/sleef/build")
set(SLEEF_INCLUDE_DIR "${SLEEF_ROOT}/include")
set(SLEEF_LIB_DIR "${SLEEF_ROOT}/lib")

# Check if SLEEF paths exist
if(NOT EXISTS ${SLEEF_INCLUDE_DIR})
    message(FATAL_ERROR "SLEEF include directory not found: ${SLEEF_INCLUDE_DIR}")
endif()

if(NOT EXISTS ${SLEEF_LIB_DIR})
    message(FATAL_ERROR "SLEEF library directory not found: ${SLEEF_LIB_DIR}")
endif()

# Find OpenMP
find_package(OpenMP)

# Create executable
add_executable(quadblas_test test_quadblas.cpp)

# Set C++ standard
target_compile_features(quadblas_test PRIVATE cxx_std_17)

# Add compiler flags for optimization
target_compile_options(quadblas_test PRIVATE 
    -O3 -march=native -ffast-math
    $<$<CONFIG:DEBUG>:-g -fsanitize=address>
)

# Include directories
target_include_directories(quadblas_test PRIVATE ${SLEEF_INCLUDE_DIR})

# Link directories
target_link_directories(quadblas_test PRIVATE ${SLEEF_LIB_DIR})

# Link libraries
target_link_libraries(quadblas_test 
    sleef
    sleefquad
    tlfloat
    $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
    $<$<CONFIG:DEBUG>:-fsanitize=address>
)

# Print configuration info
message(STATUS "SLEEF include directory: ${SLEEF_INCLUDE_DIR}")
message(STATUS "SLEEF library directory: ${SLEEF_LIB_DIR}")
message(STATUS "OpenMP found: ${OpenMP_FOUND}")