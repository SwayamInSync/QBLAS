# Each test target is a C executable linking the qblas shared lib.

set(QBLAS_TESTS
    test_smoke
    test_level1
    test_level2
    test_gemm
    test_level3_extra)

foreach(t ${QBLAS_TESTS})
    add_executable(${t} ${t}.c)
    target_link_libraries(${t} PRIVATE qblas m)
    target_include_directories(${t} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# -----------------------------------------------------------------------
# Python numpy-comparison test
# -----------------------------------------------------------------------
# Builds a small shim library (`libqblas_shim.so`) that re-exposes the
# qblas entry points with `Sleef_quad *` instead of `Sleef_quad` so that
# Python's ctypes / libffi handles the calling convention reliably.
# The Python harness then calls every routine and compares the result
# (rounded back to double) against numpy's double-precision reference.
add_library(qblas_shim SHARED python/qblas_shim.c)
target_link_libraries(qblas_shim PRIVATE qblas)
target_include_directories(qblas_shim PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(qblas_shim PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Look for a Python 3 interpreter — required only for the numpy test.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME test_against_numpy
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/python/test_against_numpy.py)
    set_tests_properties(test_against_numpy PROPERTIES
        DEPENDS qblas_shim
        ENVIRONMENT "QBLAS_TEST_SHIM=$<TARGET_FILE:qblas_shim>")
endif()
