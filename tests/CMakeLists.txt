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
