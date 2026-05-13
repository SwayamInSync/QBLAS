add_executable(qblas_test test_qblas.c)
target_link_libraries(qblas_test PRIVATE qblas)
target_include_directories(qblas_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME qblas_test COMMAND qblas_test)
