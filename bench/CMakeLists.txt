# Google Benchmark (vendored under third_party/benchmark).
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/benchmark/CMakeLists.txt")
    set(BENCHMARK_ENABLE_TESTING       OFF CACHE BOOL "")
    set(BENCHMARK_ENABLE_GTEST_TESTS   OFF CACHE BOOL "")
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "")
    set(BENCHMARK_INSTALL_DOCS         OFF CACHE BOOL "")
    if(NOT TARGET benchmark::benchmark)
        add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/benchmark
                         ${CMAKE_CURRENT_BINARY_DIR}/benchmark_build EXCLUDE_FROM_ALL)
    endif()
else()
    message(STATUS "Google Benchmark submodule missing - bench/ disabled.")
    return()
endif()

add_executable(qblas_bench bench_qblas.cc)
target_link_libraries(qblas_bench PRIVATE qblas benchmark::benchmark)
target_include_directories(qblas_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
