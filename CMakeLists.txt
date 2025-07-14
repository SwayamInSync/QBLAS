cmake_minimum_required(VERSION 3.15)
project(QuadBLAS)
set(SLEEF_CUSTOM_ROOT "$ENV{SLEEF_ROOT}")

if(SLEEF_CUSTOM_ROOT)
    message(STATUS "Using custom SLEEF installation: ${SLEEF_CUSTOM_ROOT}")
    set(SLEEF_INCLUDE_DIR "${SLEEF_CUSTOM_ROOT}/include")
    set(SLEEF_LIB_DIR "${SLEEF_CUSTOM_ROOT}/lib")
    if(NOT EXISTS ${SLEEF_INCLUDE_DIR})
        message(FATAL_ERROR "SLEEF include directory not found: ${SLEEF_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS ${SLEEF_LIB_DIR})
        message(FATAL_ERROR "SLEEF library directory not found: ${SLEEF_LIB_DIR}")
    endif()
    
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SLEEF REQUIRED sleef)
endif()
set(OpenMP_FOUND FALSE)

if(APPLE)
    message(STATUS "Configuring OpenMP for macOS with Apple Clang...")
    set(HOMEBREW_PREFIXES "/opt/homebrew" "/usr/local")
    
    foreach(PREFIX ${HOMEBREW_PREFIXES})
        set(FOUND_LIBOMP FALSE)
        set(LIBOMP_LIB_PATH "")
        set(LIBOMP_INCLUDE_PATH "")
        if(EXISTS "${PREFIX}/lib/libomp.dylib" AND EXISTS "${PREFIX}/include/omp.h")
            set(FOUND_LIBOMP TRUE)
            set(LIBOMP_LIB_PATH "${PREFIX}/lib/libomp.dylib")
            set(LIBOMP_INCLUDE_PATH "${PREFIX}/include")
            message(STATUS "Found libomp at: ${PREFIX} (linked)")
        elseif(EXISTS "${PREFIX}/opt/libomp/lib/libomp.dylib" AND EXISTS "${PREFIX}/opt/libomp/include/omp.h")
            set(FOUND_LIBOMP TRUE)
            set(LIBOMP_LIB_PATH "${PREFIX}/opt/libomp/lib/libomp.dylib")
            set(LIBOMP_INCLUDE_PATH "${PREFIX}/opt/libomp/include")
            message(STATUS "Found libomp at: ${PREFIX}/opt/libomp (keg-only)")
        else()
            file(GLOB LIBOMP_CELLAR_DIRS "${PREFIX}/Cellar/libomp/*/")
            foreach(CELLAR_DIR ${LIBOMP_CELLAR_DIRS})
                if(EXISTS "${CELLAR_DIR}/lib/libomp.dylib" AND EXISTS "${CELLAR_DIR}/include/omp.h")
                    set(FOUND_LIBOMP TRUE)
                    set(LIBOMP_LIB_PATH "${CELLAR_DIR}/lib/libomp.dylib")
                    set(LIBOMP_INCLUDE_PATH "${CELLAR_DIR}/include")
                    message(STATUS "Found libomp at: ${CELLAR_DIR} (cellar)")
                    break()
                endif()
            endforeach()
        endif()
        
        if(FOUND_LIBOMP)
            set(OpenMP_FOUND TRUE)
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${LIBOMP_LIB_PATH}")
            if(NOT TARGET OpenMP::OpenMP_CXX)
                add_library(OpenMP::OpenMP_CXX UNKNOWN IMPORTED)
                set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
                    IMPORTED_LOCATION "${LIBOMP_LIB_PATH}"
                    INTERFACE_INCLUDE_DIRECTORIES "${LIBOMP_INCLUDE_PATH}"
                    INTERFACE_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp"
                    INTERFACE_LINK_LIBRARIES "${LIBOMP_LIB_PATH}"
                )
            endif()
            
            if(NOT TARGET OpenMP::OpenMP_C)
                add_library(OpenMP::OpenMP_C UNKNOWN IMPORTED)
                set_target_properties(OpenMP::OpenMP_C PROPERTIES
                    IMPORTED_LOCATION "${LIBOMP_LIB_PATH}"
                    INTERFACE_INCLUDE_DIRECTORIES "${LIBOMP_INCLUDE_PATH}"
                    INTERFACE_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp"
                    INTERFACE_LINK_LIBRARIES "${LIBOMP_LIB_PATH}"
                )
            endif()
            break()
        endif()
    endforeach()
    
    if(NOT OpenMP_FOUND)
        message(WARNING "OpenMP (libomp) not found. Install with: brew install libomp")
        message(WARNING "Building without OpenMP support.")
    endif()
else()
    find_package(OpenMP)
endif()
add_executable(quadblas_test test_quadblas.cpp)
add_executable(quadblas_benchmark benchmark.cpp)
target_compile_features(quadblas_test PRIVATE cxx_std_17)
target_compile_features(quadblas_benchmark PRIVATE cxx_std_17)
target_compile_options(quadblas_test PRIVATE 
    -O3 -march=native -ffast-math
    $<$<CONFIG:DEBUG>:-g -fsanitize=address>
)
target_compile_options(quadblas_benchmark PRIVATE 
    -O3 -march=native -ffast-math
    $<$<CONFIG:DEBUG>:-g -fsanitize=address>
)
if(SLEEF_CUSTOM_ROOT)
    target_include_directories(quadblas_test PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}  # For QuadBLAS headers
        ${SLEEF_INCLUDE_DIR}
    )
    target_include_directories(quadblas_benchmark PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}  # For QuadBLAS headers
        ${SLEEF_INCLUDE_DIR}
    )
    target_link_directories(quadblas_test PRIVATE ${SLEEF_LIB_DIR})
    target_link_directories(quadblas_benchmark PRIVATE ${SLEEF_LIB_DIR})
    
    target_link_libraries(quadblas_test 
        sleef sleefquad
        $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>
        $<$<CONFIG:DEBUG>:-fsanitize=address>
    )
    target_link_libraries(quadblas_benchmark 
        sleef sleefquad
        $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>
        $<$<CONFIG:DEBUG>:-fsanitize=address>
    )
else()
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
        $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>
        $<$<CONFIG:DEBUG>:-fsanitize=address>
    )
    target_link_libraries(quadblas_benchmark 
        ${SLEEF_LIBRARIES}
        $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>
        $<$<CONFIG:DEBUG>:-fsanitize=address>
    )
endif()
message(STATUS "SLEEF configuration:")
if(SLEEF_CUSTOM_ROOT)
    message(STATUS "  Custom installation: ${SLEEF_CUSTOM_ROOT}")
else()
    message(STATUS "  System installation via pkg-config")
    message(STATUS "  Include dirs: ${SLEEF_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${SLEEF_LIBRARIES}")
endif()
message(STATUS "OpenMP found: ${OpenMP_FOUND}")
if(OpenMP_FOUND)
    message(STATUS "  OpenMP C flags: ${OpenMP_C_FLAGS}")
    message(STATUS "  OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
    message(STATUS "  OpenMP library: ${OpenMP_omp_LIBRARY}")
else()
    message(STATUS "  Building without OpenMP support")
endif()