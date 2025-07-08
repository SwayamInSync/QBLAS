#ifndef QUADBLAS_OPERATIONS_HPP
#define QUADBLAS_OPERATIONS_HPP

#include "quadblas_core.hpp"

namespace QuadBLAS {

// ================================================================================
// VECTOR-VECTOR DOT PRODUCT (LEVEL 1 BLAS)
// ================================================================================

// Vectorized dot product kernel for aligned data
inline Sleef_quad dot_kernel_vectorized(Sleef_quad* x, Sleef_quad* y, size_t n) {
    const size_t vec_n = n / VECTOR_SIZE;
    const size_t remainder = n % VECTOR_SIZE;
    
    QuadVector sum_vec(SLEEF_QUAD_C(0.0));
    
    // Vectorized loop
    for (size_t i = 0; i < vec_n; ++i) {
        QuadVector x_vec = QuadVector::load(&x[i * VECTOR_SIZE]);
        QuadVector y_vec = QuadVector::load(&y[i * VECTOR_SIZE]);
        sum_vec = sum_vec.fma(x_vec, y_vec);
    }
    
    Sleef_quad result = sum_vec.horizontal_sum();
    
    // Handle remainder
    for (size_t i = vec_n * VECTOR_SIZE; i < n; ++i) {
        result = Sleef_fmaq1_u05(x[i], y[i], result);
    }
    
    return result;
}

// Parallel dot product for large vectors
inline Sleef_quad dot_parallel(Sleef_quad* x, Sleef_quad* y, size_t n) {
    if (n < PARALLEL_THRESHOLD) {
        return dot_kernel_vectorized(x, y, n);
    }
    
#ifdef _OPENMP
    const int num_threads = get_num_threads();
    const size_t chunk_size = n / num_threads;
    Sleef_quad result = SLEEF_QUAD_C(0.0);
    
    #pragma omp parallel reduction(+:result)
    {
        int tid = omp_get_thread_num();
        size_t start = tid * chunk_size;
        size_t end = (tid == num_threads - 1) ? n : start + chunk_size;
        
        if (start < end) {
            result += dot_kernel_vectorized(&x[start], &y[start], end - start);
        }
    }
    
    return result;
#else
    return dot_kernel_vectorized(x, y, n);
#endif
}

// Main dot product function with stride support
inline Sleef_quad dot(size_t n, Sleef_quad* x, size_t incx, 
                      Sleef_quad* y, size_t incy) {
    if (n == 0) return SLEEF_QUAD_C(0.0);
    
    // Fast path for unit strides
    if (incx == 1 && incy == 1) {
        return dot_parallel(x, y, n);
    }
    
    // Strided access
    Sleef_quad result = SLEEF_QUAD_C(0.0);
    
    if (n >= PARALLEL_THRESHOLD) {
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:result)
        for (size_t i = 0; i < n; ++i) {
            result = Sleef_fmaq1_u05(x[i * incx], y[i * incy], result);
        }
#else
        for (size_t i = 0; i < n; ++i) {
            result = Sleef_fmaq1_u05(x[i * incx], y[i * incy], result);
        }
#endif
    } else {
        for (size_t i = 0; i < n; ++i) {
            result = Sleef_fmaq1_u05(x[i * incx], y[i * incy], result);
        }
    }
    
    return result;
}

// ================================================================================
// MATRIX-VECTOR MULTIPLICATION (LEVEL 2 BLAS)
// ================================================================================

// GEMV: y = alpha * A * x + beta * y
// Row-major implementation
inline void gemv_row_major(size_t m, size_t n, Sleef_quad alpha, 
                          Sleef_quad* A, size_t lda,
                          Sleef_quad* x, size_t incx,
                          Sleef_quad beta, Sleef_quad* y, size_t incy) {
    
    if (m == 0 || n == 0) return;
    
    const bool use_parallel = m >= PARALLEL_THRESHOLD;
    
#ifdef _OPENMP
    #pragma omp parallel for if(use_parallel)
#endif
    for (size_t i = 0; i < m; ++i) {
        // Compute dot product of row i with vector x
        Sleef_quad sum = SLEEF_QUAD_C(0.0);
        
        Sleef_quad* row = &A[i * lda];
        
        // Vectorized inner loop
        if (incx == 1) {
            sum = dot_kernel_vectorized(row, x, n);
        } else {
            // Strided access
            for (size_t j = 0; j < n; ++j) {
                sum = Sleef_fmaq1_u05(row[j], x[j * incx], sum);
            }
        }
        
        // y[i] = alpha * sum + beta * y[i]
        size_t y_idx = i * incy;
        y[y_idx] = Sleef_fmaq1_u05(alpha, sum, Sleef_mulq1_u05(beta, y[y_idx]));
    }
}

// Column-major implementation
inline void gemv_col_major(size_t m, size_t n, Sleef_quad alpha,
                          const Sleef_quad* A, size_t lda,
                          const Sleef_quad* x, size_t incx,
                          Sleef_quad beta, Sleef_quad* y, size_t incy) {
    
    if (m == 0 || n == 0) return;
    
    // Scale y by beta first
    for (size_t i = 0; i < m; ++i) {
        y[i * incy] = Sleef_mulq1_u05(beta, y[i * incy]);
    }
    
    // Add alpha * A * x
    for (size_t j = 0; j < n; ++j) {
        Sleef_quad x_j = Sleef_mulq1_u05(alpha, x[j * incx]);
        const Sleef_quad* col = &A[j * lda];
        
#ifdef _OPENMP
        #pragma omp parallel for if(m >= PARALLEL_THRESHOLD)
#endif
        for (size_t i = 0; i < m; ++i) {
            y[i * incy] = Sleef_fmaq1_u05(col[i], x_j, y[i * incy]);
        }
    }
}

// Main GEMV function
inline void gemv(Layout layout, size_t m, size_t n, Sleef_quad alpha,
                Sleef_quad* A, size_t lda,
                Sleef_quad* x, size_t incx,
                Sleef_quad beta, Sleef_quad* y, size_t incy) {
    
    if (layout == Layout::RowMajor) {
        gemv_row_major(m, n, alpha, A, lda, x, incx, beta, y, incy);
    } else {
        gemv_col_major(m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
}

// ================================================================================
// MATRIX-MATRIX MULTIPLICATION (LEVEL 3 BLAS)
// ================================================================================

// Micro-kernel for small matrix blocks - highly optimized inner loop
inline void gemm_micro_kernel(size_t mr, size_t nr, size_t kc,
                             Sleef_quad alpha,
                             Sleef_quad* A, Sleef_quad* B,
                             Sleef_quad beta, Sleef_quad* C, size_t ldc) {
    
    // Accumulate in registers
    QuadVector c_vec[mr/VECTOR_SIZE][nr/VECTOR_SIZE];
    
    // Initialize accumulators
    for (size_t i = 0; i < mr/VECTOR_SIZE; ++i) {
        for (size_t j = 0; j < nr/VECTOR_SIZE; ++j) {
            c_vec[i][j] = QuadVector(SLEEF_QUAD_C(0.0));
        }
    }
    
    // Main computation loop
    for (size_t k = 0; k < kc; ++k) {
        for (size_t i = 0; i < mr/VECTOR_SIZE; ++i) {
            QuadVector a_vec = QuadVector::load(&A[i * VECTOR_SIZE * kc + k * VECTOR_SIZE]);
            
            for (size_t j = 0; j < nr/VECTOR_SIZE; ++j) {
                QuadVector b_vec = QuadVector::load(&B[k * nr + j * VECTOR_SIZE]);
                c_vec[i][j] = c_vec[i][j].fma(a_vec, b_vec);
            }
        }
    }
    
    // Store results back to C with alpha and beta scaling
    QuadVector alpha_vec(alpha);
    QuadVector beta_vec(beta);
    
    for (size_t i = 0; i < mr/VECTOR_SIZE; ++i) {
        for (size_t j = 0; j < nr/VECTOR_SIZE; ++j) {
            Sleef_quad* c_ptr = &C[i * VECTOR_SIZE * ldc + j * VECTOR_SIZE];
            QuadVector c_old = QuadVector::load(c_ptr);
            QuadVector c_new = (c_vec[i][j] * alpha_vec) + (c_old * beta_vec);
            c_new.store(c_ptr);
        }
    }
}

// Macro-kernel for medium-sized blocks
inline void gemm_macro_kernel(size_t mc, size_t nc, size_t kc,
                             Sleef_quad alpha,
                             Sleef_quad* A, Sleef_quad* B,
                             Sleef_quad beta, Sleef_quad* C, size_t ldc) {
    
    constexpr size_t MR = 4;  // Micro-panel height
    constexpr size_t NR = 4;  // Micro-panel width
    
    for (size_t i = 0; i < mc; i += MR) {
        size_t mr = std::min(MR, mc - i);
        
        for (size_t j = 0; j < nc; j += NR) {
            size_t nr = std::min(NR, nc - j);
            
            gemm_micro_kernel(mr, nr, kc, alpha,
                             &A[i * kc], &B[j],
                             beta, &C[i * ldc + j], ldc);
        }
    }
}

// Main GEMM function: C = alpha * A * B + beta * C
inline void gemm(Layout layout, size_t m, size_t n, size_t k,
                Sleef_quad alpha,
                const Sleef_quad* A, size_t lda,
                const Sleef_quad* B, size_t ldb,
                Sleef_quad beta, Sleef_quad* C, size_t ldc) {
    
    if (m == 0 || n == 0 || k == 0) return;
    
    BlockingParams params(m, n, k);
    
    // Allocate temporary packed matrices for better cache performance
    Sleef_quad* A_packed = aligned_alloc<Sleef_quad>(params.mc * params.kc);
    Sleef_quad* B_packed = aligned_alloc<Sleef_quad>(params.kc * params.nc);
    
    if (!A_packed || !B_packed) {
        // Fallback to simple implementation if allocation fails
        aligned_free(A_packed);
        aligned_free(B_packed);
        
        // Simple triple loop with OpenMP parallelization
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) if(m * n >= PARALLEL_THRESHOLD)
#endif
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                Sleef_quad sum = SLEEF_QUAD_C(0.0);
                
                for (size_t l = 0; l < k; ++l) {
                    size_t a_idx = (layout == Layout::RowMajor) ? i * lda + l : l * lda + i;
                    size_t b_idx = (layout == Layout::RowMajor) ? l * ldb + j : j * ldb + l;
                    sum = Sleef_fmaq1_u05(A[a_idx], B[b_idx], sum);
                }
                
                size_t c_idx = (layout == Layout::RowMajor) ? i * ldc + j : j * ldc + i;
                C[c_idx] = Sleef_fmaq1_u05(alpha, sum, Sleef_mulq1_u05(beta, C[c_idx]));
            }
        }
        return;
    }
    
    // Blocked GEMM implementation
    for (size_t kk = 0; kk < k; kk += params.kc) {
        size_t kc = std::min(params.kc, k - kk);
        
        for (size_t mm = 0; mm < m; mm += params.mc) {
            size_t mc = std::min(params.mc, m - mm);
            
            // Pack A panel
            for (size_t i = 0; i < mc; ++i) {
                for (size_t j = 0; j < kc; ++j) {
                    size_t src_idx = (layout == Layout::RowMajor) ? 
                                   (mm + i) * lda + (kk + j) : 
                                   (kk + j) * lda + (mm + i);
                    A_packed[i * kc + j] = A[src_idx];
                }
            }
            
            for (size_t nn = 0; nn < n; nn += params.nc) {
                size_t nc = std::min(params.nc, n - nn);
                
                // Pack B panel
                for (size_t i = 0; i < kc; ++i) {
                    for (size_t j = 0; j < nc; ++j) {
                        size_t src_idx = (layout == Layout::RowMajor) ? 
                                       (kk + i) * ldb + (nn + j) : 
                                       (nn + j) * ldb + (kk + i);
                        B_packed[i * nc + j] = B[src_idx];
                    }
                }
                
                // Compute C block
                Sleef_quad* C_block = &C[(layout == Layout::RowMajor) ? 
                                        mm * ldc + nn : nn * ldc + mm];
                
                gemm_macro_kernel(mc, nc, kc, alpha,
                                A_packed, B_packed,
                                (kk == 0) ? beta : SLEEF_QUAD_C(1.0),
                                C_block, ldc);
            }
        }
    }
    
    aligned_free(A_packed);
    aligned_free(B_packed);
}

} // namespace QuadBLAS

#endif // QUADBLAS_OPERATIONS_HPP