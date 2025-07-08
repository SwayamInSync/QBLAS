#ifndef QUADBLAS_HPP
#define QUADBLAS_HPP

#include "quadblas_core.hpp"
#include "quadblas_operations.hpp"

// C Interface for easy integration with Python/numpy
extern "C" {

// ================================================================================
// C API - LEVEL 1 BLAS (Vector operations)
// ================================================================================

// QDOT: dot product of two vectors
double quadblas_qdot(int n, void* x, int incx, void* y, int incy) {
    Sleef_quad* qx = static_cast<Sleef_quad*>(x);
    Sleef_quad* qy = static_cast<Sleef_quad*>(y);

    Sleef_quad result = QuadBLAS::dot(static_cast<size_t>(n), qx,
                                     static_cast<size_t>(incx), qy,
                                     static_cast<size_t>(incy));
    
    return static_cast<double>(Sleef_cast_to_doubleq1(result));
}

// QNRM2: Euclidean norm of a vector
double quadblas_qnrm2(int n, void* x, int incx) {
    Sleef_quad* qx = static_cast<Sleef_quad*>(x);
    
    Sleef_quad result = QuadBLAS::dot(static_cast<size_t>(n), qx, 
                                     static_cast<size_t>(incx), qx, 
                                     static_cast<size_t>(incx));
    
    result = Sleef_sqrtq1_u05(result);
    return static_cast<double>(Sleef_cast_to_doubleq1(result));
}

// QAXPY: y := alpha*x + y
void quadblas_qaxpy(int n, double alpha, void* x, int incx, void* y, int incy) {
    Sleef_quad* qx = static_cast<Sleef_quad*>(x);
    Sleef_quad* qy = static_cast<Sleef_quad*>(y);
    Sleef_quad qalpha = Sleef_cast_from_doubleq1(alpha);
    
    if (n <= 0) return;
    
    size_t un = static_cast<size_t>(n);
    size_t uincx = static_cast<size_t>(incx);
    size_t uincy = static_cast<size_t>(incy);
    
#ifdef _OPENMP
    #pragma omp parallel for if(un >= QuadBLAS::PARALLEL_THRESHOLD)
#endif
    for (size_t i = 0; i < un; ++i) {
        qy[i * uincy] = Sleef_fmaq1_u05(qalpha, qx[i * uincx], qy[i * uincy]);
    }
}

// ================================================================================
// C API - LEVEL 2 BLAS (Matrix-vector operations)
// ================================================================================

// QGEMV: matrix-vector multiplication
void quadblas_qgemv(char layout, char trans, int m, int n, double alpha,
                   void* A, int lda, void* x, int incx,
                   double beta, void* y, int incy) {

    Sleef_quad* qA = static_cast<Sleef_quad*>(A);
    Sleef_quad* qx = static_cast<Sleef_quad*>(x);
    Sleef_quad* qy = static_cast<Sleef_quad*>(y);
    
    Sleef_quad qalpha = Sleef_cast_from_doubleq1(alpha);
    Sleef_quad qbeta = Sleef_cast_from_doubleq1(beta);
    
    QuadBLAS::Layout quadblas_layout = (layout == 'C' || layout == 'c') ? 
                                      QuadBLAS::Layout::ColMajor : 
                                      QuadBLAS::Layout::RowMajor;
    
    // Handle transpose (swap dimensions)
    if (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c') {
        std::swap(m, n);
        if (quadblas_layout == QuadBLAS::Layout::RowMajor) {
            quadblas_layout = QuadBLAS::Layout::ColMajor;
        } else {
            quadblas_layout = QuadBLAS::Layout::RowMajor;
        }
    }
    
    QuadBLAS::gemv(quadblas_layout, static_cast<size_t>(m), static_cast<size_t>(n),
                  qalpha, qA, static_cast<size_t>(lda), qx, static_cast<size_t>(incx),
                  qbeta, qy, static_cast<size_t>(incy));
}

// ================================================================================
// C API - LEVEL 3 BLAS (Matrix-matrix operations)
// ================================================================================

// QGEMM: matrix-matrix multiplication
void quadblas_qgemm(char layout, char transa, char transb, int m, int n, int k,
                   double alpha, void* A, int lda, void* B, int ldb,
                   double beta, void* C, int ldc) {
    
    Sleef_quad* qA = static_cast<Sleef_quad*>(A);
    Sleef_quad* qB = static_cast<Sleef_quad*>(B);
    Sleef_quad* qC = static_cast<Sleef_quad*>(C);
    
    Sleef_quad qalpha = Sleef_cast_from_doubleq1(alpha);
    Sleef_quad qbeta = Sleef_cast_from_doubleq1(beta);
    
    QuadBLAS::Layout quadblas_layout = (layout == 'C' || layout == 'c') ? 
                                      QuadBLAS::Layout::ColMajor : 
                                      QuadBLAS::Layout::RowMajor;
    
    // For simplicity, we assume no transpose for now
    // Full transpose support would require additional matrix manipulation
    if (transa != 'N' && transa != 'n') {
        // TODO: Handle transpose A
    }
    if (transb != 'N' && transb != 'n') {
        // TODO: Handle transpose B  
    }
    
    QuadBLAS::gemm(quadblas_layout, static_cast<size_t>(m), static_cast<size_t>(n), 
                  static_cast<size_t>(k), qalpha, qA, static_cast<size_t>(lda),
                  qB, static_cast<size_t>(ldb), qbeta, qC, static_cast<size_t>(ldc));
}

// ================================================================================
// Utility functions
// ================================================================================

// Set number of threads for OpenMP
void quadblas_set_num_threads(int num_threads) {
    QuadBLAS::set_num_threads(num_threads);
}

// Get number of threads
int quadblas_get_num_threads(void) {
    return QuadBLAS::get_num_threads();
}

// Version information
const char* quadblas_get_version(void) {
    return "QuadBLAS 1.0.0 - High Performance Quad Precision BLAS";
}

// Memory alignment check
int quadblas_is_aligned(void* ptr) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    return (addr % QuadBLAS::ALIGNMENT) == 0;
}

} // extern "C"

// ================================================================================
// C++ Convenience Classes
// ================================================================================

namespace QuadBLAS {

// Simple vector class for convenience
template<Layout layout>
class Vector {
private:
    Sleef_quad* data_;
    size_t size_;
    size_t stride_;
    bool owns_memory_;

public:
    Vector(size_t size) : size_(size), stride_(1), owns_memory_(true) {
        data_ = aligned_alloc<Sleef_quad>(size);
        std::fill(data_, data_ + size, SLEEF_QUAD_C(0.0));
    }
    
    Vector(Sleef_quad* data, size_t size, size_t stride = 1) 
        : data_(data), size_(size), stride_(stride), owns_memory_(false) {}
    
    ~Vector() {
        if (owns_memory_) {
            aligned_free(data_);
        }
    }
    
    // Move constructor
    Vector(Vector&& other) noexcept 
        : data_(other.data_), size_(other.size_), stride_(other.stride_), 
          owns_memory_(other.owns_memory_) {
        other.owns_memory_ = false;
    }
    
    // Disable copy constructor to avoid double-free
    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;
    
    Sleef_quad& operator[](size_t i) { return data_[i * stride_]; }
    Sleef_quad& operator[](size_t i) const { return data_[i * stride_]; }
    
    size_t size() const { return size_; }
    size_t stride() const { return stride_; }
    Sleef_quad* data() { return data_; }
    Sleef_quad* data() const { return data_; }
    
    // Dot product
    Sleef_quad dot(const Vector& other) const {
        return QuadBLAS::dot(size_, data_, stride_, other.data_, other.stride_);
    }
    
    // AXPY: this = alpha * other + this
    void axpy(Sleef_quad alpha, const Vector& other) {
        for (size_t i = 0; i < size_; ++i) {
            (*this)[i] = Sleef_fmaq1_u05(alpha, other[i], (*this)[i]);
        }
    }
    
    // Norm
    Sleef_quad norm() const {
        return Sleef_sqrtq1_u05(dot(*this));
    }
};

// Simple matrix class for convenience
template<Layout layout>
class Matrix {
private:
    Sleef_quad* data_;
    size_t rows_, cols_;
    size_t ld_;
    bool owns_memory_;

public:
    Matrix(size_t rows, size_t cols) 
        : rows_(rows), cols_(cols), ld_(cols), owns_memory_(true) {
        data_ = aligned_alloc<Sleef_quad>(rows * cols);
        std::fill(data_, data_ + rows * cols, SLEEF_QUAD_C(0.0));
    }
    
    Matrix(Sleef_quad* data, size_t rows, size_t cols, size_t ld = 0)
        : data_(data), rows_(rows), cols_(cols), 
          ld_(ld == 0 ? cols : ld), owns_memory_(false) {}
    
    ~Matrix() {
        if (owns_memory_) {
            aligned_free(data_);
        }
    }
    
    // Move constructor
    Matrix(Matrix&& other) noexcept
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_),
          ld_(other.ld_), owns_memory_(other.owns_memory_) {
        other.owns_memory_ = false;
    }
    
    // Disable copy constructor
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    
    Sleef_quad& operator()(size_t i, size_t j) {
        return layout == Layout::RowMajor ? data_[i * ld_ + j] : data_[j * ld_ + i];
    }
    
    Sleef_quad& operator()(size_t i, size_t j) const {
        return layout == Layout::RowMajor ? data_[i * ld_ + j] : data_[j * ld_ + i];
    }
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t leading_dimension() const { return ld_; }
    Sleef_quad* data() { return data_; }
    Sleef_quad* data() const { return data_; }
    
    // Matrix-vector multiplication
    void gemv(Sleef_quad alpha, const Vector<layout>& x, Sleef_quad beta, Vector<layout>& y) const {
        QuadBLAS::gemv(layout, rows_, cols_, alpha, data_, ld_, 
                      x.data(), x.stride(), beta, y.data(), y.stride());
    }
    
    // Matrix-matrix multiplication
    void gemm(Sleef_quad alpha, Matrix& B, Sleef_quad beta, Matrix& C) const {
        QuadBLAS::gemm(layout, rows_, B.cols(), cols_, alpha, data_, ld_,
                      B.data(), B.leading_dimension(), beta, C.data(), C.leading_dimension());
    }
};

// Convenient type aliases with default template arguments
using VectorRowMajor = Vector<Layout::RowMajor>;
using VectorColMajor = Vector<Layout::ColMajor>;
using MatrixRowMajor = Matrix<Layout::RowMajor>;
using MatrixColMajor = Matrix<Layout::ColMajor>;

// Default to row major
template<Layout layout = Layout::RowMajor>
using DefaultVector = Vector<layout>;

template<Layout layout = Layout::RowMajor>  
using DefaultMatrix = Matrix<layout>;

} // namespace QuadBLAS

#endif // QUADBLAS_HPP