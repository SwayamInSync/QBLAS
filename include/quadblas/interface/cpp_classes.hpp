#ifndef QUADBLAS_INTERFACE_CPP_CLASSES_HPP
#define QUADBLAS_INTERFACE_CPP_CLASSES_HPP

#include "../core/types.hpp"
#include "../core/platform.hpp"
#include "../memory/allocation.hpp"
#include "../algorithms/level1.hpp"
#include "../algorithms/level2.hpp"
#include "../algorithms/level3.hpp"
#include <algorithm>

namespace QuadBLAS
{

  // C++ Convenience Classes

  // Simple vector class for convenience
  template <Layout layout>
  class Vector
  {
  private:
    Sleef_quad *data_;
    size_t size_;
    size_t stride_;
    bool owns_memory_;

  public:
    Vector(size_t size) : size_(size), stride_(1), owns_memory_(true)
    {
      data_ = aligned_alloc<Sleef_quad>(size);
      std::fill(data_, data_ + size, SLEEF_QUAD_C(0.0));
    }

    Vector(Sleef_quad *data, size_t size, size_t stride = 1)
        : data_(data), size_(size), stride_(stride), owns_memory_(false) {}

    ~Vector()
    {
      if (owns_memory_)
      {
        aligned_free(data_);
      }
    }

    // Move constructor
    Vector(Vector &&other) noexcept
        : data_(other.data_), size_(other.size_), stride_(other.stride_),
          owns_memory_(other.owns_memory_)
    {
      other.owns_memory_ = false;
    }

    // Disable copy constructor to avoid double-free
    Vector(const Vector &) = delete;
    Vector &operator=(const Vector &) = delete;

    Sleef_quad &operator[](size_t i) { return data_[i * stride_]; }
    const Sleef_quad &operator[](size_t i) const { return data_[i * stride_]; }

    size_t size() const { return size_; }
    size_t stride() const { return stride_; }
    Sleef_quad *data() { return data_; }
    const Sleef_quad *data() const { return data_; }

    // Dot product
    Sleef_quad dot(const Vector &other) const
    {
      return QuadBLAS::dot(size_, data_, stride_, other.data_, other.stride_);
    }

    // AXPY: this = alpha * other + this
    void axpy(Sleef_quad alpha, const Vector &other)
    {
      QuadBLAS::axpy(size_, alpha, other.data_, other.stride_, data_, stride_);
    }

    // Norm
    Sleef_quad norm() const
    {
      return Sleef_sqrtq1_u05(dot(*this));
    }
  };

  // Simple matrix class for convenience
  template <Layout layout>
  class Matrix
  {
  private:
    Sleef_quad *data_;
    size_t rows_, cols_;
    size_t ld_;
    bool owns_memory_;

  public:
    Matrix(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), ld_(cols), owns_memory_(true)
    {
      data_ = aligned_alloc<Sleef_quad>(rows * cols);
      std::fill(data_, data_ + rows * cols, SLEEF_QUAD_C(0.0));
    }

    Matrix(Sleef_quad *data, size_t rows, size_t cols, size_t ld = 0)
        : data_(data), rows_(rows), cols_(cols),
          ld_(ld == 0 ? cols : ld), owns_memory_(false) {}

    ~Matrix()
    {
      if (owns_memory_)
      {
        aligned_free(data_);
      }
    }

    // Move constructor
    Matrix(Matrix &&other) noexcept
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_),
          ld_(other.ld_), owns_memory_(other.owns_memory_)
    {
      other.owns_memory_ = false;
    }

    // Disable copy constructor
    Matrix(const Matrix &) = delete;
    Matrix &operator=(const Matrix &) = delete;

    Sleef_quad &operator()(size_t i, size_t j)
    {
      return layout == Layout::RowMajor ? data_[i * ld_ + j] : data_[j * ld_ + i];
    }

    const Sleef_quad &operator()(size_t i, size_t j) const
    {
      return layout == Layout::RowMajor ? data_[i * ld_ + j] : data_[j * ld_ + i];
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t leading_dimension() const { return ld_; }
    Sleef_quad *data() { return data_; }
    const Sleef_quad *data() const { return data_; }

    // Matrix-vector multiplication
    void gemv(Sleef_quad alpha, const Vector<layout> &x, Sleef_quad beta, Vector<layout> &y) const
    {
      QuadBLAS::gemv(layout, rows_, cols_, alpha, data_, ld_,
                     x.data(), x.stride(), beta, y.data(), y.stride());
    }

    // Matrix-matrix multiplication
    void gemm(Sleef_quad alpha, const Matrix &B, Sleef_quad beta, Matrix &C) const
    {
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
  template <Layout layout = Layout::RowMajor>
  using DefaultVector = Vector<layout>;

  template <Layout layout = Layout::RowMajor>
  using DefaultMatrix = Matrix<layout>;

} // namespace QuadBLAS

#endif // QUADBLAS_INTERFACE_CPP_CLASSES_HPP