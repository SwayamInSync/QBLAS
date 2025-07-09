#ifndef QUADBLAS_SIMD_QUAD_VECTOR_HPP
#define QUADBLAS_SIMD_QUAD_VECTOR_HPP

#include "../core/platform.hpp"
#include "../memory/allocation.hpp"

namespace QuadBLAS
{

  // SIMD wrapper for platform abstraction
  class QuadVector
  {
  private:
#ifdef QUADBLAS_X86_64
    Sleef_quadx2 data_;
#elif defined(QUADBLAS_AARCH64)
    Sleef_quadx2 data_;
#else
    Sleef_quad data_[2];
#endif

  public:
    QuadVector() = default;

    explicit QuadVector(Sleef_quad value)
    {
#ifdef QUADBLAS_X86_64
      data_ = Sleef_splatq2_sse2(value);
#elif defined(QUADBLAS_AARCH64)
      data_ = Sleef_splatq2_advsimd(value);
#else
      data_[0] = data_[1] = value;
#endif
    }

    QuadVector(Sleef_quad a, Sleef_quad b)
    {
      // Universal fallback: create via load from heap memory
      Sleef_quad *temp = aligned_alloc<Sleef_quad>(2);
      if (temp)
      {
        temp[0] = a;
        temp[1] = b;
        *this = QuadVector::load(temp); // Use the working load function
        aligned_free(temp);
      }
      else
      {
        // Emergency fallback if allocation fails
#ifdef QUADBLAS_X86_64
        data_ = Sleef_splatq2_sse2(a); // At least get one value
#elif defined(QUADBLAS_AARCH64)
        data_ = Sleef_splatq2_advsimd(a);
#else
        data_[0] = a;
        data_[1] = a; // Not ideal but safe
#endif
      }
    }

    static QuadVector load(Sleef_quad *ptr)
    {
      QuadVector result;
#ifdef QUADBLAS_X86_64
      result.data_ = Sleef_loadq2_sse2(ptr);
#elif defined(QUADBLAS_AARCH64)
      result.data_ = Sleef_loadq2_advsimd(ptr);
#else
      result.data_[0] = ptr[0];
      result.data_[1] = ptr[1];
#endif
      return result;
    }

    void store(Sleef_quad *ptr) const
    {
#ifdef QUADBLAS_X86_64
      // Work around SLEEF store bug - use get instead
      ptr[0] = Sleef_getq2_sse2(data_, 0);
      ptr[1] = Sleef_getq2_sse2(data_, 1);
#elif defined(QUADBLAS_AARCH64)
      // Work around SLEEF store bug - use get instead
      ptr[0] = Sleef_getq2_advsimd(data_, 0);
      ptr[1] = Sleef_getq2_advsimd(data_, 1);
#else
      ptr[0] = data_[0];
      ptr[1] = data_[1];
#endif
    }

    Sleef_quad get(int index) const
    {
#ifdef QUADBLAS_X86_64
      return Sleef_getq2_sse2(data_, index);
#elif defined(QUADBLAS_AARCH64)
      return Sleef_getq2_advsimd(data_, index);
#else
      return data_[index];
#endif
    }

    QuadVector operator+(const QuadVector &other) const
    {
      QuadVector result;
#ifdef QUADBLAS_X86_64
      result.data_ = Sleef_addq2_u05sse2(data_, other.data_);
#elif defined(QUADBLAS_AARCH64)
      result.data_ = Sleef_addq2_u05advsimd(data_, other.data_);
#else
      result.data_[0] = Sleef_addq1_u05(data_[0], other.data_[0]);
      result.data_[1] = Sleef_addq1_u05(data_[1], other.data_[1]);
#endif
      return result;
    }

    QuadVector operator*(const QuadVector &other) const
    {
      QuadVector result;
#ifdef QUADBLAS_X86_64
      result.data_ = Sleef_mulq2_u05sse2(data_, other.data_);
#elif defined(QUADBLAS_AARCH64)
      result.data_ = Sleef_mulq2_u05advsimd(data_, other.data_);
#else
      result.data_[0] = Sleef_mulq1_u05(data_[0], other.data_[0]);
      result.data_[1] = Sleef_mulq1_u05(data_[1], other.data_[1]);
#endif
      return result;
    }

    QuadVector fma(const QuadVector &b, const QuadVector &c) const
    {
      QuadVector result;
#ifdef QUADBLAS_X86_64
      result.data_ = Sleef_fmaq2_u05sse2(data_, b.data_, c.data_);
#elif defined(QUADBLAS_AARCH64)
      result.data_ = Sleef_fmaq2_u05advsimd(data_, b.data_, c.data_);
#else
      result.data_[0] = Sleef_fmaq1_u05(data_[0], b.data_[0], c.data_[0]);
      result.data_[1] = Sleef_fmaq1_u05(data_[1], b.data_[1], c.data_[1]);
#endif
      return result;
    }

    Sleef_quad horizontal_sum() const
    {
#ifdef QUADBLAS_X86_64
      return Sleef_addq1_u05(Sleef_getq2_sse2(data_, 0), Sleef_getq2_sse2(data_, 1));
#elif defined(QUADBLAS_AARCH64)
      return Sleef_addq1_u05(Sleef_getq2_advsimd(data_, 0), Sleef_getq2_advsimd(data_, 1));
#else
      return Sleef_addq1_u05(data_[0], data_[1]);
#endif
    }
  };

} // namespace QuadBLAS

#endif // QUADBLAS_SIMD_QUAD_VECTOR_HPP