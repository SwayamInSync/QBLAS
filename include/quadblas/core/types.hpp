#ifndef QUADBLAS_CORE_TYPES_HPP
#define QUADBLAS_CORE_TYPES_HPP

namespace QuadBLAS
{

  // Matrix layout types
  enum class Layout
  {
    RowMajor,
    ColMajor
  };

  // Forward declarations (no default arguments here to avoid redefinition)
  template <Layout layout>
  class Matrix;

  template <Layout layout>
  class Vector;

} // namespace QuadBLAS

#endif // QUADBLAS_CORE_TYPES_HPP