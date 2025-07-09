#ifndef QUADBLAS_MEMORY_ALLOCATION_HPP
#define QUADBLAS_MEMORY_ALLOCATION_HPP

#include "../core/constants.hpp"
#include <cstddef>
#include <cstdlib>

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace QuadBLAS
{

  // Aligned memory allocation
  template <typename T>
  inline T *aligned_alloc(size_t count)
  {
    void *ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(count * sizeof(T), ALIGNMENT);
#else
    if (posix_memalign(&ptr, ALIGNMENT, count * sizeof(T)) != 0)
    {
      ptr = nullptr;
    }
#endif
    return static_cast<T *>(ptr);
  }

  template <typename T>
  inline void aligned_free(T *ptr)
  {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

} // namespace QuadBLAS

#endif // QUADBLAS_MEMORY_ALLOCATION_HPP