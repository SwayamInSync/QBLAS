#ifndef QUADBLAS_THREADING_OPENMP_UTILS_HPP
#define QUADBLAS_THREADING_OPENMP_UTILS_HPP

#include "../core/platform.hpp"

namespace QuadBLAS
{

  // Utility functions for threading
  inline int get_num_threads()
  {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
  }

  inline void set_num_threads(int num_threads)
  {
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif
  }

} // namespace QuadBLAS

#endif // QUADBLAS_THREADING_OPENMP_UTILS_HPP