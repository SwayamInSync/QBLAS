#ifndef QUADBLAS_CORE_HPP
#define QUADBLAS_CORE_HPP

#include <sleefquad.h>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

// Platform detection
#if defined(__x86_64__) || defined(_M_X64)
    #define QUADBLAS_X86_64
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define QUADBLAS_AARCH64
    #include <arm_neon.h>
#endif

namespace QuadBLAS {

// Configuration constants
constexpr size_t VECTOR_SIZE = 2;  // Sleef quad vector size
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t L1_CACHE_SIZE = 32768;
constexpr size_t L2_CACHE_SIZE = 262144;
constexpr size_t PARALLEL_THRESHOLD = 1000;
constexpr size_t GEMM_BLOCK_SIZE = 64;

// Memory alignment for SIMD operations
constexpr size_t ALIGNMENT = 32;

// Aligned memory allocation
template<typename T>
inline T* aligned_alloc(size_t count) {
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(count * sizeof(T), ALIGNMENT);
#else
    if (posix_memalign(&ptr, ALIGNMENT, count * sizeof(T)) != 0) {
        ptr = nullptr;
    }
#endif
    return static_cast<T*>(ptr);
}

template<typename T>
inline void aligned_free(T* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Matrix layout types
enum class Layout {
    RowMajor,
    ColMajor
};

// Forward declarations (no default arguments here)
template<Layout layout>
class Matrix;

template<Layout layout>
class Vector;

// SIMD wrapper for platform abstraction
class QuadVector {
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
    
    explicit QuadVector(Sleef_quad value) {
#ifdef QUADBLAS_X86_64
        data_ = Sleef_splatq2_sse2(value);
#elif defined(QUADBLAS_AARCH64)
        data_ = Sleef_splatq2_advsimd(value);
#else
        data_[0] = data_[1] = value;
#endif
    }
    
    QuadVector(Sleef_quad a, Sleef_quad b) {
        // Use scalar approach for constructor since it's not performance-critical
        // The load() function (which is performance-critical) works perfectly
#if defined(QUADBLAS_X86_64) || defined(QUADBLAS_AARCH64)
        // For SIMD platforms, we'll store values and reconstruct via get/set operations
        // This is a workaround for SLEEF load issues with stack arrays
        // Since this constructor is not used in hot paths, simplicity is preferred
        data_ = Sleef_splatq2_sse2(SLEEF_QUAD_C(0.0));  // Initialize
        // Unfortunately, there's no direct way to set individual elements in Sleef_quadx2
        // So we'll create two single-element vectors and combine them
        // For now, use the fallback approach
#endif
        
        // Universal fallback: create via load from heap memory
        Sleef_quad* temp = aligned_alloc<Sleef_quad>(2);
        if (temp) {
            temp[0] = a;
            temp[1] = b;
            *this = QuadVector::load(temp);  // Use the working load function
            aligned_free(temp);
        } else {
            // Emergency fallback if allocation fails
#ifdef QUADBLAS_X86_64
            data_ = Sleef_splatq2_sse2(a);  // At least get one value
#elif defined(QUADBLAS_AARCH64)
            data_ = Sleef_splatq2_advsimd(a);
#else
            data_[0] = a;
            data_[1] = a;  // Not ideal but safe
#endif
        }
    }
    
    static QuadVector load(Sleef_quad* ptr) {
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
    
    void store(Sleef_quad* ptr) const {
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
    
    Sleef_quad get(int index) const {
#ifdef QUADBLAS_X86_64
        return Sleef_getq2_sse2(data_, index);
#elif defined(QUADBLAS_AARCH64)
        return Sleef_getq2_advsimd(data_, index);
#else
        return data_[index];
#endif
    }
    
    QuadVector operator+(const QuadVector& other) const {
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
    
    QuadVector operator*(const QuadVector& other) const {
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
    
    QuadVector fma(const QuadVector& b, const QuadVector& c) const {
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
    
    Sleef_quad horizontal_sum() const {
#ifdef QUADBLAS_X86_64
        return Sleef_addq1_u05(Sleef_getq2_sse2(data_, 0), Sleef_getq2_sse2(data_, 1));
#elif defined(QUADBLAS_AARCH64)
        return Sleef_addq1_u05(Sleef_getq2_advsimd(data_, 0), Sleef_getq2_advsimd(data_, 1));
#else
        return Sleef_addq1_u05(data_[0], data_[1]);
#endif
    }
};

// Utility functions for threading
inline int get_num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline void set_num_threads(int num_threads) {
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif
}

// Cache-friendly blocking parameters
struct BlockingParams {
    size_t mc, kc, nc;  // Panel sizes for GEMM
    
    BlockingParams(size_t m = 0, size_t n = 0, size_t k = 0) {
        // Calculate optimal blocking sizes based on cache hierarchy
        const size_t quad_size = sizeof(Sleef_quad);
        const size_t l1_quads = L1_CACHE_SIZE / (3 * quad_size);
        const size_t l2_quads = L2_CACHE_SIZE / quad_size;
        
        if (m * n * k == 0) {
            // Default blocking
            mc = std::min(static_cast<size_t>(GEMM_BLOCK_SIZE), l1_quads / 4);
            kc = std::min(static_cast<size_t>(GEMM_BLOCK_SIZE), l1_quads / 4);
            nc = std::min(static_cast<size_t>(GEMM_BLOCK_SIZE), l2_quads / (mc + kc));
        } else {
            // Adaptive blocking based on matrix sizes
            mc = std::min({m, static_cast<size_t>(GEMM_BLOCK_SIZE), l1_quads / 4});
            kc = std::min({k, static_cast<size_t>(GEMM_BLOCK_SIZE), l1_quads / 4});
            nc = std::min({n, static_cast<size_t>(GEMM_BLOCK_SIZE), l2_quads / (mc + kc)});
        }
        
        // Ensure sizes are multiples of vector size for better vectorization
        mc = (mc / VECTOR_SIZE) * VECTOR_SIZE;
        nc = (nc / VECTOR_SIZE) * VECTOR_SIZE;
        
        // Minimum sizes
        mc = std::max(mc, VECTOR_SIZE);
        kc = std::max(kc, static_cast<size_t>(4));
        nc = std::max(nc, VECTOR_SIZE);
    }
};

} // namespace QuadBLAS

#endif // QUADBLAS_CORE_HPP