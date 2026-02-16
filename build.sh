#!/bin/bash

# Clean and setup
rm -rf build
mkdir build
cd build

# Environment setup
export SLEEF_ROOT="/Users/swayam/Desktop/numpy_dtypes/sleef/build"

# Build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)


echo "=== Running Tests ==="
./quadblas_test

# Google Benchmark
echo ""
echo "=== Running Google Benchmarks ==="
if [ -f "./quadblas_benchmark" ]; then
    ./quadblas_benchmark --benchmark_min_time=0.5s
else
    echo "Google Benchmark not available, running legacy..."
    ./quadblas_benchmark_legacy
fi