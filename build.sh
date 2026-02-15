#!/bin/bash
set -e

# Clean and setup
rm -rf build
mkdir build
cd build

# Environment setup
export SLEEF_ROOT="/Users/swayam/Desktop/numpy_dtypes/sleef/build"

# Detect available parallelism
if command -v nproc &> /dev/null; then
    NJOBS=$(nproc)
elif command -v sysctl &> /dev/null; then
    NJOBS=$(sysctl -n hw.ncpu)
else
    NJOBS=4
fi

# Initialize submodules if needed
if [ ! -f "../third_party/benchmark/CMakeLists.txt" ]; then
    echo "Initializing Google Benchmark submodule..."
    (cd .. && git submodule update --init --recursive)
fi

# Build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j"${NJOBS}"

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
