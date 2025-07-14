rm -rf build
mkdir build
cd build

export SLEEF_ROOT="/Users/swayam/Desktop/numpy_dtypes/sleef/build"
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
cmake ..
make
./quadblas_test
./quadblas_benchmark