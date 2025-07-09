rm -rf build
mkdir build
cd build

export SLEEF_ROOT=/datadrive/guestuser/t-swsingh/temp/sleef/build
cmake ..
make
./quadblas_test
./quadblas_benchmark