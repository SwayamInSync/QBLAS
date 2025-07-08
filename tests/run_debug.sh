g++ -O3 -fopenmp \
    -I/datadrive/guestuser/t-swsingh/temp/sleef/build/include \
    -L/datadrive/guestuser/t-swsingh/temp/sleef/build/lib \
    -Wl,-rpath,/datadrive/guestuser/t-swsingh/temp/sleef/build/lib \
    -o benchmark_threaded benchmark.cpp \
    -lsleef -lsleefquad

./benchmark_threaded