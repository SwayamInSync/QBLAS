#!/usr/bin/env bash
# Build SLEEF with full SIMD path coverage (sse2/avx2/avx512f on x86, advsimd
# on aarch64) and install into the repo-local prefix .sleef-prefix/.  This
# is what the top-level CMakeLists.txt finds by default.
#
# Why a script: SLEEF's default CMake config only builds for the host CPU,
# so a binary built on this machine would have no SSE2 quad path and would
# crash on older CPUs.  Forcing every tier on guarantees runtime portability.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
sleef_src="${repo}/third_party/sleef"
sleef_build="${sleef_src}/build-local"
prefix="${repo}/.sleef-prefix"

if [[ ! -f "${sleef_src}/CMakeLists.txt" ]]; then
    echo "[bootstrap] initialising sleef submodule..."
    (cd "${repo}" && git submodule update --init --recursive third_party/sleef)
fi

mkdir -p "${sleef_build}"
cmake -S "${sleef_src}" -B "${sleef_build}" \
    -DSLEEF_BUILD_QUAD=ON \
    -DSLEEF_BUILD_SHARED_LIBS=ON \
    -DSLEEF_ENABLE_SSE2=ON \
    -DSLEEF_ENABLE_SSE4=ON \
    -DSLEEF_ENABLE_AVX=ON \
    -DSLEEF_ENABLE_AVX2=ON \
    -DSLEEF_ENABLE_AVX512F=ON \
    -DSLEEF_DISABLE_NATIVE_KEYWORD=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${prefix}"

if command -v nproc >/dev/null 2>&1; then
    NJOBS=$(nproc)
elif [[ "$(uname)" == "Darwin" ]]; then
    NJOBS=$(sysctl -n hw.ncpu)
else
    NJOBS=4
fi
cmake --build "${sleef_build}" --parallel "${NJOBS}"

rm -rf "${prefix}"
cmake --install "${sleef_build}"

# SLEEF's install rules don't include the vendored libtlfloat artifact; copy
# it manually so binaries can load it via rpath/runpath.
tl_dir="${sleef_build}/ext_tlfloat-prefix/src/ext_tlfloat-build/lib"
if [[ -d "${tl_dir}" ]]; then
    if [[ "$(uname)" == "Darwin" ]]; then
        cp -a "${tl_dir}"/libtlfloat.*.dylib "${prefix}/lib/" 2>/dev/null || true
        cp -a "${tl_dir}"/libtlfloat.dylib   "${prefix}/lib/" 2>/dev/null || true
    else
        cp -a "${tl_dir}"/libtlfloat.so* "${prefix}/lib/"
    fi
fi

echo "[bootstrap] SLEEF installed at ${prefix}"
