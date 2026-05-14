#!/usr/bin/env bash
# Build SLEEF with all SIMD paths enabled and install into .sleef-prefix/.
# SLEEF's default build only emits host-CPU paths, which would crash on
# older CPUs at runtime once qblas dispatches to a lower tier.
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

# libtlfloat is built by SLEEF but not installed; copy it in so rpath
# resolution finds it.
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
