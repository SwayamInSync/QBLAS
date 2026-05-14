# Meson build for QBLAS — equivalent to the CMake build but designed to
# be consumable as a Meson subproject by downstream packages (notably
# numpy-quaddtype).
#
# Produces a static library `libqblas.a` plus a `qblas_dep` dependency
# that includes both the public headers and the link line.  Downstream
# only needs to do `dependency('qblas', fallback: ['qblas', 'qblas_dep'])`.

project('qblas', 'c',
        version: '0.1.0',
        default_options: ['c_std=c11', 'b_pie=true', 'b_ndebug=if-release'],
        meson_version: '>=1.1')

cc = meson.get_compiler('c')

# ---------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------
# SLEEF is the only mandatory dep.  Downstream (numpy-quaddtype) already
# pulls SLEEF in via its own wrap; standalone builds find it via the
# install prefix's pkg-config.  SLEEF doesn't ship a .pc file for the
# quad library, so we have to find it manually as a plain library.
sleef_dep = dependency('sleef', required: false)
if not sleef_dep.found()
    sleef_dep = cc.find_library('sleef', required: true)
endif
sleefquad_dep = dependency('sleefquad', required: false)
if not sleefquad_dep.found()
    sleefquad_dep = cc.find_library('sleefquad', required: true)
endif

openmp_dep = dependency('openmp', required: false)
if openmp_dep.found()
    omp_args = ['-DQBLAS_USE_OPENMP=1']
else
    omp_args = []
endif

# ---------------------------------------------------------------------
# Architecture / per-ISA tier discovery
# ---------------------------------------------------------------------
arch = host_machine.cpu_family()
is_x86 = (arch == 'x86_64' or arch == 'x86')
is_arm = (arch == 'aarch64' or arch == 'arm64')

isa_tiers = ['generic']
if is_x86
    isa_tiers += ['sse2', 'avx2', 'avx512']
elif is_arm
    isa_tiers += ['neon']
endif

isa_flags = {
    'generic': [],
    'sse2'   : ['-msse2'],
    'avx2'   : ['-mavx2', '-mfma', '-mbmi2'],
    'avx512' : ['-mavx512f', '-mavx512dq', '-mavx512bw', '-mfma', '-mbmi2'],
    'neon'   : [],
}
isa_has_defines = {
    'generic': '-DQBLAS_HAS_GENERIC=1',
    'sse2'   : '-DQBLAS_HAS_SSE2=1',
    'avx2'   : '-DQBLAS_HAS_AVX2=1',
    'avx512' : '-DQBLAS_HAS_AVX512=1',
    'neon'   : '-DQBLAS_HAS_NEON=1',
}

# Build each ISA tier as a static library, then aggregate.
qblas_inc = include_directories('include', 'src')
qblas_core_args = ['-O3', '-D_POSIX_C_SOURCE=200809L'] + omp_args
foreach tier : isa_tiers
    foreach d : isa_has_defines.keys()
        # nothing — we add all tier flags to the main lib below
    endforeach
endforeach

tier_libs = []
foreach tier : isa_tiers
    tier_lib = static_library(
        'qblas_kernels_@0@'.format(tier),
        'src/kernels/@0@/kernels_@0@.c'.format(tier),
        c_args: qblas_core_args + isa_flags[tier] + [isa_has_defines[tier]],
        include_directories: qblas_inc,
        dependencies: [sleef_dep, sleefquad_dep],
        install: false,
    )
    tier_libs += tier_lib
endforeach

# Per-arch HAS_* defines for the core (so qblas_cpu.c knows which
# tier-registration functions are available to call).
core_defines = ['-DQBLAS_HAS_GENERIC=1']
if is_x86
    core_defines += ['-DQBLAS_HAS_SSE2=1', '-DQBLAS_HAS_AVX2=1', '-DQBLAS_HAS_AVX512=1']
elif is_arm
    core_defines += ['-DQBLAS_HAS_NEON=1']
endif

core_srcs = [
    'src/cpu/qblas_cpu.c',
    'src/level1/qblas_level1.c',
    'src/level2/qblas_level2.c',
    'src/level3/qblas_gemm.c',
    'src/level3/qblas_level3_extra.c',
]

qblas_lib = static_library(
    'qblas',
    core_srcs,
    c_args: qblas_core_args + core_defines,
    include_directories: qblas_inc,
    link_with: tier_libs,
    dependencies: [sleef_dep, sleefquad_dep] + (openmp_dep.found() ? [openmp_dep] : []),
    install: false,
)

qblas_dep = declare_dependency(
    link_with: qblas_lib,
    include_directories: include_directories('include'),
    dependencies: [sleef_dep, sleefquad_dep] + (openmp_dep.found() ? [openmp_dep] : []),
    version: meson.project_version(),
)
