"""End-to-end throughput comparison: numpy float64 (scipy-openblas) vs
numpy-quaddtype (this QBLAS) for the BLAS-1/2/3 hot path.

This is the harness that produced the numbers in docs/perf_comparison_with_old.md
and in the QBLAS 1.5.0 release notes. Reproducing those numbers from a
clean checkout should give you the same shape (single-thread quad-FMA cost
~800x f64; multi-thread closes that gap to ~50-550x depending on size and
thread count). Exact values depend on host CPU, memory bandwidth, and
OpenBLAS tier.

Methodology (matching what docs/perf_comparison_with_old.md describes):
  - Pin both libraries to BENCH_THREADS via OMP_NUM_THREADS / OPENBLAS_NUM_THREADS
    before importing numpy. Same thread count on both sides; no asymmetric
    advantage.
  - For each (op, dtype, N): generate arrays once, warm up, then time an
    inner loop that is auto-calibrated so each timed sample takes >= 50 ms
    (or 1 iteration if a single call already exceeds 50 ms).
  - Repeat the timed sample N_REPEATS times. Report median (robust to GC
    and OS-noise outliers) and stdev/median as a stability indicator. Trim
    the fastest + slowest sample (Olympic trimming) when N_REPEATS >= 5.
  - Same arrays used for both libraries: no setup-cost asymmetry.

Requirements:
  pip install numpy-quaddtype   (which depends on numpy)

Usage:
  BENCH_THREADS=1  python bench/bench_quad_vs_numpy.py
  BENCH_THREADS=16 python bench/bench_quad_vs_numpy.py

Output paths can be redirected with BENCH_OUTPUT_DIR (default: $PWD).
"""
import os
import sys
import gc
import time
import json
from statistics import median, stdev

THREADS = int(os.environ.get("BENCH_THREADS", "16"))
os.environ.setdefault("OMP_NUM_THREADS",      str(THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS",      str(THREADS))
os.environ.setdefault("OMP_PROC_BIND",        "close")
os.environ.setdefault("OMP_PLACES",           "cores")

OUT_DIR = os.environ.get("BENCH_OUTPUT_DIR", os.getcwd())

import numpy as np
try:
    import numpy_quaddtype as nq
except ImportError:
    sys.exit("error: this harness requires numpy-quaddtype; install with "
             "`pip install numpy-quaddtype` or build it from source against this QBLAS.")

QPRC = nq.QuadPrecDType()
RNG  = np.random.default_rng(0)

TARGET_SAMPLE_S = 0.05   # auto-calibrate inner loop to >= this many seconds
N_WARMUP        = 3
N_REPEATS       = 9      # odd so median is well-defined; 9 -> trim top/bottom 2


def timed(fn, *args, target=TARGET_SAMPLE_S):
    """Return (per_call_seconds, stdev_seconds, n_inner, n_samples)."""
    # Calibrate inner loop count so each timed sample is >= `target` seconds.
    inner = 1
    while True:
        gc.collect(); gc.disable()
        t0 = time.perf_counter()
        for _ in range(inner):
            fn(*args)
        dt = time.perf_counter() - t0
        gc.enable()
        if dt >= target or inner >= 2048:
            break
        inner = max(2 * inner, int(inner * target / max(dt, 1e-9) * 1.2))

    for _ in range(N_WARMUP):
        for _ in range(inner):
            fn(*args)

    times = []
    for _ in range(N_REPEATS):
        gc.collect(); gc.disable()
        t0 = time.perf_counter()
        for _ in range(inner):
            fn(*args)
        times.append((time.perf_counter() - t0) / inner)
        gc.enable()

    if len(times) >= 5:
        times = sorted(times)[1:-1]
    med = median(times)
    sd  = stdev(times) if len(times) > 1 else 0.0
    return med, sd, inner, len(times)


def gops(fmas, t): return fmas / t / 1e9

def fmt_t(t):
    if t < 1e-3: return f"{t*1e6:7.1f} us"
    if t < 1:    return f"{t*1e3:7.2f} ms"
    return                  f"{t:7.3f} s "

def run_op(name, ops_per_call, dtype_label, fn, *args):
    med, sd, inner, n = timed(fn, *args)
    rate = gops(ops_per_call, med)
    rel  = sd / med if med > 0 else 0
    unit = "GFLOPS" if dtype_label == "f64" else "GFMA/s"
    print(f"  {name:<22} {fmt_t(med)}   {rate:8.3f} {unit:<8}  "
          f"+/-{rel*100:5.2f}%   (inner={inner}, n={n})")
    return med


print(f"# Threads:         {THREADS}")
print(f"# numpy:           {np.__version__}")
print(f"# numpy_quaddtype: {nq.get_quadblas_version()}")
print(f"# QBLAS threads:   {nq.get_num_threads()}")
print(f"# Methodology:     {N_WARMUP} warmup + {N_REPEATS} timed samples (median, trim hi/lo);")
print(f"#                   inner loop auto-calibrated to >= {int(TARGET_SAMPLE_S*1000)} ms per sample\n")

DOT_SIZES  = [1 << 16, 1 << 18, 1 << 20, 1 << 22]
GEMV_SIZES = [512, 1024, 2048]
# Skip n=1024 quad gemm at single thread (~4.6s/call x 9 samples = too slow).
GEMM_SIZES = [128, 256, 512, 1024] if THREADS > 1 else [128, 256, 512]

results = {}

print("== BLAS-1: dot (vector-vector inner product) ==")
for n in DOT_SIZES:
    xd = RNG.random(n);    yd = RNG.random(n)
    xq = xd.astype(QPRC);  yq = yd.astype(QPRC)
    fmas = n
    print(f" N = {n:>7}")
    td = run_op("f64",  fmas, "f64",  np.matmul, xd, yd)
    tq = run_op("quad", fmas, "quad", np.matmul, xq, yq)
    results[("dot", n)] = (td, tq)
    del xd, yd, xq, yq

print("\n== BLAS-2: gemv (matrix-vector product) ==")
for n in GEMV_SIZES:
    Ad = RNG.random((n, n)); xd = RNG.random(n)
    Aq = Ad.astype(QPRC);    xq = xd.astype(QPRC)
    fmas = 2 * n * n
    print(f" N = {n:>5}")
    td = run_op("f64",  fmas, "f64",  np.matmul, Ad, xd)
    tq = run_op("quad", fmas, "quad", np.matmul, Aq, xq)
    results[("gemv", n)] = (td, tq)
    del Ad, xd, Aq, xq

print("\n== BLAS-3: gemm (matrix-matrix product) ==")
for n in GEMM_SIZES:
    Ad = RNG.random((n, n)); Bd = RNG.random((n, n))
    Aq = Ad.astype(QPRC);    Bq = Bd.astype(QPRC)
    fmas = 2 * n * n * n
    print(f" N = {n:>5}")
    td = run_op("f64",  fmas, "f64",  np.matmul, Ad, Bd)
    tq = run_op("quad", fmas, "quad", np.matmul, Aq, Bq)
    results[("gemm", n)] = (td, tq)
    del Ad, Bd, Aq, Bq

print("\n" + "="*78)
print(f"SUMMARY  (threads={THREADS})")
print("="*78)
print(f"{'op':<5} {'N':>6} {'f64 time':>11} {'f64 rate':>13}   {'quad time':>11} {'quad rate':>13}   slowdown")
print("-"*78)
for (op, n), (td, tq) in results.items():
    fmas = {"dot": n, "gemv": 2*n*n, "gemm": 2*n*n*n}[op]
    print(f"{op:<5} {n:>6} {fmt_t(td):>11} {gops(fmas, td):>9.2f} GFLOPS"
          f"   {fmt_t(tq):>11} {gops(fmas, tq):>8.3f} GFMA/s   {tq/td:>6.0f}x")

out_path = os.path.join(OUT_DIR, f"bench_quad_vs_numpy_t{THREADS}.json")
with open(out_path, "w") as f:
    json.dump({
        "threads":         THREADS,
        "numpy_version":   np.__version__,
        "qblas_version":   nq.get_quadblas_version(),
        "results":         {f"{op}/n={n}": {"f64_s": td, "quad_s": tq}
                            for (op, n), (td, tq) in results.items()},
    }, f, indent=2)
print(f"\nWrote {out_path}")
