# QBLAS — remaining performance headroom

This file captures the perf opportunities the current rewrite did **not**
take.  Each section names the win, the rough size, the cost (in
engineering and/or precision), and an honest recommendation.

Use this as the decision sheet for the next perf push.

---

## 1.  Hand-tuned DD-FMA micro-kernel (the big one)

**The ceiling we hit.**  Single-thread GEMM peaks at ~30 M quad-FMAs/sec/core
on AVX2.  A tight microbench with 8 independent FMAs in flight tops out at
the same 30.8 M/core, which is exactly the throughput of
`Sleef_fmaq4_u05avx2`.  We're at the SLEEF symbol's ceiling.

Measured: see `/tmp/sleef_peak.c` style microbench (Zen 3 @ ~3 GHz):
~97 cycles per quad-FMA call.

**What it would unlock.**  ~2–3× single-thread, multiplied across all
cores.  Concretely, expectation:
- Single-thread: 30 M → ~80–100 M quad-FMAs/sec/core
- 16-thread GEMM: 430 M → ~1.2 G items/sec
- 96-thread GEMM: 1.6 G → ~5 G items/sec

**Why we didn't do it.**  Writing a u05-accurate quad FMA on AVX2 doubles
from scratch is real work.  SLEEF's u05 uses **triple-double (TD, 3-term)**
internally, which is more involved than the classical 2-term DD.  A
straightforward AVX2 DD-FMA implementation gives ~106-bit accuracy vs
quad's 113 bits — a ~7-bit (≈2 decimal-digit) precision loss.

**Decision needed.**  Three options:

| option | accuracy | est. speedup vs current | effort |
|---|---|---|---|
| (a) Keep SLEEF u05 (status quo) | 0.5 ULP (~113 bits) | 1× | 0 |
| (b) Hand DD-FMA, u10-like (~106 bits, 1.5–2 ULP)| ~106 bits | 2.5–3× per core | ~1 week |
| (c) Inline-DD via SLEEF's `sleefquadinline_avx2.h` | 0.5 ULP (same) | ~1.05× | 2 days |
| (d) Hand TD-FMA replicating SLEEF u05 | 0.5 ULP (same) | hard to predict, ~1.2–1.5× best | ~2 weeks |

Recommendation: **(c) first** as a near-zero-risk 5 % win, then **(b)** if
losing ~7 bits is acceptable for the perf jump.  The IEEE-754 binary128
guarantee in the README would need a footnote for (b).

---

## 2.  Header-only "inline" path (for small-N callers)

**The gap.**  Compiled-library dispatch + function-call overhead costs
the new build ~3–5 % at n ≤ 1 000 for L1 ops (dot, axpy, nrm2) compared
to the old header-only build.  Single-call overhead is ~50 ns and below
~1 K elements that's a measurable fraction of total cost.

**What it would unlock.**  Close the small-N regression entirely;
possibly modest wins at medium-N too if the compiler can fuse user code
with our inner loops.

**Why we didn't do it.**  We chose compiled-library by default because
of per-ISA `-march` correctness and stable C ABI (see project notes).
A *companion* `qblas/qblas_inline.h` could keep both modes.

**Decision needed.**  Should we ship an inline header alongside?
- (a) No: keep compiled-only.  Simpler.
- (b) Yes: ship `qblas_inline.h` that pulls in the chosen ISA tier
  inline.  Consumer chooses `-DQBLAS_INLINE` at build.
- (c) BLIS-style: split the C API into `qblas_*` (compiled) and
  `qblas_inline_*` (header).

Recommendation: **(b)**, only if a user actually asks for it.

---

## 3.  AVX-512 tier untested on hardware

**The gap.**  Code compiles `qblas_kernels_avx512.c` against `__m512`
SLEEF symbols (`Sleef_fmaq8_u05avx512f`), and the dispatcher *would* pick
it on a Sapphire Rapids / Skylake-X box, but it has never been run there.
There's a strong chance of small bugs in the q=8 width branch.

**What it would unlock.**  q8 vs q4 → 2× quad-FMA throughput per core if
SLEEF's q8 fma is fully 2-way wider than q4. Realistically expect 1.4–
1.7×.

**Why we didn't do it.**  We don't have AVX-512 hardware in the bench
loop.  Building works (clang/gcc with `-mavx512f`), but the runtime
dispatch on this Zen 3 always picks AVX2.

**Decision needed.**  Run on an AVX-512 host.  Two paths:
- Spin up a Sapphire Rapids VM in Azure for one CI run.
- Skip until a user opens the issue.

Recommendation: add an **AVX-512 host to CI** so the q8 path gets at
least built and tested on every push.

---

## 4.  Level-2: qsymv, qtrsv, qger micro-kernels

**The gap.**  qgemv (both N and T) is properly vectorised + threaded.
The other three Level-2 routines still drop through to scalar +
qblas_dispatch_qaxpy / qdot.

**Per current bench (16 threads, n=1600):**
- qgemv: **431 M items/s** (peak)
- qger: not benched, expected ~250–350 M/s (uses qaxpy per row)
- qsymv: not benched, expected ~200–300 M/s (two passes)
- qtrsv: not benched, sequential (only the trailing axpy is SIMD)

**What it would unlock.**  qger directly enters LU factorisation hot
paths; qsymv is the heart of eigen / Cholesky.  ~2× would be reasonable
for proper blocking.

**Why we didn't do it.**  Time.  qger and qsymv are easy follow-ups
(panel + gemv pattern).  qtrsv is inherently serial along one axis but
can be blocked the same way as qtrsm.

**Decision needed.**  Schedule the work after the bigger ceilings get
cracked.

---

## 5.  L1 small-N parallel break-even

**The gap.**  Below n ≈ 1 000 we run single-thread because OpenMP fork
costs ≈ 18 µs and processing 1 000 quad ops at 30 M/s = 33 µs of work —
the parallel speedup is consumed by the fork.

**What it would unlock.**  Not much, frankly.  Maybe 1.2–1.5× at n=1 000.
The break-even is fundamental to OpenMP's region cost.

**Why we didn't do it.**  Marginal.

**Decision needed.**  If small-N matters for a specific user, switching
to a persistent thread pool (libomp's `LIBOMP_*` env vars + `OMP_PROC_BIND`)
would help. Currently we don't touch any of that.

---

## 6.  NUMA-aware allocation on multi-socket hosts

**The gap.**  `qblas_aligned_alloc` is just `posix_memalign` — no
first-touch policy, no node binding.  On a 2-socket EPYC, GEMM panels
allocated in `qgemm()` may live on the wrong NUMA node for some threads,
costing 1.5–2× on the affected accesses.

**What it would unlock.**  ~1.3–1.5× on dual-socket GEMM, maybe more on
multi-thread Level 1 with large arrays.

**Why we didn't do it.**  The current bench machine has all 96 cores in
one logical view; we haven't seen multi-socket behavior.

**Decision needed.**  Implement when the issue first surfaces on a real
2P box.  `numa_alloc_onnode` + libnuma dependency.

---

## 7.  Persistent packed-buffer cache for GEMM

**The gap.**  Every `cblas_qgemm` call inside `cblas_qtrsm` etc. allocates
new `Ap` and `Bp` buffers inside its parallel region.  For a trsm that
calls gemm O(m / NB) times, that's m/NB allocations × n_threads.

**What it would unlock.**  ~5–10 % on routines that loop on gemm
(trsm/trmm specifically).  Negligible for one-shot gemm callers.

**Why we didn't do it.**  Adds lifecycle complexity (per-thread arena
needs a clean-up hook).

**Decision needed.**  Yes for the next perf push.

---

## How to evaluate the impact of any of these

Repro path for any change:

```bash
cd build
ctest                                  # correctness MUST stay green
OMP_NUM_THREADS=16 ./bench/qblas_bench_compare \
    --benchmark_min_time=0.3s --benchmark_repetitions=3 \
    --benchmark_report_aggregates_only=true
# Compare against PERF_COMPARISON.md numbers.
```

Single change at a time.  Keep the bench harness untouched.
