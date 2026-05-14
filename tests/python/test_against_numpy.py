"""Compare QBLAS results (rounded to float64) against numpy's float64
reference for every routine.  This is the user-facing correctness check:
if you cast the quad output back to double, you get the same answer
numpy gives, up to a tolerance derived from double-precision rounding.

Tolerances
----------
QBLAS computes in quad (~1e-34 ULP), numpy in double (~2.22e-16 ULP).
After rounding QBLAS back to double, the two should agree to within
~1e-13 relative for accumulated-rounding-friendly problems and looser
for ill-conditioned ones.  We pick:

  * 1e-13 for elementwise L1/axpy/scal/copy
  * 1e-11 for inner products (dot/nrm2 accumulate n rounding errors)
  * 1e-10 for matrix-vector
  * 1e-9  for matrix-matrix (k FMAs of accumulation)

Random data ~U(-1, 1) keeps condition numbers reasonable.
"""

from __future__ import annotations

import sys
import numpy as np

import qblas_ctypes as qb

RNG = np.random.default_rng(2026_05_14)


def rel_err(actual: np.ndarray | float, expected: np.ndarray | float) -> float:
    a = np.asarray(actual,   dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    num = np.linalg.norm((a - e).ravel(), ord=np.inf)
    den = np.linalg.norm(e.ravel(), ord=np.inf) + 1e-300
    return float(num / den)


def gen_vec(n: int, scale: float = 1.0) -> np.ndarray:
    return RNG.uniform(-scale, scale, size=n)


def gen_mat(m: int, n: int, scale: float = 1.0) -> np.ndarray:
    return RNG.uniform(-scale, scale, size=(m, n))


# ----------------------------------------------------------------------
# Test runner
# ----------------------------------------------------------------------
PASSED = 0
FAILED = 0
FAILS: list[str] = []

def check(name: str, actual, expected, tol: float):
    global PASSED, FAILED
    try:
        r = rel_err(actual, expected)
    except Exception as e:
        FAILED += 1
        FAILS.append(f"{name}: EXCEPTION {e}")
        return
    if r <= tol:
        PASSED += 1
    else:
        FAILED += 1
        FAILS.append(f"{name}: rel_err={r:.3e} > tol={tol:.0e}")


# ----------------------------------------------------------------------
# Level 1
# ----------------------------------------------------------------------
def test_qdot():
    for n in (1, 7, 64, 1023, 4096, 16384):
        x = gen_vec(n); y = gen_vec(n)
        check(f"qdot n={n}", qb.qdot(x, y), x @ y, 1e-11)


def test_qnrm2():
    for n in (1, 7, 64, 1023, 4096, 16384):
        x = gen_vec(n)
        check(f"qnrm2 n={n}", qb.qnrm2(x), np.linalg.norm(x), 1e-12)


def test_qasum():
    for n in (1, 7, 64, 1023, 4096, 16384):
        x = gen_vec(n)
        check(f"qasum n={n}", qb.qasum(x), np.sum(np.abs(x)), 1e-12)


def test_iqamax():
    for n in (1, 7, 64, 1023, 4096):
        x = gen_vec(n)
        if abs(np.max(np.abs(x)) - np.min(np.abs(x))) < 1e-12:
            continue  # ambiguous tie
        got = qb.iqamax(x)
        want = int(np.argmax(np.abs(x)))
        check(f"iqamax n={n}", got, want, 0)  # exact integer match


def test_qaxpy():
    for n in (1, 7, 64, 1023, 4096, 16384):
        x = gen_vec(n); y = gen_vec(n); alpha = float(RNG.uniform(-2, 2))
        got = qb.qaxpy(alpha, x, y)
        want = alpha * x + y
        check(f"qaxpy n={n}", got, want, 1e-13)


def test_qscal():
    for n in (1, 7, 64, 1023, 4096, 16384):
        x = gen_vec(n); alpha = float(RNG.uniform(-2, 2))
        got = qb.qscal(alpha, x)
        want = alpha * x
        check(f"qscal n={n}", got, want, 1e-13)


# ----------------------------------------------------------------------
# Level 2
# ----------------------------------------------------------------------
def test_qgemv():
    for (m, n) in [(1, 1), (5, 7), (32, 32), (64, 33), (127, 65)]:
        for trans in (qb.QblasNoTrans, qb.QblasTrans):
            A = gen_mat(m, n)
            xn = n if trans == qb.QblasNoTrans else m
            yn = m if trans == qb.QblasNoTrans else n
            x = gen_vec(xn); y = gen_vec(yn)
            alpha = float(RNG.uniform(-2, 2))
            beta  = float(RNG.uniform(-2, 2))
            got = qb.qgemv(A, x, y.copy(), alpha=alpha, beta=beta, trans=trans)
            opA = A if trans == qb.QblasNoTrans else A.T
            want = alpha * (opA @ x) + beta * y
            check(f"qgemv m={m} n={n} trans={trans}", got, want, 1e-10)


# ----------------------------------------------------------------------
# Level 3
# ----------------------------------------------------------------------
def test_qger():
    for (m, n) in [(1, 1), (5, 7), (32, 32), (64, 33)]:
        A = gen_mat(m, n); x = gen_vec(m); y = gen_vec(n)
        alpha = float(RNG.uniform(-2, 2))
        got = qb.qger(A.copy(), x, y, alpha=alpha)
        want = A + alpha * np.outer(x, y)
        check(f"qger m={m} n={n}", got, want, 1e-12)


def test_qsyrk():
    for (n, k) in [(3, 3), (8, 5), (17, 11), (32, 32)]:
        A = gen_mat(n, k)
        C = gen_mat(n, n)
        alpha = float(RNG.uniform(-2, 2))
        beta  = float(RNG.uniform(-2, 2))
        got = qb.qsyrk(A, C.copy(), alpha=alpha, beta=beta,
                       uplo=qb.QblasUpper, trans=qb.QblasNoTrans)
        want = alpha * (A @ A.T) + beta * C
        # Only the upper triangle is meaningful for the spec; compare full.
        tol = max(1e-12, 1e-14 * k)
        check(f"qsyrk n={n} k={k}", got, want, tol)


def test_qtrmm():
    for (m, n) in [(3, 3), (8, 5), (17, 11), (64, 33)]:
        # Diagonally-dominant lower triangular A for stability.
        A = np.tril(gen_mat(m, m, scale=0.5))
        for i in range(m): A[i, i] = float(RNG.uniform(1.5, 2.5))
        B = gen_mat(m, n)
        alpha = float(RNG.uniform(-2, 2))
        got = qb.qtrmm(A, B.copy(), alpha=alpha,
                       side=qb.QblasLeft, uplo=qb.QblasLower,
                       trans=qb.QblasNoTrans, diag=qb.QblasNonUnit)
        want = alpha * (A @ B)
        tol = max(1e-12, 1e-14 * m)
        check(f"qtrmm m={m} n={n}", got, want, tol)


def test_qtrsm():
    for (m, n) in [(3, 3), (8, 5), (17, 11), (64, 33), (129, 11)]:
        A = np.tril(gen_mat(m, m, scale=0.5))
        for i in range(m): A[i, i] = float(RNG.uniform(1.5, 2.5))
        B = gen_mat(m, n)
        alpha = float(RNG.uniform(-2, 2))
        got = qb.qtrsm(A, B.copy(), alpha=alpha,
                       side=qb.QblasLeft, uplo=qb.QblasLower,
                       trans=qb.QblasNoTrans, diag=qb.QblasNonUnit)
        # Want: solve A * X = alpha * B for X.  numpy gives that as
        # np.linalg.solve(A, alpha * B) for the lower-triangular A.
        want = np.linalg.solve(A, alpha * B)
        tol = max(1e-11, 1e-14 * m * m)  # solver error grows like cond * eps
        check(f"qtrsm m={m} n={n}", got, want, tol)


def test_qgemm():
    for (m, n, k) in [(1, 1, 1), (3, 5, 7), (16, 16, 16),
                      (17, 19, 23), (64, 64, 64), (128, 65, 33)]:
        for ta in (qb.QblasNoTrans, qb.QblasTrans):
            for tb in (qb.QblasNoTrans, qb.QblasTrans):
                A_shape = (m, k) if ta == qb.QblasNoTrans else (k, m)
                B_shape = (k, n) if tb == qb.QblasNoTrans else (n, k)
                A = gen_mat(*A_shape)
                B = gen_mat(*B_shape)
                C = gen_mat(m, n)
                alpha = float(RNG.uniform(-2, 2))
                beta  = float(RNG.uniform(-2, 2))
                got = qb.qgemm(A, B, C.copy(),
                               alpha=alpha, beta=beta, ta=ta, tb=tb)
                opA = A if ta == qb.QblasNoTrans else A.T
                opB = B if tb == qb.QblasNoTrans else B.T
                want = alpha * (opA @ opB) + beta * C
                # Tolerance scales with k (each output element is sum of k FMAs).
                tol = max(1e-12, 1e-14 * k)
                check(f"qgemm m={m} n={n} k={k} ta={ta} tb={tb}",
                      got, want, tol)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> int:
    print(f"QBLAS dispatch tier  : {qb.dispatch_tier()}")
    print(f"QBLAS threads        : {qb._qblas.qblas_get_num_threads()}")
    print(f"numpy version        : {np.__version__}")
    print()

    tests = [
        ("qdot",   test_qdot),
        ("qnrm2",  test_qnrm2),
        ("qasum",  test_qasum),
        ("iqamax", test_iqamax),
        ("qaxpy",  test_qaxpy),
        ("qscal",  test_qscal),
        ("qgemv",  test_qgemv),
        ("qger",   test_qger),
        ("qgemm",  test_qgemm),
        ("qsyrk",  test_qsyrk),
        ("qtrmm",  test_qtrmm),
        ("qtrsm",  test_qtrsm),
    ]
    for name, fn in tests:
        print(f"  running {name} ...", end=" ", flush=True)
        before = (PASSED, FAILED)
        fn()
        added_pass = PASSED - before[0]
        added_fail = FAILED - before[1]
        print(f"{added_pass} ok, {added_fail} fail")

    print()
    print(f"Total: {PASSED} passed, {FAILED} failed")
    if FAILED:
        print()
        print("Failures:")
        for line in FAILS:
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
