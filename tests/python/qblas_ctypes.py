"""ctypes wrapper around libqblas via qblas_shim.so.  The shim re-exports
every entry point with Sleef_quad scalars passed by pointer, since
ctypes / libffi does not reliably handle 16-byte struct-by-value args."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
SLEEF_LIBDIR = REPO / ".sleef-prefix" / "lib"

def _load(libpath, name: str):
    p = Path(libpath)
    if not p.exists():
        raise FileNotFoundError(f"{name} not found at {p}")
    return ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)

# Load NEEDED chain in dependency order.
_tlfloat   = _load(os.environ.get("QBLAS_TEST_TLFLOAT",   SLEEF_LIBDIR / "libtlfloat.so.1"),   "libtlfloat")
_sleef     = _load(os.environ.get("QBLAS_TEST_SLEEF",     SLEEF_LIBDIR / "libsleef.so"),       "libsleef")
_sleefquad = _load(os.environ.get("QBLAS_TEST_SLEEFQUAD", SLEEF_LIBDIR / "libsleefquad.so"),   "libsleefquad")
_qblas     = _load(os.environ.get("QBLAS_TEST_QBLAS",     REPO / "build" / "src" / "libqblas.so"), "libqblas")
_shim      = _load(os.environ.get("QBLAS_TEST_SHIM",      REPO / "build" / "libqblas_shim.so"),    "libqblas_shim")

class Quad(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint64), ("y", ctypes.c_uint64)]

QUAD_DTYPE = np.dtype([("x", "<u8"), ("y", "<u8")])
assert QUAD_DTYPE.itemsize == 16

_shim.shim_d2q.argtypes = [ctypes.c_double, ctypes.POINTER(Quad)]
_shim.shim_d2q.restype  = None
_shim.shim_q2d.argtypes = [ctypes.POINTER(Quad)]
_shim.shim_q2d.restype  = ctypes.c_double

def d2q(d: float) -> Quad:
    q = Quad()
    _shim.shim_d2q(ctypes.c_double(d), ctypes.byref(q))
    return q

def q2d(q: Quad) -> float:
    return float(_shim.shim_q2d(ctypes.byref(q)))

def doubles_to_quads(arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr, dtype=np.float64).ravel()
    out = np.empty(arr.shape, dtype=QUAD_DTYPE)
    for i, x in enumerate(arr):
        q = d2q(float(x))
        out[i] = (q.x, q.y)
    return out

def quads_to_doubles(qs: np.ndarray) -> np.ndarray:
    qs = qs.reshape(-1)
    out = np.empty(qs.shape, dtype=np.float64)
    q = Quad()
    for i in range(qs.shape[0]):
        q.x = int(qs[i]["x"]); q.y = int(qs[i]["y"])
        out[i] = q2d(q)
    return out

QblasRowMajor = 101
QblasColMajor = 102
QblasNoTrans  = 111
QblasTrans    = 112
QblasUpper    = 121
QblasLower    = 122
QblasNonUnit  = 131
QblasUnit     = 132
QblasLeft     = 141
QblasRight    = 142

_qblas.qblas_get_dispatch_tier.restype = ctypes.c_char_p
def dispatch_tier() -> str:
    return _qblas.qblas_get_dispatch_tier().decode()

_qblas.qblas_get_num_threads.restype = ctypes.c_int
def num_threads() -> int:
    return int(_qblas.qblas_get_num_threads())

PQ = ctypes.POINTER(Quad)
VP = ctypes.c_void_p
INT = ctypes.c_int
SZ  = ctypes.c_size_t

_shim.shim_qdot.argtypes  = [INT, VP, INT, VP, INT, PQ]
_shim.shim_qdot.restype   = None
_shim.shim_qnrm2.argtypes = [INT, VP, INT, PQ]
_shim.shim_qnrm2.restype  = None
_shim.shim_qasum.argtypes = [INT, VP, INT, PQ]
_shim.shim_qasum.restype  = None
_shim.shim_iqamax.argtypes = [INT, VP, INT]
_shim.shim_iqamax.restype  = SZ
_shim.shim_qaxpy.argtypes = [INT, PQ, VP, INT, VP, INT]
_shim.shim_qaxpy.restype  = None
_shim.shim_qscal.argtypes = [INT, PQ, VP, INT]
_shim.shim_qscal.restype  = None
_shim.shim_qgemv.argtypes = [INT, INT, INT, INT, PQ, VP, INT, VP, INT, PQ, VP, INT]
_shim.shim_qgemv.restype  = None
_shim.shim_qger.argtypes  = [INT, INT, INT, PQ, VP, INT, VP, INT, VP, INT]
_shim.shim_qger.restype   = None
_shim.shim_qgemm.argtypes = [INT, INT, INT, INT, INT, INT, PQ, VP, INT, VP, INT, PQ, VP, INT]
_shim.shim_qgemm.restype  = None
_shim.shim_qsyrk.argtypes = [INT, INT, INT, INT, INT, PQ, VP, INT, PQ, VP, INT]
_shim.shim_qsyrk.restype  = None
_shim.shim_qtrmm.argtypes = [INT, INT, INT, INT, INT, INT, INT, PQ, VP, INT, VP, INT]
_shim.shim_qtrmm.restype  = None
_shim.shim_qtrsm.argtypes = [INT, INT, INT, INT, INT, INT, INT, PQ, VP, INT, VP, INT]
_shim.shim_qtrsm.restype  = None

def _quad_buf(arr_f64):
    buf = doubles_to_quads(arr_f64)
    return buf, buf.ctypes.data

def _q_ptr(d: float):
    q = d2q(d)
    return q, ctypes.byref(q)

def qdot(x, y) -> float:
    xb, xp = _quad_buf(x); yb, yp = _quad_buf(y)
    out = Quad()
    _shim.shim_qdot(x.size, xp, 1, yp, 1, ctypes.byref(out))
    return q2d(out)

def qnrm2(x) -> float:
    xb, xp = _quad_buf(x); out = Quad()
    _shim.shim_qnrm2(x.size, xp, 1, ctypes.byref(out))
    return q2d(out)

def qasum(x) -> float:
    xb, xp = _quad_buf(x); out = Quad()
    _shim.shim_qasum(x.size, xp, 1, ctypes.byref(out))
    return q2d(out)

def iqamax(x) -> int:
    xb, xp = _quad_buf(x)
    return int(_shim.shim_iqamax(x.size, xp, 1))

def qaxpy(alpha: float, x, y):
    n = x.size
    xb, xp = _quad_buf(x); yb, yp = _quad_buf(y)
    a, ap = _q_ptr(alpha)
    _shim.shim_qaxpy(n, ap, xp, 1, yp, 1)
    return quads_to_doubles(yb)

def qscal(alpha: float, x):
    xb, xp = _quad_buf(x); a, ap = _q_ptr(alpha)
    _shim.shim_qscal(x.size, ap, xp, 1)
    return quads_to_doubles(xb)

def qgemv(A, x, y, alpha=1.0, beta=0.0,
          trans=QblasNoTrans, layout=QblasRowMajor):
    A = np.ascontiguousarray(A, dtype=np.float64)
    m, n = A.shape; lda = n
    Ab, Ap = _quad_buf(A); xb, xp = _quad_buf(x); yb, yp = _quad_buf(y)
    a, ap = _q_ptr(alpha); b, bp = _q_ptr(beta)
    _shim.shim_qgemv(layout, trans, m, n, ap, Ap, lda, xp, 1, bp, yp, 1)
    return quads_to_doubles(yb)

def qger(A, x, y, alpha=1.0, layout=QblasRowMajor):
    A = np.ascontiguousarray(A, dtype=np.float64)
    m, n = A.shape; lda = n
    Ab, Ap = _quad_buf(A); xb, xp = _quad_buf(x); yb, yp = _quad_buf(y)
    a, ap = _q_ptr(alpha)
    _shim.shim_qger(layout, m, n, ap, xp, 1, yp, 1, Ap, lda)
    return quads_to_doubles(Ab).reshape(m, n)

def qgemm(A, B, C, alpha=1.0, beta=0.0,
          ta=QblasNoTrans, tb=QblasNoTrans, layout=QblasRowMajor):
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    if ta == QblasNoTrans: m, k = A.shape
    else:                  k, m = A.shape
    if tb == QblasNoTrans: kk, n = B.shape
    else:                  n, kk = B.shape
    assert k == kk, f"k mismatch: {k} vs {kk}"
    assert C.shape == (m, n)
    lda = A.shape[1]; ldb = B.shape[1]; ldc = n
    Ab, Ap = _quad_buf(A); Bb, Bp = _quad_buf(B); Cb, Cp = _quad_buf(C)
    a, ap = _q_ptr(alpha); b, bp = _q_ptr(beta)
    _shim.shim_qgemm(layout, ta, tb, m, n, k, ap, Ap, lda, Bp, ldb, bp, Cp, ldc)
    return quads_to_doubles(Cb).reshape(m, n)

def qsyrk(A, C, alpha=1.0, beta=0.0,
          uplo=QblasUpper, trans=QblasNoTrans, layout=QblasRowMajor):
    A = np.ascontiguousarray(A, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    if trans == QblasNoTrans: n, k = A.shape
    else:                     k, n = A.shape
    assert C.shape == (n, n)
    Ab, Ap = _quad_buf(A); Cb, Cp = _quad_buf(C)
    a, ap = _q_ptr(alpha); b, bp = _q_ptr(beta)
    _shim.shim_qsyrk(layout, uplo, trans, n, k, ap, Ap, A.shape[1], bp, Cp, n)
    return quads_to_doubles(Cb).reshape(n, n)

def qtrmm(A, B, alpha=1.0,
          side=QblasLeft, uplo=QblasLower, trans=QblasNoTrans, diag=QblasNonUnit,
          layout=QblasRowMajor):
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    m, n = B.shape; lda = A.shape[1]
    Ab, Ap = _quad_buf(A); Bb, Bp = _quad_buf(B)
    a, ap = _q_ptr(alpha)
    _shim.shim_qtrmm(layout, side, uplo, trans, diag, m, n, ap, Ap, lda, Bp, n)
    return quads_to_doubles(Bb).reshape(m, n)

def qtrsm(A, B, alpha=1.0,
          side=QblasLeft, uplo=QblasLower, trans=QblasNoTrans, diag=QblasNonUnit,
          layout=QblasRowMajor):
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    m, n = B.shape; lda = A.shape[1]
    Ab, Ap = _quad_buf(A); Bb, Bp = _quad_buf(B)
    a, ap = _q_ptr(alpha)
    _shim.shim_qtrsm(layout, side, uplo, trans, diag, m, n, ap, Ap, lda, Bp, n)
    return quads_to_doubles(Bb).reshape(m, n)
