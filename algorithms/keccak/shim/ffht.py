"""Pure-Python stand-in for the FFHT C extension: in-place unnormalised Walsh-Hadamard transform."""
import numpy as np
from scipy.linalg import hadamard

_H = {}

def fht(x):
    n = x.shape[0]
    if n not in _H:
        _H[n] = hadamard(n).astype(x.dtype)
    x[:] = _H[n] @ x
