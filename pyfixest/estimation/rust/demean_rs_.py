import numpy as np
from pyfixest_core import demean_rs as _demean_rs

def demean_rs(x: np.ndarray,
              flist: np.ndarray,
              weights: np.ndarray,
              tol: float = 1e-08,
              maxiter: int = 100_000,
              ) -> tuple[np.ndarray, bool]:
    return _demean_rs(x, flist.astype(np.uint), weights, tol, maxiter)