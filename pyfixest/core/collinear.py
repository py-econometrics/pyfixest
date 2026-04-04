import numpy as np
from numpy.typing import NDArray

from ._core_impl import _find_collinear_variables_rs


def find_collinear_variables(
    x: NDArray[np.float64], tol: float = 1e-10
) -> tuple[np.ndarray, int, bool]:
    """Coerce the Gram matrix into a NumPy float64 array before calling Rust."""
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError("`x` must be a 2D array.")
    return _find_collinear_variables_rs(x_arr, tol)
