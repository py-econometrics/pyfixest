import numpy as np
from numpy.typing import NDArray

def _find_collinear_variables_rs(x: NDArray[np.float64], tol: float = 1e-10): ...
def _crv1_meat_loop_rs(
    scores: NDArray[np.float64],
    clustid: NDArray[np.float64],
    cluster_col: NDArray[np.float64],
) -> NDArray: ...
def _demean_rs(
    x: NDArray[np.float64],
    flist: NDArray[np.uint],
    weights: NDArray[np.float64],
    tol: float = 1e-08,
    maxiter: int = 100_000,
) -> tuple[np.ndarray, bool]: ...
def _count_fixef_fully_nested_all_rs(
    all_fixef_array: NDArray,
    cluster_colnames: NDArray,
    cluster_data: NDArray[np.uint],
    fe_data: NDArray[np.uint],
) -> tuple[np.ndarray, int]: ...
