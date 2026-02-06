from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

class DemeanResult(TypedDict):
    """Result from the Rust demeaning function."""

    demeaned: NDArray[np.float64]
    fe_coefficients: NDArray[np.float64]
    success: bool

def _find_collinear_variables_rs(x: NDArray[np.float64], tol: float = 1e-10): ...
def _crv1_meat_loop_rs(
    scores: NDArray[np.float64],
    clustid: NDArray[np.uint64],
    cluster_col: NDArray[np.uint64],
) -> NDArray[np.float64]: ...
def _demean_rs(
    x: NDArray[np.float64],
    flist: NDArray[np.uint64],
    weights: NDArray[np.float64] | None = None,
    tol: float = 1e-08,
    maxiter: int = 100_000,
    reorder_fe: bool = False,
) -> DemeanResult: ...
def _count_fixef_fully_nested_all_rs(
    all_fixef_array: NDArray,
    cluster_colnames: NDArray,
    cluster_data: NDArray[np.uint64],
    fe_data: NDArray[np.uint64],
) -> tuple[np.ndarray, int]: ...
def _detect_singletons_rs(ids: NDArray[np.uint32]) -> NDArray[np.bool_]: ...
