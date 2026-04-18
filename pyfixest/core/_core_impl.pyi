import numpy as np
from numpy.typing import NDArray

class _WithinPreconditionerHandle: ...

def _find_collinear_variables_rs(x: NDArray[np.float64], tol: float = 1e-10): ...
def _crv1_meat_loop_rs(
    scores: NDArray[np.float64],
    clustid: NDArray[np.uint64],
    cluster_col: NDArray[np.uint64],
) -> NDArray[np.float64]: ...
def _crv1_vcov_loop_qreg_rs(
    x: NDArray[np.float64],
    clustid: NDArray[np.uint64],
    cluster_col: NDArray[np.uint64],
    q: float,
    u_hat: NDArray[np.float64],
    delta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def _demean_rs(
    x: NDArray[np.float64],
    flist: NDArray[np.uint64],
    weights: NDArray[np.float64],
    tol: float = 1e-08,
    maxiter: int = 100_000,
) -> tuple[np.ndarray, bool]: ...
def _demean_within_rs(
    x: NDArray[np.float64],
    flist: NDArray[np.uint32],
    weights: NDArray[np.float64],
    tol: float = 1e-06,
    maxiter: int = 1_000,
    krylov_method: str = "cg",
    gmres_restart: int = 30,
    preconditioner_type: str = "additive",
    preconditioner_handle: _WithinPreconditionerHandle | None = None,
) -> tuple[np.ndarray, bool]: ...
def _build_within_preconditioner_rs(
    flist: NDArray[np.uint32],
    weights: NDArray[np.float64],
    preconditioner_type: str = "additive",
) -> _WithinPreconditionerHandle: ...
def _serialize_within_preconditioner_rs(
    preconditioner_handle: _WithinPreconditionerHandle,
) -> bytes: ...
def _deserialize_within_preconditioner_rs(
    data: bytes,
) -> _WithinPreconditionerHandle: ...
def _count_fixef_fully_nested_all_rs(
    all_fixef_array: NDArray,
    cluster_colnames: NDArray,
    cluster_data: NDArray[np.uint64],
    fe_data: NDArray[np.uint64],
) -> tuple[np.ndarray, int]: ...
def _detect_singletons_rs(ids: NDArray[np.uint32]) -> NDArray[np.bool_]: ...
def _nw_meat_panel_rs(
    scores: NDArray[np.float64],
    starts: NDArray[np.uint64],
    counts: NDArray[np.uint64],
    lag: int,
) -> NDArray[np.float64]: ...
def _nw_meat_time_rs(
    scores: NDArray[np.float64],
    time_arr: NDArray[np.float64],
    lag: int,
) -> NDArray[np.float64]: ...
def _dk_meat_panel_rs(
    scores: NDArray[np.float64],
    idx: NDArray[np.uint64],
    lag: int,
) -> NDArray[np.float64]: ...
