import numpy as np
from numpy.typing import NDArray

from ._core_impl import _demean_rs, _demean_within_rs


def _sanitize_krylov_and_preconditioner(
    krylov_method: str,
    preconditioner_type: str,
) -> tuple[str, str]:
    krylov_method = krylov_method.lower()
    preconditioner_type = preconditioner_type.lower()

    if krylov_method not in {"cg", "gmres"}:
        raise ValueError("`krylov_method` must be either 'cg' or 'gmres'.")
    if preconditioner_type not in {"additive", "multiplicative"}:
        raise ValueError(
            "`preconditioner_type` must be either 'additive' or 'multiplicative'."
        )
    if preconditioner_type == "multiplicative" and krylov_method != "gmres":
        raise ValueError("Multiplicative Schwarz requires `krylov_method='gmres'`.")

    return krylov_method, preconditioner_type


# Legacy: used by BACKENDS dict and _set_demeaner_backend.
# Remove once all callers use the typed demeaner= API.
def demean(
    x: NDArray[np.float64],
    flist: NDArray[np.uint64],
    weights: NDArray[np.float64],
    tol: float = 1e-08,
    maxiter: int = 100_000,
) -> tuple[NDArray, bool]:
    """
    Demean an array.

    Workhorse for demeaning an input array `x` based on the specified fixed
    effects and weights via the alternating projections algorithm.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of shape (n_samples, n_features). Needs to be of type float.
    flist : numpy.ndarray
        Array of shape (n_samples, n_factors) specifying the fixed effects.
        Needs to already be converted to integers.
    weights : numpy.ndarray
        Array of shape (n_samples,) specifying the weights.
    tol : float, optional
        Tolerance criterion for convergence. Defaults to 1e-08.
    maxiter : int, optional
        Maximum number of iterations. Defaults to 100_000.

    Returns
    -------
    tuple[numpy.ndarray, bool]
        A tuple containing the demeaned array of shape (n_samples, n_features)
        and a boolean indicating whether the algorithm converged successfully.
    """
    return _demean_rs(
        x.astype(np.float64, copy=False),
        flist.astype(np.uint64, copy=False),
        weights.astype(np.float64, copy=False),
        tol,
        maxiter,
    )


# Legacy: used by BACKENDS dict and _set_demeaner_backend.
# Remove once all callers use the typed demeaner= API.
def demean_within(
    x: NDArray[np.float64],
    flist: NDArray[np.uint32],
    weights: NDArray[np.float64],
    tol: float = 1e-06,
    maxiter: int = 1_000,
    krylov_method: str = "cg",
    gmres_restart: int = 30,
    preconditioner_type: str = "additive",
) -> tuple[NDArray, bool]:
    """Demean an array using the configurable `within` backend."""
    krylov_method, preconditioner_type = _sanitize_krylov_and_preconditioner(
        krylov_method,
        preconditioner_type,
    )

    if flist.ndim == 1 or flist.shape[1] == 1:
        return _demean_rs(
            x.astype(np.float64, copy=False),
            flist.astype(np.uint64, copy=False),
            weights.astype(np.float64, copy=False),
            tol,
            maxiter,
        )

    return _demean_within_rs(
        x.astype(np.float64, copy=False),
        np.asfortranarray(flist, dtype=np.uint32),
        weights.astype(np.float64, copy=False).reshape(-1),
        tol,
        maxiter,
        krylov_method,
        gmres_restart,
        preconditioner_type,
    )
