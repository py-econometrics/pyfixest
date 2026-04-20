from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from pyfixest.demeaners import PreconditionerType, WithinKrylovMethod

from ._core_impl import _demean_rs, _demean_within_rs


def _prepare_within_flist(flist: NDArray[np.uint32]) -> NDArray[np.uint32]:
    flist_arr = np.asfortranarray(flist, dtype=np.uint32)
    if flist_arr.ndim == 1:
        flist_arr = flist_arr.reshape((-1, 1), order="F")
    return flist_arr


def _prepare_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(weights, dtype=np.float64).reshape(-1)


def _sanitize_krylov_and_preconditioner(
    krylov_method: WithinKrylovMethod,
    preconditioner_type: PreconditionerType,
) -> tuple[WithinKrylovMethod, PreconditionerType]:
    if not isinstance(krylov_method, str):
        raise TypeError("`krylov_method` must be a string.")
    if not isinstance(preconditioner_type, str):
        raise TypeError("`preconditioner_type` must be a string.")
    if krylov_method not in {"cg", "gmres"}:
        raise ValueError("`krylov_method` must be either 'cg' or 'gmres'.")
    if preconditioner_type not in {"additive", "multiplicative"}:
        raise ValueError(
            "`preconditioner_type` must be either 'additive' or 'multiplicative'."
        )
    if preconditioner_type == "multiplicative" and krylov_method != "gmres":
        raise ValueError("Multiplicative Schwarz requires `krylov_method='gmres'`.")

    return (
        cast(WithinKrylovMethod, krylov_method),
        cast(PreconditionerType, preconditioner_type),
    )


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

    Examples
    --------
    ```{python}
    import numpy as np
    import pyfixest as pf
    from pyfixest.utils.dgps import get_blw
    from pyfixest.estimation.internals.demean_ import demean
    from formulaic import model_matrix

    fml = "y ~ treat | state + year"

    data = get_blw()
    data.head()

    Y, rhs = model_matrix(fml, data)
    X = rhs[0].drop(columns="Intercept")
    fe = rhs[1].drop(columns="Intercept")
    YX = np.concatenate([Y, X], axis=1)

    # to numpy
    Y = Y.to_numpy()
    X = X.to_numpy()
    YX = np.concatenate([Y, X], axis=1)
    fe = fe.to_numpy().astype(int)  # demean requires fixed effects as ints!

    YX_demeaned, success = demean(YX, fe, weights = np.ones(YX.shape[0]))
    Y_demeaned = YX_demeaned[:, 0]
    X_demeaned = YX_demeaned[:, 1:]

    print(np.linalg.lstsq(X_demeaned, Y_demeaned, rcond=None)[0])
    print(pf.feols(fml, data).coef())
    ```
    """
    return _demean_rs(
        x.astype(np.float64, copy=False),
        flist.astype(np.uint64, copy=False),
        weights.astype(np.float64, copy=False),
        tol,
        maxiter,
    )


def demean_within(
    x: NDArray[np.float64],
    flist: NDArray[np.uint32],
    weights: NDArray[np.float64],
    tol: float = 1e-06,
    maxiter: int = 1_000,
    krylov_method: WithinKrylovMethod = "cg",
    gmres_restart: int = 30,
    preconditioner_type: PreconditionerType = "additive",
) -> tuple[NDArray, bool]:
    """Demean an array using the configurable `within` backend.

    Uses Krylov-based solvers with Schwarz preconditioning. Converges faster
    than alternating projections on weakly-connected or block-diagonal
    fixed-effect structures.

    For single fixed effects, falls back to alternating projections
    (``_demean_rs``) because the Schwarz preconditioner is designed for
    multi-way FE problems.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of shape (n_samples, n_features).
    flist : numpy.ndarray
        Array of shape (n_samples, n_factors) specifying the fixed effects
        (integer-encoded).
    weights : numpy.ndarray
        Array of shape (n_samples,) specifying the weights.
    tol : float, optional
        Convergence tolerance. Defaults to 1e-06.
    maxiter : int, optional
        Maximum number of Krylov iterations. Defaults to 1_000.
    krylov_method : {"cg", "gmres"}, optional
        Krylov solver to use. Defaults to "cg".
    gmres_restart : int, optional
        Restart parameter for GMRES. Defaults to 30.
    preconditioner_type : {"additive", "multiplicative"}, optional
        Schwarz preconditioner variant. Defaults to "additive".

    Returns
    -------
    tuple[numpy.ndarray, bool]
        Demeaned array and convergence flag.
    """
    krylov_method, preconditioner_type = _sanitize_krylov_and_preconditioner(
        krylov_method,
        preconditioner_type,
    )

    flist_arr = _prepare_within_flist(flist)
    if flist_arr.shape[1] == 1:
        return _demean_rs(
            x.astype(np.float64, copy=False),
            flist_arr.astype(np.uint64, copy=False),
            _prepare_weights(weights),
            tol,
            maxiter,
        )

    weights_arr = _prepare_weights(weights)
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape((-1, 1))

    return _demean_within_rs(
        x_arr,
        flist_arr,
        weights_arr,
        tol,
        maxiter,
        krylov_method,
        gmres_restart,
        preconditioner_type,
    )
