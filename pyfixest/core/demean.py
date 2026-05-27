import numpy as np
from numpy.typing import NDArray

from ._core_impl import _demean_rs, _demean_within_rs


def demean(
    x: NDArray[np.float64],
    flist: NDArray[np.uint64],
    weights: NDArray[np.float64],
    tol: float = 1e-06,
    maxiter: int = 10_000,
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
        Tolerance criterion for convergence. Defaults to 1e-06.
    maxiter : int, optional
        Maximum number of iterations. Defaults to 10_000.

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
    from pyfixest.estimation import demean
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
    krylov: str = "lsmr",
    preconditioner: str = "additive",
    local_size: int | None = None,
) -> tuple[NDArray, bool]:
    """
    Demean an array using LSMR and Schwarz preconditioning via `within`.

    Uses a modified LSMR solver with Schwarz preconditioning. Converges faster
    than alternating projections on weakly connected or block-diagonal
    fixed-effect structures.

    For single fixed effects, falls back to alternating projections (``_demean_rs``)
    because the ``within`` solver is designed for multi-way FE problems.

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
        Maximum number of LSMR iterations. Defaults to 1_000.
    krylov : {"lsmr"}, optional
        Solver used for multi-way fixed effects. Defaults to ``"lsmr"``.
    preconditioner : {"additive", "off"}, optional
        Schwarz preconditioner used for multi-way fixed effects. ``"off"``
        disables preconditioning.
        Defaults to ``"additive"``.
    local_size : int | None, optional
        Optional reorthogonalization window for LSMR.

    Returns
    -------
    tuple[numpy.ndarray, bool]
        Demeaned array and convergence flag.
    """
    if krylov != "lsmr":
        raise ValueError("`krylov` must be 'lsmr'.")
    if preconditioner not in ("additive", "off"):
        raise ValueError("`preconditioner` must be one of ('additive', 'off').")
    if local_size is not None:
        if isinstance(local_size, bool) or not isinstance(local_size, int):
            raise TypeError("`local_size` must be an int.")
        if local_size <= 0:
            raise ValueError("`local_size` must be strictly positive.")

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
        krylov,
        preconditioner,
        local_size,
    )
