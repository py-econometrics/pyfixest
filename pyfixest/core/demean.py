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
    weights: NDArray[np.float64] | None = None,
    tol: float = 1e-08,
    maxiter: int = 1_000,
    local_size: int | None = None,
    preconditioner: str = "schwarz",
) -> tuple[NDArray, bool]:
    """
    Demean an array using modified LSMR via `within`.

    Uses `within`'s modified LSMR solver with additive Schwarz preconditioning.
    This backend is designed to be fast for sparse / poorly connected fixed effect
    structures, where the method of alternating projections (MAP) can struggle.

    For single fixed effects, falls back to alternating projections (``_demean_rs``)
    because the sparse iterative solver is designed for multi-way FE problems.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of shape (n_samples, n_features).
    flist : numpy.ndarray
        Array of shape (n_samples, n_factors) specifying the fixed effects
        (integer-encoded).
    weights : numpy.ndarray or None, optional
        Array of shape (n_samples,) specifying observation weights. ``None``
        (default) solves the unweighted problem and lets `within` skip the
        per-iteration weight multiplication.
    tol : float, optional
        Convergence tolerance. Defaults to 1e-08.
    maxiter : int, optional
        Maximum number of LSMR iterations. Defaults to 1_000.
    local_size : int or None, optional
        Numerical-stability knob for the LSMR solver. ``None`` (default) is
        usually fine and is the fastest setting. Try a small integer
        (typically ``5`` to ``20``) if the solver fails to converge on a
        numerically difficult problem — for example, fixed effects with
        very unequal group sizes, near-collinear factors, or extreme
        weights. Larger values are more numerically robust but use more
        memory: the solver keeps ``local_size`` extra working vectors,
        each the length of the total fixed-effect coefficient count,
        and twice that many when a preconditioner is active. Under the
        hood this enables windowed Gram-Schmidt reorthogonalization
        inside LSMR's bidiagonalization.
    preconditioner : {"schwarz", "none"}, optional
        Preconditioner choice for `within`'s LSMR solver. ``"schwarz"``
        (default) uses additive Schwarz preconditioning; ``"none"`` disables
        preconditioning.

    Returns
    -------
    tuple[numpy.ndarray, bool]
        Demeaned array and convergence flag.
    """
    flist_2d = flist.reshape(-1, 1) if flist.ndim == 1 else flist

    if flist_2d.shape[1] == 1:
        weights_for_map = (
            np.ones(x.shape[0], dtype=np.float64)
            if weights is None
            else weights.astype(np.float64, copy=False)
        )
        return _demean_rs(
            x.astype(np.float64, copy=False),
            flist_2d.astype(np.uint64, copy=False),
            weights_for_map,
            tol,
            maxiter,
        )
    weights_arg = (
        None if weights is None else weights.astype(np.float64, copy=False).reshape(-1)
    )
    return _demean_within_rs(
        x.astype(np.float64, copy=False),
        np.asfortranarray(flist_2d, dtype=np.uint32),
        weights_arg,
        tol,
        maxiter,
        local_size,
        preconditioner,
    )
