import numpy as np
from numpy.typing import NDArray

from ._core_impl import _demean_rs


def demean(
    x: NDArray[np.float64],
    flist: NDArray[np.uint64],
    weights: NDArray[np.float64],
    tol: float = 1e-08,
    maxiter: int = 100_000,
    reorder_fe: bool = False,
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
    reorder_fe : bool, optional
        Whether to reorder fixed effects by size (largest first) before demeaning.
        This can improve convergence for some datasets. Defaults to False.

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
    from pyfixest.estimation.demean_ import demean
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
    # Check if weights are uniform (all equal) - use fast unweighted path
    weights_f64 = weights.astype(np.float64, copy=False)
    is_uniform = np.allclose(weights_f64, weights_f64.flat[0], atol=1e-10, rtol=0)

    result = _demean_rs(
        x.astype(np.float64, copy=False),
        flist.astype(np.uint64, copy=False),
        None if is_uniform else weights_f64,
        tol,
        maxiter,
        reorder_fe,
    )
    return result["demeaned"], result["success"]
