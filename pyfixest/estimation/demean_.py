from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numba as nb
import numpy as np
import pandas as pd
from jax import config


def demean_model(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: Optional[pd.DataFrame],
    weights: Optional[np.ndarray],
    lookup_demeaned_data: dict[str, Any],
    na_index_str: str,
    fixef_tol: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Demean a regression model.

    Demeans a single regression model via the alternating projections algorithm
    (see `demean` function). Prior to demeaning, the function checks if some of
    the variables have already been demeaned and uses values from the cache
    `lookup_demeaned_data` if possible. If the model has no fixed effects, the
    function does not demean the data.

    Parameters
    ----------
    Y : pandas.DataFrame
        A DataFrame of the dependent variable.
    X : pandas.DataFrame
        A DataFrame of the covariates.
    fe : pandas.DataFrame or None
        A DataFrame of the fixed effects. None if no fixed effects specified.
    weights : numpy.ndarray or None
        A numpy array of weights. None if no weights.
    lookup_demeaned_data : dict[str, Any]
        A dictionary with keys for each fixed effects combination and potentially
        values of demeaned data frames. The function checks this dictionary to
        see if some of the variables have already been demeaned.
    na_index_str : str
        A string with indices of dropped columns. Used for caching of demeaned
        variables.
    fixef_tol: float
        The tolerance for the demeaning algorithm.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]
        A tuple of the following elements:
        - Yd : pd.DataFrame
            A DataFrame of the demeaned dependent variable.
        - Xd : pd.DataFrame
            A DataFrame of the demeaned covariates.
        - Id : pd.DataFrame or None
            A DataFrame of the demeaned Instruments. None if no IV.
    """
    YX = pd.concat([Y, X], axis=1)

    yx_names = YX.columns
    YX_array = YX.to_numpy()

    if YX_array.dtype != np.dtype("float64"):
        YX_array = YX_array.astype(np.float64)

    if weights is not None and weights.ndim > 1:
        weights = weights.flatten()

    if fe is not None:
        fe_array = fe.to_numpy()
        # check if looked dict has data for na_index
        if lookup_demeaned_data.get(na_index_str) is not None:
            # get data out of lookup table: list of [algo, data]
            value = lookup_demeaned_data.get(na_index_str)
            if value is not None:
                try:
                    _, YX_demeaned_old = value
                except ValueError:
                    print("Error: Expected the value to be iterable with two elements.")
            else:
                pass

            # get not yet demeaned covariates
            var_diff_names = list(set(yx_names) - set(YX_demeaned_old.columns))

            # if some variables still need to be demeaned
            if var_diff_names:
                # var_diff_names = var_diff_names

                yx_names_list = list(yx_names)
                var_diff_index = [yx_names_list.index(item) for item in var_diff_names]
                # var_diff_index = list(yx_names).index(var_diff_names)
                var_diff = YX_array[:, var_diff_index]
                if var_diff.ndim == 1:
                    var_diff = var_diff.reshape(len(var_diff), 1)

                YX_demean_new, success = demean(
                    x=var_diff, flist=fe_array, weights=weights, tol=fixef_tol
                )
                if success is False:
                    raise ValueError("Demeaning failed after 100_000 iterations.")

                YX_demeaned = pd.DataFrame(YX_demean_new)
                YX_demeaned = np.concatenate([YX_demeaned_old, YX_demean_new], axis=1)
                YX_demeaned = pd.DataFrame(YX_demeaned)

                # check if var_diff_names is a list
                if isinstance(var_diff_names, str):
                    var_diff_names = [var_diff_names]

                YX_demeaned.columns = pd.Index(
                    list(YX_demeaned_old.columns) + var_diff_names
                )

            else:
                # all variables already demeaned
                YX_demeaned = YX_demeaned_old[yx_names]

        else:
            YX_demeaned, success = demean(
                x=YX_array, flist=fe_array, weights=weights, tol=fixef_tol
            )
            if success is False:
                raise ValueError("Demeaning failed after 100_000 iterations.")

            YX_demeaned = pd.DataFrame(YX_demeaned)
            YX_demeaned.columns = yx_names

        lookup_demeaned_data[na_index_str] = [None, YX_demeaned]

    else:
        # nothing to demean here
        pass

        YX_demeaned = pd.DataFrame(YX_array)
        YX_demeaned.columns = yx_names

    # get demeaned Y, X (if no fixef, equal to Y, X, I)
    Yd = YX_demeaned[Y.columns]
    Xd = YX_demeaned[X.columns]

    return Yd, Xd


@nb.njit
def _sad_converged(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    for i in range(a.size):
        if np.abs(a[i] - b[i]) >= tol:
            return False
    return True


@nb.njit(locals=dict(id=nb.uint32))
def _subtract_weighted_group_mean(
    x: np.ndarray,
    sample_weights: np.ndarray,
    group_ids: np.ndarray,
    group_weights: np.ndarray,
    _group_weighted_sums: np.ndarray,
) -> None:
    _group_weighted_sums[:] = 0

    for i in range(x.size):
        id = group_ids[i]
        _group_weighted_sums[id] += sample_weights[i] * x[i]

    for i in range(x.size):
        id = group_ids[i]
        x[i] -= _group_weighted_sums[id] / group_weights[id]


@nb.njit
def _calc_group_weights(
    sample_weights: np.ndarray, group_ids: np.ndarray, n_groups: np.ndarray
):
    n_samples, n_factors = group_ids.shape
    dtype = sample_weights.dtype
    group_weights = np.zeros((n_factors, n_groups), dtype=dtype).T

    for j in range(n_factors):
        for i in range(n_samples):
            id = group_ids[i, j]
            group_weights[id, j] += sample_weights[i]

    return group_weights


@nb.njit(parallel=True)
def demean(
    x: np.ndarray,
    flist: np.ndarray,
    weights: np.ndarray,
    tol: float = 1e-08,
    maxiter: int = 100_000,
) -> tuple[np.ndarray, bool]:
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
    n_samples, n_features = x.shape
    n_factors = flist.shape[1]

    if x.flags.f_contiguous:
        res = np.empty((n_features, n_samples), dtype=x.dtype).T
    else:
        res = np.empty((n_samples, n_features), dtype=x.dtype)

    n_threads = nb.get_num_threads()

    n_groups = flist.max() + 1
    group_weights = _calc_group_weights(weights, flist, n_groups)
    _group_weighted_sums = np.empty((n_threads, n_groups), dtype=x.dtype)

    x_curr = np.empty((n_threads, n_samples), dtype=x.dtype)
    x_prev = np.empty((n_threads, n_samples), dtype=x.dtype)

    not_converged = 0
    for k in nb.prange(n_features):
        tid = nb.get_thread_id()

        xk_curr = x_curr[tid, :]
        xk_prev = x_prev[tid, :]
        for i in range(n_samples):
            xk_curr[i] = x[i, k]
            xk_prev[i] = x[i, k] - 1.0

        for _ in range(maxiter):
            for j in range(n_factors):
                _subtract_weighted_group_mean(
                    xk_curr,
                    weights,
                    flist[:, j],
                    group_weights[:, j],
                    _group_weighted_sums[tid, :],
                )
            if _sad_converged(xk_curr, xk_prev, tol):
                break

            xk_prev[:] = xk_curr[:]
        else:
            not_converged += 1

        res[:, k] = xk_curr[:]

    success = not not_converged
    return (res, success)


@partial(jax.jit, static_argnames=("n_groups", "tol", "maxiter"))
def _demean_jax_impl(
    x: jnp.ndarray,
    flist: jnp.ndarray,
    weights: jnp.ndarray,
    n_groups: int,
    tol: float,
    maxiter: int,
) -> tuple[jnp.ndarray, bool]:
    """JIT-compiled implementation of demeaning."""
    n_factors = flist.shape[1]

    @jax.jit
    def _apply_factor(carry, j):
        """Process a single factor."""
        x = carry
        factor_ids = flist[:, j]
        wx = x * weights[:, None]

        # Compute group weights and weighted sums
        group_weights = jnp.bincount(factor_ids, weights=weights, length=n_groups)
        group_sums = jax.vmap(
            lambda col: jnp.bincount(factor_ids, weights=col, length=n_groups)
        )(wx.T).T

        # Compute and subtract means
        means = group_sums / group_weights[:, None]
        return x - means[factor_ids], None

    @jax.jit
    def _demean_step(x_curr):
        """Single demeaning step for all factors."""
        # Process all factors using scan
        result, _ = jax.lax.scan(_apply_factor, x_curr, jnp.arange(n_factors))
        return result

    @jax.jit
    def _body_fun(state):
        """Body function for while_loop."""
        i, x_curr, x_prev, converged = state
        x_new = _demean_step(x_curr)
        max_diff = jnp.max(jnp.abs(x_new - x_curr))
        has_converged = max_diff < tol
        return i + 1, x_new, x_curr, has_converged

    @jax.jit
    def _cond_fun(state):
        """Condition function for while_loop."""
        i, _, _, converged = state
        return jnp.logical_and(i < maxiter, jnp.logical_not(converged))

    # Run the iteration loop using while_loop
    init_state = (0, x, x - 1.0, False)
    final_i, final_x, _, converged = jax.lax.while_loop(
        _cond_fun, _body_fun, init_state
    )

    return final_x, converged


def demean_jax(
    x: np.ndarray,
    flist: np.ndarray,
    weights: np.ndarray,
    tol: float = 1e-08,
    maxiter: int = 100_000,
) -> tuple[np.ndarray, bool]:
    """Fast and reliable JAX implementation with static shapes."""
    # Enable float64 precision
    config.update("jax_enable_x64", True)

    # Compute n_groups before JIT
    n_groups = int(np.max(flist) + 1)

    # Convert inputs to JAX arrays
    x_jax = jnp.asarray(x, dtype=jnp.float64)
    flist_jax = jnp.asarray(flist, dtype=jnp.int32)
    weights_jax = jnp.asarray(weights, dtype=jnp.float64)

    # Call the JIT-compiled implementation
    result_jax, converged = _demean_jax_impl(
        x_jax, flist_jax, weights_jax, n_groups, tol, maxiter
    )
    return np.array(result_jax), converged
