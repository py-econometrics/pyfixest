from collections.abc import Callable
from dataclasses import replace
from importlib import import_module
from typing import cast

import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pyfixest.core.demean import demean_within
from pyfixest.demeaners import AnyDemeaner, LsmrDemeaner, MapDemeaner, WithinDemeaner
from pyfixest.estimation.internals.literals import DemeanerBackendOptions


def demean_model(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: pd.DataFrame | None,
    weights: np.ndarray | None,
    lookup_demeaned_data: dict[frozenset[int], pd.DataFrame],
    na_index: frozenset[int],
    demeaner: AnyDemeaner,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Demean a regression model.

    Demeans a single regression model via the configured demeaner backend.
    Prior to demeaning, the function checks if some of the variables have
    already been demeaned and uses values from the cache
    `lookup_demeaned_data` if possible. If the model has no fixed effects,
    the function does not demean the data.

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
    lookup_demeaned_data : dict[frozenset[int], pd.DataFrame]
        A dictionary with keys for each fixed effects combination and potentially
        values of demeaned data frames. The function checks this dictionary to
        see if some of the variables have already been demeaned.
    na_index : frozenset[int]
        A frozenset of indices of dropped rows. Used as a hashable cache key
        for demeaned variables.
    demeaner : AnyDemeaner
        Resolved typed demeaner configuration. Backend-specific runtime options
        are taken from this object.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple of the following elements:
        - Yd : pd.DataFrame
            A DataFrame of the demeaned dependent variable.
        - Xd : pd.DataFrame
            A DataFrame of the demeaned covariates.
    """
    YX = pd.concat([Y, X], axis=1)

    yx_names = YX.columns
    YX_array = YX.to_numpy()

    if YX_array.dtype != np.dtype("float64"):
        YX_array = YX_array.astype(np.float64)

    if weights is None:
        weights_array = np.ones(YX_array.shape[0], dtype=np.float64)
    elif weights.ndim > 1:
        weights_array = weights.flatten()
    else:
        weights_array = weights

    if fe is not None:
        YX_demeaned: pd.DataFrame
        fe_array = fe.to_numpy()

        # check if lookup dict has data for na_index
        YX_demeaned_old = lookup_demeaned_data.get(na_index)
        if YX_demeaned_old is not None:
            # get not yet demeaned covariates
            var_diff_names = list(set(yx_names) - set(YX_demeaned_old.columns))

            # if some variables still need to be demeaned
            if var_diff_names:
                yx_names_list = list(yx_names)
                var_diff_index = [yx_names_list.index(item) for item in var_diff_names]
                var_diff = YX_array[:, var_diff_index]
                if var_diff.ndim == 1:
                    var_diff = var_diff.reshape(len(var_diff), 1)

                YX_demean_new, success = dispatch_demean(
                    x=var_diff,
                    flist=fe_array,
                    weights=weights_array,
                    demeaner=demeaner,
                )
                if success is False:
                    raise ValueError(
                        f"Demeaning failed after {demeaner.fixef_maxiter} iterations."
                    )

                YX_demeaned = pd.DataFrame(
                    np.concatenate([YX_demeaned_old, YX_demean_new], axis=1)
                )

                if isinstance(var_diff_names, str):
                    var_diff_names = [var_diff_names]

                YX_demeaned.columns = pd.Index(
                    list(YX_demeaned_old.columns) + var_diff_names
                )

            else:
                # all variables already demeaned
                YX_demeaned = YX_demeaned_old[yx_names]

        else:
            YX_demeaned_array, success = dispatch_demean(
                x=YX_array,
                flist=fe_array,
                weights=weights_array,
                demeaner=demeaner,
            )
            if success is False:
                raise ValueError(
                    f"Demeaning failed after {demeaner.fixef_maxiter} iterations."
                )

            YX_demeaned = pd.DataFrame(YX_demeaned_array)
            YX_demeaned.columns = yx_names

        lookup_demeaned_data[na_index] = YX_demeaned

    else:
        YX_demeaned = pd.DataFrame(YX_array)
        YX_demeaned.columns = yx_names

    # get demeaned Y, X (if no fixef, equal to Y, X, I)
    Yd = YX_demeaned[Y.columns]
    Xd = YX_demeaned[X.columns]

    return Yd, Xd


def _override_demeaner_tol(
    demeaner: AnyDemeaner,
    *,
    tol: float | None = None,
) -> AnyDemeaner:
    """Override FE tolerance on a typed demeaner when needed. Used for IWLS acceleration."""
    if tol is None or tol == demeaner.fixef_tol:
        return demeaner
    if isinstance(demeaner, LsmrDemeaner):
        return replace(
            demeaner,
            fixef_tol=tol,
            solver_atol=tol,
            solver_btol=tol,
        )
    return replace(demeaner, fixef_tol=tol)


def dispatch_demean(
    x: np.ndarray,
    flist: np.ndarray,
    weights: np.ndarray,
    demeaner: AnyDemeaner,
) -> tuple[np.ndarray, bool]:
    """Demean an array using the configured backend for the resolved demeaner."""
    flist_uint = flist.astype(np.uintp, copy=False)

    if isinstance(demeaner, WithinDemeaner):
        return demean_within(
            x=x,
            flist=flist.astype(np.uint32, copy=False),
            weights=weights,
            tol=demeaner.fixef_tol,
            maxiter=demeaner.fixef_maxiter,
        )

    if isinstance(demeaner, LsmrDemeaner):
        if demeaner.backend == "torch":
            try:
                torch = import_module("torch")
                torch_demean_module = import_module(
                    "pyfixest.estimation.torch.demean_torch_"
                )
            except ImportError:
                return demean(
                    x=x,
                    flist=flist_uint,
                    weights=weights,
                    tol=demeaner.fixef_tol,
                    maxiter=demeaner.fixef_maxiter,
                )

            dtype = torch.float32 if demeaner.precision == "float32" else torch.float64
            tol = max(demeaner.solver_atol, demeaner.solver_btol)
            flist_uint64 = flist.astype(np.uint64, copy=False)

            if demeaner.device == "auto":
                demean_torch = cast(
                    Callable[..., tuple[np.ndarray, bool]],
                    torch_demean_module.demean_torch,
                )
                return demean_torch(
                    x=x,
                    flist=flist_uint64,
                    weights=weights,
                    tol=tol,
                    maxiter=demeaner.fixef_maxiter,
                    dtype=dtype,
                )

            demean_torch_on_device = cast(
                Callable[..., tuple[np.ndarray, bool]],
                torch_demean_module._demean_torch_on_device_impl,
            )
            return demean_torch_on_device(
                x=x,
                flist=flist_uint64,
                weights=weights,
                tol=tol,
                maxiter=demeaner.fixef_maxiter,
                device=torch.device(demeaner.device),
                dtype=dtype,
            )

        cupy_demean_module = import_module("pyfixest.estimation.cupy.demean_cupy_")
        fe_df = pd.DataFrame(
            flist_uint,
            columns=[f"f{i + 1}" for i in range(flist_uint.shape[1])],
            copy=False,
        )
        fe_sparse_matrix = cast(
            sp.spmatrix,
            cupy_demean_module.create_fe_sparse_matrix(fe_df),
        )
        cupy_demeaner = cupy_demean_module.CupyFWLDemeaner(
            device=demeaner.device,
            solver_atol=demeaner.solver_atol,
            solver_btol=demeaner.solver_btol,
            solver_maxiter=demeaner.fixef_maxiter,
            warn_on_cpu_fallback=demeaner.warn_on_cpu_fallback,
            dtype=np.float32 if demeaner.precision == "float32" else np.float64,
            use_preconditioner=demeaner.use_preconditioner,
        )
        return cupy_demeaner.demean(
            x=x,
            flist=flist_uint,
            weights=weights,
            fe_sparse_matrix=fe_sparse_matrix,
        )

    if isinstance(demeaner, MapDemeaner):
        backend = demeaner.backend
        if backend == "numba":
            demean_func = demean
        elif backend == "rust":
            from pyfixest.core.demean import demean as demean_rs

            demean_func = demean_rs
        elif backend == "jax":
            from pyfixest.estimation.jax.demean_jax_ import demean_jax

            demean_func = demean_jax
        else:
            raise ValueError(f"Unknown MapDemeaner backend: {backend!r}")

        return demean_func(
            x=x,
            flist=flist_uint,
            weights=weights,
            tol=demeaner.fixef_tol,
            maxiter=demeaner.fixef_maxiter,
        )

    raise TypeError(f"Unsupported demeaner type: {type(demeaner)!r}")


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


def _set_demeaner_backend(
    demeaner_backend: DemeanerBackendOptions,
) -> Callable:
    """Set the demeaning backend.

    Currently, we allow for a numba backend, rust backend, jax backend, and cupy backend.
    JAX and CuPy are expected to be faster on GPU for larger problems, but not necessarily
    faster than the numba and rust algos.

    Parameters
    ----------
    demeaner_backend : Literal["numba", "jax", "rust", "cupy", "cupy32", "cupy64"]
        The demeaning backend to use.

    Returns
    -------
    Callable
        The demeaning function.

    Raises
    ------
    ValueError
        If the demeaning backend is not supported.
    """
    from pyfixest.estimation.internals.backends import get_backend

    return get_backend(demeaner_backend)["demean"]
