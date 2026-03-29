from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib import import_module
from typing import cast

import numba as nb
import numpy as np
import pandas as pd

from pyfixest.core.demean import (
    WithinPreconditioner,
    build_within_preconditioner,
    demean_within,
)
from pyfixest.demeaners import LsmrDemeaner, MapDemeaner, WithinDemeaner
from pyfixest.estimation.internals.demeaner_options import ResolvedDemeaner
from pyfixest.estimation.internals.literals import DemeanerBackendOptions


@dataclass(slots=True)
class DemeanedDataCacheEntry:
    """Cached demeaned data and any reusable within preconditioner."""

    demeaned: pd.DataFrame
    preconditioner: WithinPreconditioner | None = None


def demean_model(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: pd.DataFrame | None,
    weights: np.ndarray | None,
    lookup_demeaned_data: dict[frozenset[int], DemeanedDataCacheEntry],
    na_index: frozenset[int],
    demeaner: ResolvedDemeaner,
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
    na_index : frozenset[int]
        A frozenset of indices of dropped rows. Used as a hashable cache key
        for demeaned variables.
    demeaner : ResolvedDemeaner
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
        # check if looked dict has data for na_index
        if lookup_demeaned_data.get(na_index) is not None:
            cache_entry = lookup_demeaned_data[na_index]
            YX_demeaned_old = cache_entry.demeaned

            # get not yet demeaned covariates
            var_diff_names = list(set(yx_names) - set(YX_demeaned_old.columns))

            # if some variables still need to be demeaned
            if var_diff_names:
                yx_names_list = list(yx_names)
                var_diff_index = [yx_names_list.index(item) for item in var_diff_names]
                var_diff = YX_array[:, var_diff_index]
                if var_diff.ndim == 1:
                    var_diff = var_diff.reshape(len(var_diff), 1)

                effective_demeaner, _ = _prepare_within_preconditioner(
                    flist=fe_array,
                    weights=weights_array,
                    demeaner=demeaner,
                    preconditioner=cache_entry.preconditioner,
                )
                YX_demean_new, success = dispatch_demean(
                    x=var_diff,
                    flist=fe_array,
                    weights=weights_array,
                    demeaner=effective_demeaner,
                )
                if success is False:
                    raise ValueError(
                        f"Demeaning failed after {demeaner.fixef_maxiter} iterations."
                    )

                YX_demeaned = pd.DataFrame(
                    np.concatenate([YX_demeaned_old, YX_demean_new], axis=1)
                )

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
            effective_demeaner, preconditioner = _prepare_within_preconditioner(
                flist=fe_array,
                weights=weights_array,
                demeaner=demeaner,
            )
            YX_demeaned_array, success = dispatch_demean(
                x=YX_array,
                flist=fe_array,
                weights=weights_array,
                demeaner=effective_demeaner,
            )
            if success is False:
                raise ValueError(
                    f"Demeaning failed after {demeaner.fixef_maxiter} iterations."
                )

            YX_demeaned = pd.DataFrame(YX_demeaned_array)
            YX_demeaned.columns = yx_names
            lookup_demeaned_data[na_index] = DemeanedDataCacheEntry(
                demeaned=YX_demeaned,
                preconditioner=preconditioner,
            )

        if na_index not in lookup_demeaned_data:
            lookup_demeaned_data[na_index] = DemeanedDataCacheEntry(
                demeaned=YX_demeaned
            )

    else:
        # nothing to demean here
        pass

        YX_demeaned = pd.DataFrame(YX_array)
        YX_demeaned.columns = yx_names

    # get demeaned Y, X (if no fixef, equal to Y, X, I)
    Yd = YX_demeaned[Y.columns]
    Xd = YX_demeaned[X.columns]

    return Yd, Xd


def _prepare_within_preconditioner(
    flist: np.ndarray,
    weights: np.ndarray,
    demeaner: ResolvedDemeaner,
    preconditioner: WithinPreconditioner | None = None,
    *,
    refresh_preconditioner: bool = False,
) -> tuple[ResolvedDemeaner, WithinPreconditioner | None]:
    """
    Prepare the effective demeaner and reusable within preconditioner.

    Only `WithinDemeaner` participates in preconditioner reuse. Other demeaners
    are returned unchanged with no reusable preconditioner.
    """
    if not isinstance(demeaner, WithinDemeaner):
        return demeaner, None
    flist_uint32 = flist.astype(np.uint32, copy=False)
    if flist_uint32.ndim == 1 or flist_uint32.shape[1] <= 1:
        return demeaner, None

    if demeaner.preconditioner is not None and not refresh_preconditioner:
        return demeaner, demeaner.preconditioner
    if preconditioner is not None and not refresh_preconditioner:
        return replace(demeaner, preconditioner=preconditioner), preconditioner

    built_preconditioner = build_within_preconditioner(
        flist=flist_uint32,
        weights=weights,
        preconditioner_type=demeaner.preconditioner_type,
    )
    return replace(demeaner, preconditioner=built_preconditioner), built_preconditioner


def _override_demeaner_tol(
    demeaner: ResolvedDemeaner,
    *,
    tol: float | None = None,
) -> ResolvedDemeaner:
    """Override FE tolerance on a typed demeaner when needed. Used for IWLS acceleration."""
    if tol is None or tol == demeaner.fixef_tol:
        return demeaner
    return replace(demeaner, fixef_tol=tol)


def dispatch_demean(
    x: np.ndarray,
    flist: np.ndarray,
    weights: np.ndarray,
    demeaner: ResolvedDemeaner,
) -> tuple[np.ndarray, bool]:
    """Demean an array using the configured backend for the resolved demeaner."""
    flist_uint = flist.astype(np.uintp, copy=False)

    if isinstance(demeaner, WithinDemeaner):
        return demean_within(
            x=x,
            flist=flist_uint.astype(np.uint32, copy=False),
            weights=weights,
            tol=demeaner.fixef_tol,
            maxiter=demeaner.fixef_maxiter,
            krylov_method=demeaner.krylov_method,
            gmres_restart=demeaner.gmres_restart,
            preconditioner_type=demeaner.preconditioner_type,
            preconditioner=demeaner.preconditioner,
        )

    if isinstance(demeaner, LsmrDemeaner):
        if demeaner.use_gpu is False:
            demean_scipy_configured = cast(
                Callable[..., tuple[np.ndarray, bool]],
                import_module(
                    "pyfixest.estimation.cupy.demean_cupy_"
                ).demean_scipy_configured,
            )
            return demean_scipy_configured(
                x=x,
                flist=flist_uint,
                weights=weights,
                solver_atol=demeaner.solver_atol,
                solver_btol=demeaner.solver_btol,
                solver_maxiter=demeaner.solver_maxiter,
                use_preconditioner=demeaner.use_preconditioner,
            )

        demean_cupy_configured = cast(
            Callable[..., tuple[np.ndarray, bool]],
            import_module(
                "pyfixest.estimation.cupy.demean_cupy_"
            ).demean_cupy_configured,
        )
        return demean_cupy_configured(
            x=x,
            flist=flist_uint,
            weights=weights,
            use_gpu=demeaner.use_gpu,
            solver_atol=demeaner.solver_atol,
            solver_btol=demeaner.solver_btol,
            solver_maxiter=demeaner.solver_maxiter,
            warn_on_cpu_fallback=demeaner.warn_on_cpu_fallback,
            dtype=np.float32 if demeaner.precision == "float32" else np.float64,
            use_preconditioner=demeaner.use_preconditioner,
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


# Legacy: used by the old demeaner_backend= string API.
# Remove once all callers use the typed demeaner= API.
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
    if demeaner_backend == "rust":
        from pyfixest.core.demean import demean as demean_rs

        return demean_rs
    elif demeaner_backend == "rust-cg":
        from pyfixest.core.demean import demean_within

        return demean_within
    elif demeaner_backend == "numba":
        return demean
    elif demeaner_backend == "jax":
        from pyfixest.estimation.jax.demean_jax_ import demean_jax

        return demean_jax
    elif demeaner_backend in ["cupy", "cupy64"]:
        from pyfixest.estimation.cupy.demean_cupy_ import demean_cupy64

        return demean_cupy64
    elif demeaner_backend == "cupy32":
        from pyfixest.estimation.cupy.demean_cupy_ import demean_cupy32

        return demean_cupy32
    else:
        raise ValueError(f"Invalid demeaner backend: {demeaner_backend}")
