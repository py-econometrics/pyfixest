import warnings
from collections.abc import Callable
from dataclasses import replace
from importlib import import_module
from typing import cast

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pyfixest.core.demean import WithinPreconditioner, demean_within
from pyfixest.demeaners import (
    AnyDemeaner,
    LsmrBackend,
    LsmrDemeaner,
    LsmrPreconditioner,
    MapDemeaner,
)

_PRECONDITIONER_SUPPORT: dict[LsmrBackend, tuple[set[str], str]] = {
    "within": ({"none", "schwarz", "diag"}, "schwarz"),
    "torch": ({"diag"}, "diag"),
    "cupy": ({"none", "diag"}, "diag"),
}


def _resolve_preconditioner(backend: LsmrBackend, requested: LsmrPreconditioner) -> str:
    """Resolve ``preconditioner`` against the backend's supported set.

    ``"auto"`` always resolves silently to the backend's natural default.
    An explicit but unsupported value emits a ``UserWarning`` and is replaced
    with the natural default.
    """
    supported, default = _PRECONDITIONER_SUPPORT[backend]
    if requested == "auto" or requested in supported:
        return default if requested == "auto" else requested
    warnings.warn(
        (
            f"preconditioner={requested!r} is not supported by the {backend!r} "
            f"LSMR backend; falling back to {default!r}."
        ),
        UserWarning,
        stacklevel=3,
    )
    return default


def demean_model(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: pd.DataFrame | None,
    weights: np.ndarray | None,
    lookup_demeaned_data: dict[frozenset[int], pd.DataFrame],
    na_index: frozenset[int],
    demeaner: AnyDemeaner,
    preconditioner_store: list[WithinPreconditioner] | None = None,
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
                    weights=weights,
                    demeaner=demeaner,
                    preconditioner_store=preconditioner_store,
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
                weights=weights,
                demeaner=demeaner,
                preconditioner_store=preconditioner_store,
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
    if isinstance(demeaner, LsmrDemeaner):
        if tol is None or (tol == demeaner.fixef_atol and tol == demeaner.fixef_btol):
            return demeaner
        return replace(
            demeaner,
            fixef_atol=tol,
            fixef_btol=tol,
        )
    if tol is None or tol == demeaner.fixef_tol:
        return demeaner
    return replace(demeaner, fixef_tol=tol)


def dispatch_demean(
    x: np.ndarray,
    flist: np.ndarray,
    weights: np.ndarray | None,
    demeaner: AnyDemeaner,
    preconditioner_store: list[WithinPreconditioner] | None = None,
) -> tuple[np.ndarray, bool]:
    """Demean an array using the configured backend for the resolved demeaner.

    Parameters
    ----------
    preconditioner_store : list[WithinPreconditioner] or None, optional
        Optional mutable list used to cache the within preconditioner
        (``"schwarz"`` or ``"diag"``) across repeated calls on the same
        fixed-effect design (e.g. IWLS iterations or staged demeans). On the
        first call the built preconditioner is appended; subsequent calls
        requesting the same variant reuse it. For an explicit user-supplied
        ``WithinPreconditioner``, the same object is appended after a
        successful 2+way FE solve so fitted models can expose it via
        ``fit.preconditioners``; that append is for reporting / later manual
        reuse, not cache substitution.
    """
    flist_uint = flist.astype(np.uintp, copy=False)

    if isinstance(demeaner, LsmrDemeaner) and demeaner.backend == "within":
        preconditioner: str | WithinPreconditioner
        if isinstance(demeaner.preconditioner, WithinPreconditioner):
            preconditioner = demeaner.preconditioner
        else:
            preconditioner = _resolve_preconditioner("within", demeaner.preconditioner)

        # Capture the user's explicit preconditioner BEFORE any cache
        # substitution so we can preserve its Python identity below.
        explicit_preconditioner = (
            preconditioner if isinstance(preconditioner, WithinPreconditioner) else None
        )
        # If a previously built preconditioner is cached, reuse it when the
        # requested string matches the cached variant. Explicit user-supplied
        # WithinPreconditioner instances pass through unchanged.
        _string_to_variant = {"schwarz": "Additive", "diag": "Diagonal"}
        if preconditioner_store and isinstance(preconditioner, str):
            cached = preconditioner_store[-1]
            if _string_to_variant.get(preconditioner) == cached.variant:
                preconditioner = cached

        result, success, built = demean_within(
            x=x,
            flist=flist.astype(np.uint32, copy=False),
            weights=weights,
            tol=max(demeaner.fixef_atol, demeaner.fixef_btol),
            maxiter=demeaner.fixef_maxiter,
            local_size=demeaner.local_size,
            preconditioner=preconditioner,
        )
        # First-time store: prefer the user's original Python object (identity
        # preservation for `fit.preconditioners`), fall back to the freshly
        # built one. `built is None` signals the single-FE MAP fallback was
        # taken, in which case no preconditioner was actually used.
        if (
            preconditioner_store is not None
            and not preconditioner_store
            and built is not None
        ):
            preconditioner_store.append(explicit_preconditioner or built)
        return result, success

    if weights is None:
        weights = np.ones(x.shape[0], dtype=np.float64)

    if isinstance(demeaner, LsmrDemeaner):
        if demeaner.backend == "torch":
            # Torch LSMR always uses its built-in diagonal preconditioner.
            # Call resolver for its UserWarning side effect on incompatible
            # requests; the returned value is intentionally unused.
            _ = _resolve_preconditioner(
                "torch", cast(LsmrPreconditioner, demeaner.preconditioner)
            )
            try:
                torch = import_module("torch")
                torch_demean_module = import_module(
                    "pyfixest.estimation.torch.demean_torch_"
                )
            except ImportError:
                from pyfixest.core.demean import demean as demean_rs

                return demean_rs(
                    x=x,
                    flist=flist_uint,
                    weights=weights,
                    tol=max(demeaner.fixef_atol, demeaner.fixef_btol),
                    maxiter=demeaner.fixef_maxiter,
                )

            dtype = torch.float32 if demeaner.precision == "float32" else torch.float64
            tol = max(demeaner.fixef_atol, demeaner.fixef_btol)
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

        cupy_resolved = _resolve_preconditioner(
            "cupy", cast(LsmrPreconditioner, demeaner.preconditioner)
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
            fixef_atol=demeaner.fixef_atol,
            fixef_btol=demeaner.fixef_btol,
            fixef_maxiter=demeaner.fixef_maxiter,
            warn_on_cpu_fallback=demeaner.warn_on_cpu_fallback,
            dtype=np.float32 if demeaner.precision == "float32" else np.float64,
            preconditioner=cupy_resolved,
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
            demean_func = _get_numba_demean()
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


def _get_numba_demean() -> Callable[..., tuple[np.ndarray, bool]]:
    try:
        from pyfixest.estimation.numba.demean_nb import demean as demean_nb
    except ImportError as exc:
        raise ImportError(
            "The Numba MAP backend requires the optional `numba` extra. "
            "Install it with `pip install pyfixest[numba]`, or use the default "
            "`MapDemeaner(backend='rust')` backend."
        ) from exc

    return cast(Callable[..., tuple[np.ndarray, bool]], demean_nb)
