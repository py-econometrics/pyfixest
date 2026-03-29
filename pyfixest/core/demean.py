from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._core_impl import (
    _build_within_preconditioner_rs,
    _demean_rs,
    _demean_within_rs,
)


def _prepare_within_flist(flist: NDArray[np.uint32]) -> NDArray[np.uint32]:
    flist_arr = np.asfortranarray(flist, dtype=np.uint32)
    if flist_arr.ndim == 1:
        flist_arr = flist_arr.reshape((-1, 1), order="F")
    return flist_arr


def _prepare_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(weights, dtype=np.float64).reshape(-1)


def _compute_factor_cardinalities(flist: NDArray[np.uint32]) -> tuple[int, ...]:
    # FE columns are dense 0..K-1 codes from `pd.factorize`, so cardinality is max + 1.
    return tuple((flist.max(axis=0) + 1).tolist())


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


@dataclass(frozen=True, slots=True)
class WithinPreconditioner:
    """Thin in-process wrapper around a built `within::FePreconditioner`."""

    n_obs: int
    n_factors: int
    factor_cardinalities: tuple[int, ...]
    preconditioner_type: str
    _preconditioner_handle: Any = field(repr=False)


def build_within_preconditioner(
    flist: NDArray[np.uint32],
    weights: NDArray[np.float64],
    preconditioner_type: str = "additive",
) -> WithinPreconditioner:
    """
    Build a reusable preconditioner for the `within` demeaner backend.

    Parameters
    ----------
    flist : numpy.ndarray
        Fixed-effects array of shape `(n_obs, n_factors)`.
    weights : numpy.ndarray
        Observation weights.
    preconditioner_type : {"additive", "multiplicative"}
        Schwarz preconditioner variant.

    Returns
    -------
    WithinPreconditioner
        In-process reusable preconditioner handle.

    Notes
    -----
    This is a pyfixest convenience wrapper around the upstream `within` solver
    flow: it constructs a temporary solver with upstream-compatible defaults and
    extracts its built `FePreconditioner` for reuse.
    """
    preconditioner_type = preconditioner_type.lower()
    if preconditioner_type not in {"additive", "multiplicative"}:
        raise ValueError(
            "`preconditioner_type` must be either 'additive' or 'multiplicative'."
        )

    flist_arr = _prepare_within_flist(flist)
    if flist_arr.shape[1] == 1:
        raise ValueError(
            "A reusable `within` preconditioner requires at least two fixed effects."
        )

    weights_arr = _prepare_weights(weights)
    handle = _build_within_preconditioner_rs(
        flist_arr,
        weights_arr,
        preconditioner_type,
    )
    return WithinPreconditioner(
        n_obs=flist_arr.shape[0],
        n_factors=flist_arr.shape[1],
        factor_cardinalities=_compute_factor_cardinalities(flist_arr),
        preconditioner_type=preconditioner_type,
        _preconditioner_handle=handle,
    )


def validate_within_preconditioner(
    preconditioner: WithinPreconditioner,
    flist: NDArray[np.uint32],
    *,
    preconditioner_type: str | None = None,
) -> None:
    """Validate compatibility and emit warnings for a reusable preconditioner.

    Raises ``ValueError`` for hard incompatibilities (number of factors,
    cardinalities, preconditioner type). Emits ``UserWarning`` for a differing
    observation count.
    """
    flist_arr = _prepare_within_flist(flist)

    if preconditioner.n_factors != flist_arr.shape[1]:
        raise ValueError(
            "The supplied within preconditioner is incompatible with the current "
            "fixed-effects structure: the number of fixed effects does not match."
        )
    if (
        preconditioner_type is not None
        and preconditioner.preconditioner_type != preconditioner_type
    ):
        raise ValueError(
            "The supplied within preconditioner uses "
            f"`preconditioner_type='{preconditioner.preconditioner_type}'`, but "
            f"`preconditioner_type='{preconditioner_type}'` was requested."
        )

    factor_cardinalities = _compute_factor_cardinalities(flist_arr)
    if preconditioner.factor_cardinalities != factor_cardinalities:
        raise ValueError(
            "The supplied within preconditioner is incompatible with the current "
            "fixed-effects structure: the fixed-effect cardinalities do not match."
        )

    # Soft warnings — reuse is allowed but may degrade effectiveness.
    if preconditioner.n_obs != flist_arr.shape[0]:
        warnings.warn(
            "The supplied within preconditioner was built on a different number "
            "of observations. Reuse is allowed, but effectiveness may degrade.",
            UserWarning,
            stacklevel=3,
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
    krylov_method: str = "cg",
    gmres_restart: int = 30,
    preconditioner_type: str = "additive",
    preconditioner: WithinPreconditioner | None = None,
) -> tuple[NDArray, bool]:
    """Demean an array using the configurable `within` backend."""
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

    preconditioner_handle = None
    if preconditioner is not None:
        validate_within_preconditioner(
            preconditioner,
            flist_arr,
            preconditioner_type=preconditioner_type,
        )
        preconditioner_handle = preconditioner._preconditioner_handle

    return _demean_within_rs(
        x_arr,
        flist_arr,
        weights_arr,
        tol,
        maxiter,
        krylov_method,
        gmres_restart,
        preconditioner_type,
        preconditioner_handle,
    )
