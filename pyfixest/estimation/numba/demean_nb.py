"""Provide the optional Numba demeaning reference implementation."""

import numba as nb
import numpy as np


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
    Demean an array via the Numba MAP implementation.

    Workhorse for demeaning an input array `x` based on fixed effects and weights
    via the alternating projections algorithm.
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
