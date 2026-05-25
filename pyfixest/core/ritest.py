import numpy as np
from numpy.typing import NDArray

from ._core_impl import _run_ri_rs


def run_ri(
    resampled_d: NDArray[np.float64],
    y_demean: NDArray[np.float64],
    x_demean2: NDArray[np.float64],
    fval: NDArray | None,
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Run randomization inference via Frisch-Waugh-Lovell in Rust.

    Parameters
    ----------
    resampled_d : np.ndarray
        Pre-generated resampled treatment variables, shape (N, reps).
    y_demean : np.ndarray
        Demeaned dependent variable, shape (N,).
    x_demean2 : np.ndarray
        Demeaned design matrix without the treatment column, shape (N, K).
    fval : np.ndarray or None
        Integer-encoded fixed effects, shape (N, n_fe). None if no FE.
    weights : np.ndarray
        Sample weights, shape (N,).

    Returns
    -------
    np.ndarray
        Array of RI coefficients, shape (reps,).
    """
    return _run_ri_rs(
        np.ascontiguousarray(resampled_d, dtype=np.float64),
        np.ascontiguousarray(y_demean, dtype=np.float64),
        np.ascontiguousarray(x_demean2, dtype=np.float64),
        np.ascontiguousarray(fval, dtype=np.uint64) if fval is not None else None,
        np.ascontiguousarray(weights, dtype=np.float64),
    )
