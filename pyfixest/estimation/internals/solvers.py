from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve
from scipy.sparse.linalg import lsqr

from pyfixest.estimation.internals.literals import (
    SolverOptions,
)


@dataclass(frozen=True, slots=True)
class GramFactorization:
    """Fit-local weighted cross-product and its upper Cholesky factor.

    Both arrays have shape (k, k) in retained coefficient order, with
    ``upper.T @ upper == gram`` up to roundoff. The Gram matrix is X' W X
    on the exact row sample and within design used in the rank check.
    Arrays are owned by this value and made read-only; the value must not be
    reused after changing rows, columns, weights, or the FE projection.
    """

    gram: NDArray[np.float64]
    upper: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.gram.setflags(write=False)
        self.upper.setflags(write=False)


def solve_ols(
    tZX: np.ndarray,
    tZY: np.ndarray,
    solver: SolverOptions = "np.linalg.solve",
) -> np.ndarray:
    """
    Solve the normal equations tZX @ beta = tZY with the specified solver.

    Parameters
    ----------
    tZX (array-like): Z'X, shape (k, k).
    tZY (array-like): Z'Y, shape (k,), (k, 1), or (k, m) for m right-hand
    sides solved at once, as in the first stage of 2SLS.
    solver (str): The solver to use. Supported solvers are "np.linalg.lstsq",
    "np.linalg.solve", "scipy.linalg.solve" and "scipy.sparse.linalg.lsqr".

    Returns
    -------
    array-like: The solution, flat of shape (k,) for a single right-hand
    side and of shape (k, m) otherwise.

    Raises
    ------
    ValueError: If the specified solver is not supported.
    """
    if solver == "np.linalg.lstsq":
        beta = np.linalg.lstsq(tZX, tZY, rcond=None)[0]
    elif solver == "np.linalg.solve":
        beta = np.linalg.solve(tZX, tZY)
    elif solver == "scipy.linalg.solve":
        beta = solve(tZX, tZY, assume_a="pos")
    elif solver == "scipy.sparse.linalg.lsqr":
        # lsqr accepts one right-hand side at a time.
        rhs_columns = np.reshape(tZY, (tZY.shape[0], -1)).T
        beta = np.column_stack([lsqr(tZX, rhs)[0] for rhs in rhs_columns])
    else:
        raise ValueError(f"Solver {solver} not supported.")

    single_rhs = tZY.ndim == 1 or tZY.shape[1] == 1
    return beta.flatten() if single_rhs else beta.reshape(tZY.shape)
