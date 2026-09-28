import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import lsqr

from pyfixest.estimation.internals.literals import (
    SolverOptions,
)


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
