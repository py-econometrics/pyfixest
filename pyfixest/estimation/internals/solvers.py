from collections.abc import Callable

import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import lsqr

from pyfixest.estimation.internals.literals import (
    SolverOptions,
)

_SolveFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

SOLVER_REGISTRY: dict[SolverOptions, _SolveFn] = {
    "np.linalg.lstsq": lambda tZX, tZY: np.linalg.lstsq(tZX, tZY, rcond=None)[
        0
    ].flatten(),
    "np.linalg.solve": lambda tZX, tZY: np.linalg.solve(tZX, tZY).flatten(),
    "scipy.linalg.solve": lambda tZX, tZY: solve(tZX, tZY, assume_a="pos").flatten(),
    "scipy.sparse.linalg.lsqr": lambda tZX, tZY: lsqr(tZX, tZY)[0].flatten(),
}


def solve_ols(
    tZX: np.ndarray,
    tZY: np.ndarray,
    solver: SolverOptions = "np.linalg.solve",
) -> np.ndarray:
    """
    Solve the ordinary least squares problem using the specified solver.

    Parameters
    ----------
    tZX (array-like): Z'X.
    tZY (array-like): Z'Y.
    solver (str): The solver to use. Supported solvers are "np.linalg.lstsq",
    "np.linalg.solve", "scipy.linalg.solve" and "scipy.sparse.linalg.lsqr".

    Returns
    -------
    array-like: The solution to the ordinary least squares problem.

    Raises
    ------
    ValueError: If the specified solver is not supported.
    """
    return SOLVER_REGISTRY[solver](tZX, tZY)
