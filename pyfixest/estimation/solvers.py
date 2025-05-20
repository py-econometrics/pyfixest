import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import lsqr

from pyfixest.estimation.literals import (
    SolverOptions,
)


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
    "np.linalg.solve", "scipy.linalg.solve", "scipy.sparse.linalg.lsqr" and "jax".

    Returns
    -------
    array-like: The solution to the ordinary least squares problem.

    Raises
    ------
    ValueError: If the specified solver is not supported.
    """
    if solver == "np.linalg.lstsq":
        return np.linalg.lstsq(tZX, tZY, rcond=None)[0].flatten()
    elif solver == "np.linalg.solve":
        return np.linalg.solve(tZX, tZY).flatten()
    elif solver == "scipy.linalg.solve":
        return solve(tZX, tZY, assume_a="pos").flatten()
    elif solver == "scipy.sparse.linalg.lsqr":
        return lsqr(tZX, tZY)[0].flatten()
    elif solver == "jax":
        import jax.numpy as jnp

        return jnp.linalg.lstsq(tZX, tZY, rcond=None)[0].flatten()
    else:
        raise ValueError(f"Solver {solver} not supported.")
