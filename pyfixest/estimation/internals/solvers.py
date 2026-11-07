from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from scipy.sparse import spmatrix
from scipy.sparse.linalg import lsqr

from pyfixest.errors import NonConvergenceError
from pyfixest.estimation.internals.literals import (
    SolverOptions,
)


def solve_lsqr(
    *,
    A: np.ndarray | spmatrix,
    b: np.ndarray,
    task: str,
    atol: float = 1e-6,
    btol: float = 1e-6,
) -> np.ndarray:
    """
    Solve min ||b - A x|| with LSQR and raise if it stops before converging.

    Parameters
    ----------
    A (array-like or sparse matrix): The design matrix, shape (n, k).
    b (array-like): The right-hand side, shape (n,).
    task (str): What the solve computes, used in the error message.
    atol, btol (float): Stopping tolerances for scipy.sparse.linalg.lsqr().

    Returns
    -------
    array-like: The solution x, shape (k,).

    Raises
    ------
    NonConvergenceError: If LSQR stops before meeting atol and btol.
    """
    x, istop, itn = lsqr(A, b, atol=atol, btol=btol)[:3]
    # LSQR stopping reasons (istop), from SciPy's lsqr message table:
    #   0  x = 0 is the exact solution
    #   1  Ax - b is small enough, given atol and btol
    #   2  the least-squares solution is good enough, given atol
    #   3  the estimate of cond(Abar) has exceeded conlim
    #   4  Ax - b is small enough for this machine
    #   5  the least-squares solution is good enough for this machine
    #   6  cond(Abar) seems to be too large for this machine
    #   7  the iteration limit has been reached
    # Codes 3, 6, and 7 stop before the accuracy tests are met.
    # https://github.com/scipy/scipy/blob/v1.17.1/scipy/sparse/linalg/_isolve/lsqr.py#L334-L341
    if istop in (3, 6):
        raise NonConvergenceError(
            f"{task} did not converge: LSQR stopped after {itn} iterations "
            f"because the problem is too ill-conditioned (istop={istop}), "
            f"before meeting atol={atol} and btol={btol}."
        )
    if istop == 7:
        raise NonConvergenceError(
            f"{task} did not converge: LSQR reached its iteration limit "
            f"({itn} iterations) before meeting atol={atol} and btol={btol}."
        )
    return x


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
        beta = np.column_stack(
            [
                solve_lsqr(A=tZX, b=rhs, task="The LSQR coefficient solve")
                for rhs in rhs_columns
            ]
        )
    else:
        raise ValueError(f"Solver {solver} not supported.")

    single_rhs = tZY.ndim == 1 or tZY.shape[1] == 1
    return beta.flatten() if single_rhs else beta.reshape(tZY.shape)
