import numpy as np
import pytest

from pyfixest.estimation.solvers import solve_ols


def test_solve_ols_simple_2x2():
    # Test case 1: Simple 2x2 system
    tZX = np.array([[1, 2], [3, 4]])
    tZY = np.array([5, 6])
    solver = "np.linalg.lstsq"
    solution = solve_ols(tZX, tZY, solver)
    assert np.allclose(solution, np.array([-4.0, 4.5]))
    # Verify solution satisfies the system
    assert np.allclose(tZX @ solution, tZY)


def test_solve_ols_identity():
    # Test case 2: Identity matrix
    tZX = np.eye(2)
    tZY = np.array([1, 2])
    solver = "np.linalg.lstsq"
    assert np.allclose(solve_ols(tZX, tZY, solver), tZY)


@pytest.mark.parametrize(
    argnames="solver",
    argvalues=["np.linalg.lstsq", "np.linalg.solve", "scipy.sparse.linalg.lsqr", "jax"],
    ids=["np.linalg.lstsq", "np.linalg.solve", "scipy.sparse.linalg.lsqr", "jax"],
)
def test_solve_ols_different_solvers(solver):
    # Test case 3: Test different solvers give same result
    tZX = np.array([[1, 2], [3, 4]])
    tZY = np.array([5, 6])
    solution = solve_ols(tZX, tZY, solver)
    assert np.allclose(solution, np.array([-4.0, 4.5]))
    # Verify solution satisfies the system
    assert np.allclose(tZX @ solution, tZY)


def test_solve_ols_invalid_solver():
    # Test case 4: Invalid solver
    tZX = np.array([[1, 2], [3, 4]])
    tZY = np.array([5, 6])
    with pytest.raises(ValueError):
        solve_ols(tZX, tZY, "invalid_solver")
