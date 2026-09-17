import numpy as np
import pytest

import pyfixest as pf
from pyfixest.estimation.internals.solvers import solve_ols


def test_solve_ols_simple_2x2():
    # Test case 1: Simple 2x2 system
    tZX = np.array([[4, 2], [2, 3]])
    tZY = np.array([10, 8])
    solver = "scipy.linalg.solve"
    solution = solve_ols(tZX, tZY, solver)
    assert np.allclose(solution, np.array([1.75, 1.5]))
    # Verify solution satisfies the system
    assert np.allclose(tZX @ solution, tZY)


def test_solve_ols_identity():
    # Test case 2: Identity matrix
    tZX = np.eye(2)
    tZY = np.array([1, 2])
    solver = "scipy.linalg.solve"
    assert np.allclose(solve_ols(tZX, tZY, solver), tZY)


@pytest.mark.parametrize(
    argnames="solver",
    argvalues=[
        "scipy.linalg.solve",
        "np.linalg.lstsq",
        "np.linalg.solve",
        "scipy.sparse.linalg.lsqr",
    ],
    ids=[
        "scipy.linalg.solve",
        "np.linalg.lstsq",
        "np.linalg.solve",
        "scipy.sparse.linalg.lsqr",
    ],
)
def test_solve_ols_different_solvers(solver):
    # Test case 3: Test different solvers give same result
    tZX = np.array([[4, 2], [2, 3]])
    tZY = np.array([10, 8])
    solution = solve_ols(tZX, tZY, solver)
    assert np.allclose(solution, np.array([1.75, 1.5]))
    # Verify solution satisfies the system
    assert np.allclose(tZX @ solution, tZY)


SOLVERS = [
    "scipy.linalg.solve",
    "np.linalg.lstsq",
    "np.linalg.solve",
    "scipy.sparse.linalg.lsqr",
]


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVERS)
def test_solve_ols_single_column_rhs_is_flat(solver):
    # A (k, 1) right-hand side keeps the legacy flat solution.
    tZX = np.array([[4.0, 2.0], [2.0, 3.0]])
    tZY = np.array([[10.0], [8.0]])
    solution = solve_ols(tZX, tZY, solver)
    assert solution.shape == (2,)
    assert np.allclose(solution, np.array([1.75, 1.5]))


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVERS)
def test_solve_ols_matrix_rhs_solves_each_column(solver):
    # Several right-hand sides at once, as in the 2SLS first stage.
    rng = np.random.default_rng(7)
    A = rng.standard_normal((5, 3))
    tZX = A.T @ A
    tZY = rng.standard_normal((3, 4))
    solution = solve_ols(tZX, tZY, solver)
    assert solution.shape == (3, 4)
    expected = np.column_stack(
        [solve_ols(tZX, tZY[:, j], solver) for j in range(tZY.shape[1])]
    )
    assert np.allclose(solution, expected)
    assert np.allclose(tZX @ solution, tZY)


@pytest.mark.parametrize("solver", SOLVERS[1:], ids=SOLVERS[1:])
@pytest.mark.parametrize("weights", [None, "weights"])
def test_iv_solvers_agree(solver, weights):
    # Both 2SLS stages route through the solver option, so every solver must
    # reproduce the default's coefficients and covariance.
    data = pf.get_data()
    fml = "Y ~ X2 + [X1 ~ Z1 + Z2] | f1"
    default = pf.feols(fml, data=data, weights=weights, vcov={"CRV1": "f1"})
    fit = pf.feols(fml, data=data, weights=weights, vcov={"CRV1": "f1"}, solver=solver)
    np.testing.assert_allclose(fit.coef(), default.coef(), rtol=1e-8)
    np.testing.assert_allclose(fit._vcov, default._vcov, rtol=1e-6)


def test_solve_ols_invalid_solver():
    # Test case 4: Invalid solver
    tZX = np.array([[1, 2], [3, 4]])
    tZY = np.array([5, 6])
    with pytest.raises(ValueError):
        solve_ols(tZX, tZY, "invalid_solver")
