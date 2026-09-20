import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation.quantreg.frisch_newton_ip import frisch_newton_solver
from pyfixest.estimation.quantreg.frisch_newton_ip_sparse import (
    frisch_newton_solver_sparse,
)


def _categorical_data(N=1500, n_cat=40, seed=0):
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, n_cat, N)
    x1 = rng.normal(size=N)
    x2 = rng.normal(size=N)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(size=n_cat)[cat] + rng.standard_t(5, N)
    return pd.DataFrame({"Y": y, "X1": x1, "X2": x2, "g": pd.Categorical(cat)})


def _objective(u_hat, q):
    "Check-function loss actually minimised by quantile regression."
    return float(np.sum(u_hat * (q - (u_hat < 0))))


@pytest.mark.parametrize("q", [0.25, 0.5, 0.75])
def test_sparse_solver_matches_dense_solver(q):
    """The sparse solver takes the same path as the dense one on a dense problem."""
    import scipy.sparse as sp
    from scipy.linalg import cho_factor, solve_triangular

    rng = np.random.default_rng(42)
    N, k = 400, 5
    X = np.column_stack([np.ones(N), rng.normal(size=(N, k - 1))])
    Y = X @ rng.normal(size=k) + rng.standard_t(5, N)

    b = (1 - q) * X.T @ np.ones(N)
    chol, _ = cho_factor(X.T @ X, lower=True, check_finite=False)
    chol = np.atleast_2d(chol)
    P = solve_triangular(chol, X.T, lower=True, check_finite=False)

    dense = frisch_newton_solver(
        A=X.T, b=b, c=-Y, u=np.ones(N), q=q, tol=1e-6, max_iter=N, chol=chol, P=P
    )
    sparse = frisch_newton_solver_sparse(
        A=sp.csr_matrix(X.T), b=b, c=-Y, u=np.ones(N), q=q, tol=1e-6, max_iter=N
    )

    assert dense[1] and sparse[1], "both solvers should converge"
    assert dense[2] == sparse[2], "both solvers should take the same number of steps"
    np.testing.assert_allclose(dense[0], sparse[0], atol=1e-6)


@pytest.mark.parametrize("q", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("sparse_method,dense_method", [("sfn", "fn"), ("psfn", "pfn")])
def test_sparse_method_matches_dense_method(q, sparse_method, dense_method):
    """sfn/psfn reach the same optimum as fn/pfn on a design with many dummies.

    The comparison is on the objective and on the non-categorical coefficients, not on the
    full coefficient vector. With many sparsely-populated categories the quantile-regression
    LP has a flat optimal face rather than a unique vertex, so two solvers can return
    different points with an identical objective value. Asserting coefficient equality on
    the dummies would be testing an arbitrary choice among optima.
    """
    data = _categorical_data()
    fml = "Y ~ X1 + X2 + C(g)"

    dense = pf.quantreg(fml, data=data, quantile=q, method=dense_method, seed=42)
    sparse = pf.quantreg(fml, data=data, quantile=q, method=sparse_method, seed=42)

    assert dense._has_converged and sparse._has_converged

    obj_dense = _objective(dense._u_hat, q)
    obj_sparse = _objective(sparse._u_hat, q)
    np.testing.assert_allclose(obj_sparse, obj_dense, rtol=1e-8)

    keep = [c for c in dense.coef().index if not c.startswith("C(g)")]
    np.testing.assert_allclose(
        sparse.coef()[keep].to_numpy(), dense.coef()[keep].to_numpy(), atol=1e-5
    )


def test_sparse_method_is_registered():
    """Both new methods are reachable through the public API and rejected names still raise."""
    data = _categorical_data(N=400, n_cat=10)
    for method in ("sfn", "psfn"):
        fit = pf.quantreg("Y ~ X1", data=data, quantile=0.5, method=method, seed=1)
        assert fit._has_converged

    with pytest.raises(ValueError):
        pf.quantreg("Y ~ X1", data=data, quantile=0.5, method="not-a-method")
