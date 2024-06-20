import numpy as np
import pytest

import pyfixest as pf
from pyfixest.estimation.feols_ import Feols


@pytest.fixture()
def data():
    data = pf.get_data()
    Y = data["Y"]
    X = data[["X1", "X2"]]
    Z = data[["Z1"]]
    weights = data["weights"]  # Define the variable "weights"
    return X, Y, Z, weights


def test_solver_equivalence(data):
    """
    Test the equivalence of different solvers for the feols class.
    This function initializes an object with test data and compares the results
    obtained from two different solvers: np.linalg.lstsq
    and np.linalg.solve. It asserts that
    the results are identical or very close within a tolerance.
    """
    X, Y, Z, weights = data

    obj = Feols(
        X=X,
        Y=Y,
        weights=weights,
        collin_tol=1e-08,
        coefnames=["X1", "X2"],
        weights_name="weights",
        weights_type=None,
        solver="np.linalg.solve",
    )

    # Use np.linalg.lstsq solver
    obj._solver = "np.linalg.lstsq"
    obj.get_fit()
    beta_hat_lstsq = obj._beta_hat.copy()

    # Use np.linalg.solve solver
    obj._solver = "np.linalg.solve"
    obj.get_fit()
    beta_hat_solve = obj._beta_hat.copy()

    # Assert that the results are identical or very close within a tolerance
    np.testing.assert_array_almost_equal(
        beta_hat_lstsq, beta_hat_solve, decimal=6, err_msg="Solvers' results differ"
    )
