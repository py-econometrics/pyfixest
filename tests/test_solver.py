import numpy as np
import pytest

import pyfixest as pf
from pyfixest.estimation.feiv_ import Feiv
from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.fepois_ import Fepois


@pytest.fixture()
def data_feols():
    data = pf.get_data().dropna()
    Y = data["Y"].to_numpy().reshape((-1, 1))
    X = data[["X1", "X2"]].to_numpy()
    Z = data[["Z1", "X2"]].to_numpy()
    weights = (
        data["weights"].to_numpy().reshape((-1, 1))
    )  # Define the variable "weights"
    return X, Y, Z, weights


@pytest.fixture()
def data_fepois():
    data = pf.get_data(model="Fepois").dropna()
    Y = data["Y"].to_numpy().reshape((-1, 1))
    X = data[["X1", "X2"]].to_numpy()
    weights = (
        data["weights"].to_numpy().reshape((-1, 1))
    )  # Define the variable "weights"
    return X, Y, weights


def test_solver_fepois(data_fepois):
    """
    Test the equivalence of different solvers for the fepois class.
    This function initializes an object with test data and compares the results
    obtained from two different solvers: np.linalg.lstsq
    and np.linalg.solve. It asserts that
    the results are identical or very close within a tolerance.
    """
    X, Y, weights = data_fepois
    obj = Fepois(
        X=X,
        Y=Y,
        weights=weights,
        coefnames=["X1", "X2"],
        collin_tol=1e-08,
        weights_name="weights",
        weights_type=None,
        solver="np.linalg.solve",
        fe=None,
        drop_singletons=False,
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


def test_solver_feiv(data_feols):
    """
    Test the equivalence of different solvers for the feiv class.
    This function initializes an object with test data and compares the results
    obtained from two different solvers: np.linalg.lstsq
    and np.linalg.solve. It asserts that
    the results are identical or very close within a tolerance.
    """
    X, Y, Z, weights = data_feols

    obj = Feiv(
        Y=Y,
        X=X,
        Z=Z,
        endogvar=X[:, 0].reshape((-1, 1)),
        weights=weights,
        coefnames_x=["X1", "X2"],
        collin_tol=1e-08,
        weights_name="weights",
        weights_type=None,
        solver="np.linalg.solve",
        coefnames_z=["Z1"],
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


def test_solver_feols(data_feols):
    """
    Test the equivalence of different solvers for the feols class.
    This function initializes an object with test data and compares the results
    obtained from two different solvers: np.linalg.lstsq
    and np.linalg.solve. It asserts that
    the results are identical or very close within a tolerance.
    """
    X, Y, _, weights = data_feols

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
