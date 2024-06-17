import unittest

import numpy as np

from pyfixest import feols


class TestSolvers(unittest.TestCase):
    """A test case class for testing the solvers in the feols class."""

    def setUp(self):
        """
        Set up the test environment.

        This method is called before each test case is executed.
        It initializes the test data
        by generating random arrays for X, Y, and Z.
        """
        np.random.seed(42)  # Ensure reproducibility
        self._X = np.random.rand(100, 5)
        self._Y = np.random.rand(100, 1)
        self._Z = np.random.rand(
            100, 5
        )  # Assuming Z is used similarly to X in your context

    def test_solver_equivalence(self):
        """
        Test the equivalence of different solvers for the feols class.

        This method initializes an object with test data and compares the results
        obtained from two different solvers: np.linalg.lstsq
        and np.linalg.solve. It asserts that
        the results are identical or very close within a tolerance.

        Returns
        -------
            None
        """
        obj = feols()  # Replace with your actual class initialization
        obj._X = self._X
        obj._Y = self._Y
        obj._Z = self._Z

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


if __name__ == "__main__":
    unittest.main()
