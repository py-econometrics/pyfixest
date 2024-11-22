import numpy as np
import pandas as pd
from typing import Union, Optional
from pyfixest.estimation.FormulaParser import FixestFormula
from pyfixest.estimation.model_matrix_fixest_ import model_matrix_fixest
from pyfixest.estimation.fepois_ import Fepois

class Felogit(Fepois):

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        ssc_dict: dict[str, Union[str, bool]],
        drop_singletons: bool,
        drop_intercept: bool,
        weights: Optional[str],
        weights_type: Optional[str],
        collin_tol: float,
        fixef_tol: float,
        lookup_demeaned_data: dict[str, pd.DataFrame],
        tol: float,
        maxiter: int,
        solver: str = "np.linalg.solve",
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        sample_split_var: Optional[str] = None,
        sample_split_value: Optional[Union[str, int]] = None,
        separation_check: Optional[list[str]] = None,
    ):
        super().__init__(
            FixestFormula=FixestFormula,
            data=data,
            ssc_dict=ssc_dict,
            drop_singletons=drop_singletons,
            drop_intercept=drop_intercept,
            weights=weights,
            weights_type=weights_type,
            collin_tol=collin_tol,
            fixef_tol=fixef_tol,
            lookup_demeaned_data=lookup_demeaned_data,
            tol=tol,
            maxiter=maxiter,
            solver=solver,
            store_data=store_data,
            copy_data=copy_data,
            lean=lean,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            separation_check=separation_check
        )

    def get_fit(self):

        "Fit GLM model via iterated weighted least squares."

        # initiate IWLS algorithm

        self.iterate_iwls()
        # collect results

    def iterate_iwls(self):

        "Iterate over IWLS updates until convergence."

        for i in range(self._maxiter):
            if stop_iterating:
                _convergence = True
                break
            if i == _maxiter:
                raise NonConvergenceError(
                    f"""
                    The IWLS algorithm did not converge with {_maxiter}
                    iterations. Try to increase the maximum number of iterations.
                    """
                )

            self.update_iwls()
            self.check_convergence()


    def update_iwls(self):

        "One update step of the FWLS algorithm."

        _Y = self._Y
        _X = self._X
        _fe = self._fe

        "Update GLM fit via Gauss-Newton Algorithm."
        # define custom class for each GLM method
        pass

    def check_convergence(self):
        "Check for convergence of the GLM algorithm."
        pass

    def fixef(self):
        "Obtain alpha coefficients."
        pass

    def predict(self):
        "Make predictions."
        pass