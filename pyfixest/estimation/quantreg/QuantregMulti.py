import inspect
import numpy as np
import pandas as pd
from typing import Union, Optional, Literal, Mapping, Any
from pyfixest.estimation.quantreg.quantreg_ import Quantreg
from pyfixest.estimation.FormulaParser import FixestFormula
from pyfixest.estimation.literals import (
    QuantregMethodOptions,
    SolverOptions,
)
from pyfixest.utils.dev_utils import DataFrameType


class QuantregMulti:

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
        solver: SolverOptions = "np.linalg.solve",
        demeaner_backend: Literal["numba", "jax"] = "numba",
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        context: Union[int, Mapping[str, Any]] = 0,
        sample_split_var: Optional[str] = None,
        sample_split_value: Optional[Union[str, int]] = None,
        quantile: float = 0.5,
        method: QuantregMethodOptions = "fn",
        quantile_tol: float = 1e-06,
        quantile_maxiter: Optional[int] = None,
        seed: Optional[int] = None,
    ):

        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        args_dict = {arg: values[arg] for arg in args if arg not in ("self", "quantile")}

        # initiate a list of Quantreg objects
        self.quantiles = quantile
        self.all_quantregs = [Quantreg(**args_dict, quantile = q) for q in self.quantiles]
        [q.prepare_model_matrix() for q in self.all_quantregs]
        [q.to_array() for q in self.all_quantregs]
        [q.drop_multicol_vars() for q in self.all_quantregs]

        self._X_is_empty = False


    def get_fit(self):

        "Loop over all quantiles using Algo 2 in Chernozhukov et al (2020)."

        # sort q increasing
        q = np.sort(self.quantiles)

        n_quantiles = len(q)
        q_reg_res = {}

        # initial fit
        Quantreg_0 = self.all_quantregs[0]
        beta_hat_mat = np.zeros((len(q), Quantreg_0._X.shape[1]))

        # first quantile regression
        q_reg_res[q[0]] = Quantreg_0.fit_qreg_pfn(X = Quantreg_0._X, Y = Quantreg_0._Y, q = q[0], rng = np.random.default_rng(Quantreg_0._seed))
        beta_hat_mat[0, :] = q_reg_res[q[0]][0]
        Quantreg_0._beta_hat = beta_hat_mat[0, :]
        hessian = Quantreg_0._X.T @ Quantreg_0._X
        Quantreg_0._u_hat = Quantreg_0._Y.flatten() - Quantreg_0._X @ beta_hat_mat[0, :]
        Quantreg_0._hessian = hessian

        for i in range(1, n_quantiles):
            Quantreg_i = self.all_quantregs[i]
            q_reg_res[q[i]] = Quantreg_i.fit_qreg_pfn(X = Quantreg_i._X, Y = Quantreg_i._Y, q = q[i], beta_init=beta_hat_mat[i-1, :])
            beta_hat_mat[i, :] = q_reg_res[q[i]][0]
            Quantreg_i._beta_hat = beta_hat_mat[i, :]
            Quantreg_i._u_hat = Quantreg_i._Y.flatten() - Quantreg_i._X @ beta_hat_mat[i, :]
            Quantreg_i._hessian = hessian
        return q_reg_res

    def vcov(
        self, vcov: Union[str, dict[str, str]], data: Optional[DataFrameType] = None
        ):

        [QuantReg.vcov(vcov = vcov, data = data) for QuantReg in self.all_quantregs]

        return self.all_quantregs

    def get_inference(self):

        [QuantReg.get_inference() for QuantReg in self.all_quantregs]

        return self.all_quantregs

    def prepare_model_matrix(self):
        "Placeholder, only needed due to structure of execution of FixestMulti class."
        pass

    def to_array(self):
        "Placeholder, only needed due to structure of execution of FixestMulti class."
        pass

    def drop_multicol_vars(self):
        "Placeholder, only needed due to structure of execution of FixestMulti class."
        pass

    def wls_transform(self):
        "Placeholder, only needed due to structure of execution of FixestMulti class."
        pass

    def _clear_attributes(self):
        [QuantReg._clear_attributes() for QuantReg in self.all_quantregs]

        return self.all_quantregs

