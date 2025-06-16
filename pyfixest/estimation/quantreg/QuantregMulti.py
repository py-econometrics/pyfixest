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
from scipy.stats import norm
from pyfixest.estimation.quantreg.utils import get_hall_sheather_bandwidth


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
        self.all_quantregs = {q: Quantreg(**args_dict, quantile=q) for q in self.quantiles}

        # TODO: call model_matrix(), to_array(), drop_multicol_vars() only once for model 0
        # and then copy the attributes to the other models (less robust? but faster)
        [q.prepare_model_matrix() for q in self.all_quantregs.values()]
        [q.to_array() for q in self.all_quantregs.values()]
        [q.drop_multicol_vars() for q in self.all_quantregs.values()]

        self._X_is_empty = False


    def get_fit(self):

        "Loop over all quantiles using Algo 2 in Chernozhukov et al (2020)."

        # sort q increasing
        q = np.sort(self.quantiles)

        n_quantiles = len(q)

        # data fixed across qregs, just need take from first one
        X = self.all_quantregs[q[0]]._X
        Y = self.all_quantregs[q[0]]._Y
        hessian = X.T @ X
        rng = np.random.default_rng(self.all_quantregs[q[0]]._seed)

        # fit first quantile regression using "pfn"
        beta_hat = self.all_quantregs[q[0]].fit_qreg_pfn(X = X, Y = Y, q = q[0], rng = rng)[0]
        self.all_quantregs[q[0]]._beta_hat = beta_hat
        self.all_quantregs[q[0]]._u_hat = Y.flatten() - X @ beta_hat
        self.all_quantregs[q[0]]._hessian = hessian

        # sequentially loop over all other quantiles
        for i in range(1, n_quantiles):

            beta_hat_prev = self.all_quantregs[q[i-1]]._beta_hat
            beta_hat = self.all_quantregs[q[i]].fit_qreg_pfn(X = X, Y = Y, q = q[i], beta_init=beta_hat_prev)[0]
            self.all_quantregs[q[i]]._beta_hat = beta_hat
            self.all_quantregs[q[i]]._u_hat = Y.flatten() - X @ beta_hat
            self.all_quantregs[q[i]]._hessian = hessian

        return self.all_quantregs

    def get_fit2(self):

        # sort q increasing

        q = np.sort(self.quantiles)

        n_quantiles = len(q)
        q_reg_res = {}

        # initial fit
        Quantreg_0 = self.all_quantregs[0]
        X = Quantreg_0._X
        Y = Quantreg_0._Y
        hessian = X.T @ X
        rng = np.random.default_rng(Quantreg_0._seed)


        # first quantile regression
        q_reg_res[q[0]] = Quantreg_0.fit_qreg_pfn(X = X, Y = Y, q = q[0], rng = rng)
        Quantreg_0._beta_hat = q_reg_res[q[0]][0]
        Quantreg_0._u_hat = Y.flatten() - X @ Quantreg_0._beta_hat
        Quantreg_0._hessian = hessian

        # kernel estimate at zero: J


        for i in range(1, n_quantiles):

            # compute J
            # cn in Koenker textbook p.81
            Quantreg_i_prev = self.all_quantregs[i-1]
            u_hat_prev = Quantreg_i_prev._u_hat
            N_prev = Quantreg_i_prev._N
            beta_hat_prev = Quantreg_i_prev._beta_hat

            # compute inv(J)
            kappa = np.median(np.abs(u_hat_prev - np.median(u_hat_prev)))
            h_G = get_hall_sheather_bandwidth(q=q[i-1], N=N_prev)
            delta = kappa * (norm.ppf(q[i-1] + h_G) - norm.ppf(q[i-1] - h_G))
            J = (np.sum(np.abs(u_hat_prev) < delta) * hessian) / (2 * N_prev * delta)
            Jinv = np.linalg.inv(J)

            beta_new = beta_hat_prev + Jinv @ np.mean((q[i] - (u_hat_prev < 0))[:,None] * X, axis = 0)

            self.all_quantregs[i]._beta_hat = beta_new
            self.all_quantregs[i]._u_hat = self.all_quantregs[i]._Y.flatten() - self.all_quantregs[i]._X @ beta_new
            self.all_quantregs[i]._hessian = hessian
            q_reg_res[q[i]] = self.all_quantregs[i]

        return q_reg_res


    def vcov(
        self, vcov: Union[str, dict[str, str]], data: Optional[DataFrameType] = None
        ):

        [QuantReg.vcov(vcov = vcov, data = data) for QuantReg in self.all_quantregs.values()]

        return self.all_quantregs

    def get_inference(self):

        [QuantReg.get_inference() for QuantReg in self.all_quantregs.values()]

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
        [QuantReg._clear_attributes() for QuantReg in self.all_quantregs.values()]

        return self.all_quantregs

