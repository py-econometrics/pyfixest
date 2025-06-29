import inspect
import gc
from collections.abc import Mapping
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from pyfixest.estimation.FormulaParser import FixestFormula
from pyfixest.estimation.literals import (
    QuantregMethodOptions,
    QuantregMultiOptions,
    SolverOptions,
)
from pyfixest.estimation.quantreg.quantreg_ import Quantreg
from pyfixest.estimation.quantreg.utils import get_hall_sheather_bandwidth
from pyfixest.utils.dev_utils import DataFrameType


class QuantregMulti:
    "Run the quantile regression process efficiently. Wrapper around Quantreg calls."

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        quantile: list[float],
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
        method: QuantregMethodOptions = "fn",
        multi_method: QuantregMultiOptions = "cfm1",
        quantile_tol: float = 1e-06,
        quantile_maxiter: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        frame = inspect.currentframe()
        if frame is None:
            raise ValueError("The current frame is None.")
        args, _, _, values = inspect.getargvalues(frame)
        args_dict = {
            arg: values[arg]
            for arg in args
            if arg not in ("self", "quantile", "multi_method")
        }

        # initiate a list of Quantreg objects
        self.quantiles = quantile
        self.all_quantregs = {
            q: Quantreg(**args_dict, quantile=q) for q in self.quantiles
        }
        self.method = method
        self.multi_method = multi_method
        self._is_iv = False

        # TODO: call model_matrix(), to_array(), drop_multicol_vars() only once for model 0
        # and then copy the attributes to the other models (less robust? but faster)
        [q.prepare_model_matrix() for q in self.all_quantregs.values()]
        [q.to_array() for q in self.all_quantregs.values()]
        [q.drop_multicol_vars() for q in self.all_quantregs.values()]

        self._X_is_empty = False

    def get_fit(self):
        "Fit multiple quantile regressions via either algo 2 or 3 of CFM."
        # sort q increasing
        q = np.sort(self.quantiles)
        n_quantiles = len(q)

        if n_quantiles % 2 == 1:
            q_median_idx = n_quantiles // 2
        else:
            q_median_idx = (n_quantiles // 2) - 1

        q_median = q[q_median_idx]

        # data fixed across qregs, just need take from first one
        X = self.all_quantregs[q[q_median_idx]]._X
        Y = self.all_quantregs[q[q_median_idx]]._Y
        hessian = X.T @ X
        N = self.all_quantregs[q[q_median_idx]]._N
        rng = np.random.default_rng(self.all_quantregs[q[q_median_idx]]._seed)

        # fit first quantile regression using "pfn"

        fit_kwargs = {
            "X": X,
            "Y": Y,
            "q": q_median, # first eval at the "central" quantile
        }

        if self.method == "pfn":
            fit_kwargs["rng"] = rng
        beta_hat = self.all_quantregs[q[q_median_idx]]._fit(**fit_kwargs)[0]

        self.all_quantregs[q[q_median_idx]]._beta_hat = beta_hat
        self.all_quantregs[q[q_median_idx]]._u_hat = Y.flatten() - (X @ beta_hat).flatten()
        self.all_quantregs[q[q_median_idx]]._hessian = hessian

        def _direction_helper(i, direction):
            if direction == "left":
                i_prev = i + 1
            elif direction == "right":
                i_prev = i - 1
            else:
                raise ValueError(f"Direction must be 'left' or 'right' but is {direction}.")

            return i_prev

        if self.multi_method == "cfm1":

            def _cfm1_fun(i, direction):

                i_prev = _direction_helper(i, direction)

                beta_hat_prev = self.all_quantregs[q[i_prev]]._beta_hat
                beta_hat = self.all_quantregs[q[i]].fit_qreg_pfn(
                    X=X, Y=Y, q=q[i], beta_init=beta_hat_prev, eta=0.5
                )[0]
                self.all_quantregs[q[i]]._beta_hat = beta_hat
                self.all_quantregs[q[i]]._u_hat = Y.flatten() - (X @ beta_hat).flatten()
                self.all_quantregs[q[i]]._hessian = hessian

            for i in range(q_median_idx - 1, -1, -1):
                _cfm1_fun(i, "left")

            for i in range(q_median_idx + 1, n_quantiles, 1):
                _cfm1_fun(i, "right")

        elif self.multi_method == "cfm2":

            def _cfm2_fun(i, direction):

                i_prev = _direction_helper(i, direction)

                beta_hat_prev = self.all_quantregs[q[i_prev]]._beta_hat
                u_hat_prev = self.all_quantregs[q[i_prev]]._u_hat

                kappa = np.median(np.abs(u_hat_prev - np.median(u_hat_prev)))
                h_G = get_hall_sheather_bandwidth(q=q[i_prev], N=N)
                delta = kappa * (norm.ppf(q[i_prev] + h_G) - norm.ppf(q[i_prev] - h_G))
                J = (np.sum(np.abs(u_hat_prev) < delta) * hessian) / (2 * N * delta)

                M = X.T @ (q[i] - (u_hat_prev < 0))[:, None]
                beta_new = beta_hat_prev + np.linalg.solve(J, M).flatten()

                self.all_quantregs[q[i]]._beta_hat = beta_new
                self.all_quantregs[q[i]]._u_hat = (
                    self.all_quantregs[q[i]]._Y.flatten()
                    - self.all_quantregs[q[i]]._X @ beta_new
                )
                self.all_quantregs[q[i]]._hessian = hessian

            for i in range(q_median_idx - 1, -1, -1):
                _cfm2_fun(i, "left")

            for i in range(q_median_idx + 1, n_quantiles, 1):
                _cfm2_fun(i, "right")

        else:
            raise ValueError(
                f"Multi method needs to be of type 'cfm1' or 'cfm2' but is {self.multi_method}."
            )

        # sort self.all_quantregs by q
        self.all_quantregs = dict(sorted(self.all_quantregs.items(), key=lambda item: item[0]))
        return self.all_quantregs

    def vcov(
        self, vcov: Union[str, dict[str, str]], data: Optional[DataFrameType] = None
    ):
        "Compute variance-covariance matrices for all models in the quantile regression process."
        [
            QuantReg.vcov(vcov=vcov, data=data)
            for QuantReg in self.all_quantregs.values()
        ]

        return self.all_quantregs

    def get_inference(self):
        "Compute inference for all models of the quantile regression process."
        [QuantReg.get_inference() for QuantReg in self.all_quantregs.values()]

        return self.all_quantregs

    def prepare_model_matrix(self):
        "Prepare model matrix. Placeholder, only needed due to structure of execution of FixestMulti class."
        pass

    def to_array(self):
        "Covert to array. Placeholder, only needed due to structure of execution of FixestMulti class."
        pass

    def drop_multicol_vars(self):
        "Drop multicollinear variables. Placeholder, only needed due to structure of execution of FixestMulti class."
        pass

    def wls_transform(self):
        "Apply the WLS transform. Placeholder, only needed due to structure of execution of FixestMulti class."
        pass

    def demean(self):
        "Demean the data. Placeholder, only needed due to structure of execution of FixestMulti class."
        pass

    def get_performance(self):
        "Compute performance metrics for all models of the quantile regression process."
        [QuantReg.get_performance() for QuantReg in self.all_quantregs.values()]
        return self.all_quantregs

    def _clear_attributes(self):
        "Clear all large non-necessary attributes to free memory."
        [QuantReg._clear_attributes() for QuantReg in self.all_quantregs.values()]
        del_attributes = ["_X","_Y"]
        for QuantReg in self.all_quantregs.values():
            for attr in del_attributes:
                if hasattr(QuantReg, attr):
                    delattr(QuantReg, attr)
        gc.collect()

        return self.all_quantregs
