import warnings
from collections.abc import Mapping
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from pyfixest.estimation.feglm_ import Feglm
from pyfixest.estimation.formula.parse import Formula as FixestFormula


class Feprobit(Feglm):
    "Class for the estimation of a fixed-effects probit model."

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
        fixef_maxiter: int,
        lookup_demeaned_data: dict[str, pd.DataFrame],
        tol: float,
        maxiter: int,
        solver: Literal[
            "np.linalg.lstsq",
            "np.linalg.solve",
            "scipy.linalg.solve",
            "scipy.sparse.linalg.lsqr",
            "jax",
        ],
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        sample_split_var: Optional[str] = None,
        sample_split_value: Optional[Union[str, int]] = None,
        separation_check: Optional[list[str]] = None,
        context: Union[int, Mapping[str, Any]] = 0,
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
            fixef_maxiter=fixef_maxiter,
            lookup_demeaned_data=lookup_demeaned_data,
            tol=tol,
            maxiter=maxiter,
            solver=solver,
            store_data=store_data,
            copy_data=copy_data,
            lean=lean,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            separation_check=separation_check,
            context=context,
        )

        self._method = "feglm-probit"

    def _check_dependent_variable(self) -> None:
        "Check if the dependent variable is binary with values 0 and 1."
        Y_unique = np.unique(self._Y)
        if len(Y_unique) != 2:
            raise ValueError("The dependent variable must have two unique values.")
        if np.any(~np.isin(Y_unique, [0, 1])):
            raise ValueError("The dependent variable must be binary (0 or 1).")

    def _get_deviance(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        ll_fitted = np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))

        # divide by zero warnings because of the log(0) terms
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ll_saturated = np.sum(
                np.where(y == 0, 0, y * np.log(y))
                + np.where(y == 1, 0, (1 - y) * np.log(1 - y))
            )

        return -2.0 * (ll_fitted - ll_saturated)

    def _get_dispersion_phi(self, theta: np.ndarray) -> float:
        return 1.0

    def _get_b(self, theta: np.ndarray) -> np.ndarray:
        raise ValueError("The function _get_b is not implemented for the probit model.")
        return None

    def _get_mu(self, theta: np.ndarray) -> np.ndarray:
        return norm.cdf(theta)

    def _get_link(self, mu: np.ndarray) -> np.ndarray:
        return norm.ppf(mu)

    def _update_detadmu(self, mu: np.ndarray) -> np.ndarray:
        return 1 / norm.pdf(norm.ppf(mu))

    def _get_theta(self, mu: np.ndarray) -> np.ndarray:
        return norm.ppf(mu)

    def _get_V(self, mu: np.ndarray) -> np.ndarray:
        return mu * (1 - mu)

    def _get_score(
        self, y: np.ndarray, X: np.ndarray, mu: np.ndarray, eta: np.ndarray
    ) -> np.ndarray:
        residual = (y - mu) / (mu * (1 - mu)) * norm.pdf(eta)
        return residual[:, None] * X
