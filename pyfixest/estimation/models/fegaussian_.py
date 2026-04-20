from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.models.feglm_ import Feglm


class Fegaussian(Feglm):
    "Class for the estimation of a fixed-effects GLM with normal errors."

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        ssc_dict: dict[str, str | bool],
        drop_singletons: bool,
        drop_intercept: bool,
        weights: str | None,
        weights_type: str | None,
        collin_tol: float,
        fixef_tol: float,
        fixef_maxiter: int,
        lookup_demeaned_data: dict[frozenset[int], pd.DataFrame],
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
        sample_split_var: str | None = None,
        sample_split_value: str | int | None = None,
        separation_check: list[str] | None = None,
        context: int | Mapping[str, Any] = 0,
        demeaner: AnyDemeaner | None = None,
        accelerate: bool = True,
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
            demeaner=demeaner,
            accelerate=accelerate,
        )

        self._method = "feglm-gaussian"

    def _check_dependent_variable(self) -> None:
        pass

    def _get_deviance(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return np.sum((y - mu) ** 2)

    def _get_dispersion_phi(self, theta: np.ndarray) -> float:
        return np.var(theta)

    def _get_b(self, theta: np.ndarray) -> np.ndarray:
        return theta**2 / 2

    def _get_mu(self, eta: np.ndarray) -> np.ndarray:
        return eta

    def _get_link(self, mu: np.ndarray) -> np.ndarray:
        return mu

    def _get_gprime(self, mu: np.ndarray) -> np.ndarray:
        return np.ones_like(mu)

    def _get_theta(self, mu: np.ndarray) -> np.ndarray:
        return mu

    def _get_V(self, mu: np.ndarray) -> np.ndarray:
        return np.ones_like(mu)

    def _get_score(
        self, y: np.ndarray, X: np.ndarray, mu: np.ndarray, eta: np.ndarray
    ) -> np.ndarray:
        return (y - mu)[:, None] * X
