from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.families import GAUSSIAN
from pyfixest.estimation.internals.performance_ import performance_measures
from pyfixest.estimation.internals.vcov_ import vcov_iid_ols
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
        lookup_demeaned_data: dict[frozenset[int], DemeanedData],
        tol: float,
        maxiter: int,
        solver: Literal[
            "np.linalg.lstsq",
            "np.linalg.solve",
            "scipy.linalg.solve",
            "scipy.sparse.linalg.lsqr",
        ],
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        sample_split_var: str | None = None,
        sample_split_value: str | int | None = None,
        separation_check: list[str] | None = None,
        context: int | Mapping[str, Any] = 0,
        demeaner: AnyDemeaner | None = None,
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
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
            lookup_preconditioner=lookup_preconditioner,
            accelerate=accelerate,
            family=GAUSSIAN,
        )

        self._method = "feglm-gaussian"

    def _vcov_iid(self):
        # we set gaussian glms to match pf.feols exactly
        return vcov_iid_ols(residuals=self._u_hat, bread=self._bread, N=self._N)

    def get_performance(self) -> None:
        """Compute R² measures from response-scale arrays.

        In this layer `_Y` is the sqrt(W)-scaled IRLS within response. For the
        Gaussian family W equals the observation weights, so dividing by
        sqrt(W) recovers the within response in the units of Y.
        """
        sqrt_irls_weights = np.sqrt(self._irls_weights).reshape((-1, 1))
        measures = performance_measures(
            Y=self._Y_untransformed.to_numpy(),
            Y_within=self._Y.reshape((-1, 1)) / sqrt_irls_weights,
            residuals=self._u_hat_response,
            weights=self._observation_weights.values,
            N=self._N,
            k=self._k,
            k_fe=self._n_fixef_coefficients(),
            has_intercept=not self._drop_intercept,
            has_fixef=self._has_fixef,
        )
        self._store_performance(measures)
