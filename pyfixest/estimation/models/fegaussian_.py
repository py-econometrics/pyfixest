from collections.abc import Mapping
from typing import Any, Literal

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
        return vcov_iid_ols(
            residuals=self.working_state.working_residuals,
            bread=self._bread,
            N=self._N,
            weights=self.observation_weights.values,
        )

    def get_performance(self) -> None:
        """Compute and store Gaussian fit statistics from retained model data.

        Gaussian fits retain their demeaned response and residuals in
        working_state rather than the linear model's within_data and _u_hat.
        The identity link puts those arrays in the units of Y, so they can
        be passed to the same performance_measures helper used for OLS.
        The original response comes from model_matrix for the overall R².
        """
        working_state = self.working_state
        measures = performance_measures(
            Y=self.model_matrix.dependent.to_numpy(),
            Y_within=working_state.working_response_within.reshape((-1, 1)),
            residuals=working_state.response_residuals,
            weights=self.observation_weights.values,
            N=self._N,
            k=self._k,
            k_fe=self._n_fixef_coefficients(),
            has_intercept=not self._drop_intercept,
            has_fixef=self._has_fixef,
        )
        self._store_performance(measures)
