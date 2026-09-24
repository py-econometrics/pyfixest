from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.families import POISSON
from pyfixest.estimation.internals.fit_statistics import poisson_fit_statistics
from pyfixest.estimation.internals.model_state import (
    GlmEstimationOptions,
    ModelDescription,
)
from pyfixest.estimation.models.feglm_ import Feglm
from pyfixest.estimation.models.feols_ import (
    PredictionErrorOptions,
    PredictionType,
)
from pyfixest.utils.dev_utils import DataFrameType


class Fepois(Feglm):
    """
    Estimate a Poisson regression model.

    Non user-facing class to estimate a Poisson regression model via Iterated
    Weighted Least Squares (IWLS).

    Inherits from the Feglm class. Users should not directly instantiate this class,
    but rather use the [fepois()](/reference/estimation.api.fepois.fepois.qmd) function.
    IRLS residualization is orchestrated by ``Feglm`` through the shared
    ``DemeanCache`` supplied by the estimation runner.

    The method implements the algorithm from Stata's `ppmlhdfe` module.

    Attributes
    ----------
    model_matrix : ModelMatrix
        Model frames containing the response, regressors, and other model inputs.
    observation_weights : ObservationWeights
        User-scale observation weights.
    working_state : GlmWorkingState
        Final within-scale IRLS design, response, weights, predictors and residuals.
    fitted_values : FittedValues
        The linear predictor and the response mean of the final IRLS iteration.
    sandwich : SandwichComponents
        IRLS scores, the Hessian X' W X with the final working weights, and its inverse.
    coefnames : list[str]
        Names of the coefficients in the design matrix X.
    options : GlmEstimationOptions
        The estimation options the model was built with, including the IRLS
        `maxiter` and `tol`, the separation check, and the offset.
    _data: pd.DataFrame
        The data frame used in the estimation. None if arguments `lean = True` or
        `store_data = False`.

    Examples
    --------
    `Fepois` is returned by
    [fepois()](/reference/estimation.api.fepois.fepois.qmd) and is not
    constructed directly. Post-estimation methods are inherited from
    [Feols](/reference/estimation.models.feols_.Feols.qmd).

    ```{python}
    import pyfixest as pf

    data = pf.get_data(model="Fepois")
    fit = pf.fepois("Y ~ X1 + X2 | f1", data)

    fit.tidy()
    ```
    """

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        *,
        options: GlmEstimationOptions,
        lookup_demeaned_data: dict[frozenset[int], DemeanedData],
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
        sample_split_var: str | None = None,
        sample_split_value: str | int | None = None,
    ) -> None:
        super().__init__(
            FixestFormula=FixestFormula,
            data=data,
            options=options,
            lookup_demeaned_data=lookup_demeaned_data,
            lookup_preconditioner=lookup_preconditioner,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            family=POISSON,
        )

        # Poisson-specific overrides on top of the Feglm-set defaults.
        self.capabilities = replace(
            self.capabilities,
            crv3_inference=True,
            cluster_causal_variance=False,
            decomposition=False,
        )

    def _describe_model(self, **kwargs: Any) -> ModelDescription:
        """Name the Poisson estimation function."""
        return replace(super()._describe_model(**kwargs), method="fepois")

    def _refit_estimator(self) -> Callable[..., Any]:
        "Return `fepois` for leave-out and resampled refits."
        # lazy loading to avoid circular import
        return import_module("pyfixest.estimation").fepois

    def get_fit(self) -> None:
        "Fit via Feglm IRLS, then add the Poisson likelihood measures."
        super().get_fit()
        y_orig = self.model_matrix.dependent.to_numpy().flatten()
        # ``None`` is the allocation-free unweighted path shared with the rest
        # of the estimation core; no vector of ones is materialised.
        observation_weights = self.observation_weights.values
        self.fitstat = poisson_fit_statistics(
            y=y_orig,
            mu=self.working_state.mu,
            weights=observation_weights,
            deviance=self._family.deviance(
                y_orig, self.working_state.mu, observation_weights
            ),
        )

    def predict(
        self,
        newdata: DataFrameType | None = None,
        atol: float = 1e-6,
        btol: float = 1e-6,
        type: PredictionType = "link",
        se_fit: bool | None = False,
        interval: PredictionErrorOptions | None = None,
        alpha: float = 0.05,
    ) -> np.ndarray | pd.DataFrame:
        """
        Return predicted values from regression model.

        Return a flat np.array with predicted values of the regression model.
        If new fixed effect levels are introduced in `newdata`, predicted values
        for such observations
        will be set to NaN.

        Parameters
        ----------
        newdata : Union[None, pd.DataFrame], optional
            A pd.DataFrame with the new data, to be used for prediction.
            If None (default), uses the data used for fitting the model.
        atol : Float, default 1e-6
            Stopping tolerance for scipy.sparse.linalg.lsqr().
            See https://docs.scipy.org/doc/
                scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        btol : Float, default 1e-6
            Another stopping tolerance for scipy.sparse.linalg.lsqr().
            See https://docs.scipy.org/doc/
                scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        type : str, optional
            The type of prediction to be computed.
            Can be either "response" (default) or "link".
            If type="response", the output is at the level of the response variable,
            i.e., it is the expected predictor E(Y|X).
            If "link", the output is at the level of the explanatory variables,
            i.e., the linear predictor X @ beta.
        atol : Float, default 1e-6
            Stopping tolerance for scipy.sparse.linalg.lsqr().
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        btol : Float, default 1e-6
            Another stopping tolerance for scipy.sparse.linalg.lsqr().
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        se_fit: Optional[bool], optional
            If True, the standard error of the prediction is computed. Only feasible
            for models without fixed effects. GLMs are not supported. Defaults to False.
        interval: str, optional
            The type of interval to compute. Can be either 'prediction' or None.
        alpha: float, optional
            The alpha level for the confidence interval. Defaults to 0.05. Only
            used if interval = "prediction" is not None.

        Returns
        -------
        Union[np.ndarray, pd.DataFrame]
            Returns a pd.Dataframe with columns "fit", "se_fit" and CIs if argument "interval=prediction".
            Otherwise, returns a np.ndarray with the predicted values of the model or the prediction
            standard errors if argument "se_fit=True".
        """
        if se_fit:
            raise NotImplementedError(
                "Prediction with standard errors is not implemented for Poisson regression."
            )

        return super().predict(newdata=newdata, type=type, atol=atol, btol=btol)
