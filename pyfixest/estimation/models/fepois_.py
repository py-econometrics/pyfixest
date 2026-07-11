"""Implement Poisson fixed-effects model results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.families import POISSON
from pyfixest.estimation.internals.literals import (
    SolverOptions,
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
    Note that no demeaning is performed in this class: demeaning is performed in the
    FixestMulti class (to allow for caching of demeaned variables for multiple estimation).

    The method implements the algorithm from Stata's `ppmlhdfe` module.

    Attributes
    ----------
    _Y : np.ndarray
        The demeaned dependent variable, a two-dimensional numpy array.
    _X : np.ndarray
        The demeaned independent variables, a two-dimensional numpy array.
    _fe : np.ndarray
        Fixed effects, a two-dimensional numpy array or None.
    weights : np.ndarray
        Weights, a one-dimensional numpy array or None.
    coefnames : list[str]
        Names of the coefficients in the design matrix X.
    drop_singletons : bool
        Whether to drop singleton fixed effects.
    collin_tol : float
        Tolerance level for the detection of collinearity.
    maxiter : Optional[int], default=25
        Maximum number of iterations for the IRLS algorithm.
    tol : Optional[float], default=1e-08
        Tolerance level for the convergence of the IRLS algorithm.
    solver : str, optional.
        The solver to use for the regression. Can be "np.linalg.lstsq",
        "np.linalg.solve", "scipy.linalg.solve" and "scipy.sparse.linalg.lsqr".
        Defaults to "scipy.linalg.solve".
    demeaner : Optional[AnyDemeaner]
        Resolved typed demeaner configuration.
    fixef_tol: float, default = 1e-06.
        Tolerance level for the convergence of the demeaning algorithm.
    context : int or Mapping[str, Any]
        A dictionary containing additional context variables to be used by
        formulaic during the creation of the model matrix. This can include
        custom factorization functions, transformations, or any other
        variables that need to be available in the formula environment.
    weights_name : Optional[str]
        Name of the weights variable.
    weights_type : Optional[str]
        Type of weights variable.
    _data: pd.DataFrame
        The data frame used in the estimation. None if arguments `lean = True` or
        `store_data = False`.
    """

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
        lookup_demeaned_data: dict[frozenset[int], pd.DataFrame],
        tol: float,
        maxiter: int,
        solver: SolverOptions = "np.linalg.solve",
        demeaner: AnyDemeaner | None = None,
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
        context: int | Mapping[str, Any] = 0,
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        sample_split_var: str | None = None,
        sample_split_value: str | int | None = None,
        separation_check: list[str] | None = None,
        offset: str | None = None,
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
            family=POISSON,
        )

        # Poisson-specific overrides on top of the Feglm-set defaults.
        self._method = "fepois"
        self._offset_name = offset
        self._supports_cluster_causal_variance = False
        self._support_decomposition = False

    def get_fit(self) -> None:
        "Fit via Feglm IRLS, then add Poisson-specific post-fit summary stats."
        y_orig = np.asarray(self._Y).flatten()
        user_weights = self._weights.flatten().copy()

        super().get_fit()

        self._y_hat_null = np.full_like(
            y_orig, np.average(y_orig, weights=user_weights), dtype=float
        )

        self._loglik = np.sum(
            user_weights
            * (
                y_orig * np.log(self._Y_hat_response)
                - self._Y_hat_response
                - gammaln(y_orig + 1)
            )
        )

        # cant replicate fixest atm
        if self._has_weights:
            self._loglik_null = None
            self._pseudo_r2 = None
        else:
            self._loglik_null = np.sum(
                user_weights
                * (
                    y_orig * np.log(self._y_hat_null)
                    - self._y_hat_null
                    - gammaln(y_orig + 1)
                )
            )
            self._pseudo_r2 = 1 - (self._loglik / self._loglik_null)
        self._pearson_chi2 = np.sum(
            user_weights * (y_orig - self._Y_hat_response) ** 2 / self._Y_hat_response
        )

        self.deviance = self._family.deviance(
            y_orig, self._Y_hat_response, user_weights
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
