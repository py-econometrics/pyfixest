from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.families import POISSON, pois_deviance_weighted
from pyfixest.estimation.internals.literals import (
    SolverOptions,
)
from pyfixest.estimation.internals.separation import check_for_separation
from pyfixest.estimation.internals.vcov_ import vcov_iid_glm
from pyfixest.estimation.models.feglm_ import Feglm
from pyfixest.estimation.models.feols_ import (
    Feols,
    PredictionErrorOptions,
    PredictionType,
)
from pyfixest.utils.dev_utils import DataFrameType, _check_series_or_dataframe


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
        )

        # Poisson-specific overrides on top of the Feglm-set defaults.
        self._method = "fepois"
        self._family = POISSON
        self._offset_name = offset
        self._supports_cluster_causal_variance = False
        self._support_decomposition = False

    def prepare_model_matrix(self):
        "Prepare model inputs for estimation."
        # Skip Feglm.prepare_model_matrix's separation handling; Fepois does its
        # own below with additional drops (offset_df, weights_df).
        Feols.prepare_model_matrix(self)

        # check that self._Y is a pandas Series or DataFrame
        self._Y = _check_series_or_dataframe(self._Y)

        # Y >= 0 enforcement is delegated to POISSON.check_y, invoked by
        # Feglm._check_dependent_variable after prepare_model_matrix.

        # check for separation
        na_separation: list[int] = []
        if (
            self._fe is not None
            and self.separation_check is not None
            and self.separation_check  # not an empty list
        ):
            na_separation = check_for_separation(
                Y=self._Y,
                X=self._X,
                fe=self._fe,
                fml=self._fml,
                data=self._data,
                methods=self.separation_check,
            )

        if na_separation:
            self._Y.drop(na_separation, axis=0, inplace=True)
            self._X.drop(na_separation, axis=0, inplace=True)
            self._fe.drop(na_separation, axis=0, inplace=True)
            self._data.drop(na_separation, axis=0, inplace=True)
            if self._weights_df is not None:
                self._weights_df.drop(na_separation, axis=0, inplace=True)
            if self._offset_df is not None:
                self._offset_df.drop(na_separation, axis=0, inplace=True)
            self._N = self._Y.shape[0]
            self._N_rows = self._N
            # Re-set weights after dropping rows (handles both weighted and unweighted)
            self._weights = self._set_weights()

            self.na_index = np.concatenate([self.na_index, np.array(na_separation)])
            self.n_separation_na = len(na_separation)
            # possible to have dropped fixed effects level due to separation
            self._k_fe = self._fe.nunique(axis=0) if self._has_fixef else None
            self._n_fe = np.sum(self._k_fe > 1) if self._has_fixef else 0

    def to_array(self):
        "Turn estimation DataFrames to np arrays and resolve the offset."
        super().to_array()
        if self._offset_df is not None:
            self._offset = self._offset_df.to_numpy().reshape((-1, 1))
        else:
            self._offset = np.zeros((self._N, 1))

    def get_fit(self) -> None:
        "Fit via Feglm IRLS, then add Poisson-specific post-fit summary stats."
        # Stash original Y and user weights before super().get_fit() overwrites
        # self._Y / self._weights with the WLS-transformed versions used for
        # inference.
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

        # The in-loop deviance set by Feglm is unweighted; Fepois reports the
        # user-weighted version.
        self.deviance = pois_deviance_weighted(
            y_orig, self._Y_hat_response, user_weights
        )

    def resid(self, type: str = "response") -> np.ndarray:
        """
        Return residuals from regression model.

        Parameters
        ----------
        type : str, optional
            The type of residuals to be computed.
            Can be either "response" (default) or "working".

        Returns
        -------
        np.ndarray
            A flat array with the residuals of the regression model.
        """
        if type == "response":
            return self._u_hat_response.flatten()
        elif type == "working":
            return self._u_hat_working.flatten()
        else:
            raise ValueError("type must be one of 'response' or 'working'.")

    def _vcov_iid(self):
        return vcov_iid_glm(bread=self._bread)

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


