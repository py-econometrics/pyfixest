from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.special import gammaln

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.fit_pois_ import (
    fit_pois_irls,
    pois_deviance,
)
from pyfixest.estimation.internals.literals import (
    SolverOptions,
)
from pyfixest.estimation.internals.vcov_ import vcov_iid_glm
from pyfixest.estimation.models.feols_ import (
    Feols,
    PredictionErrorOptions,
    PredictionType,
)
from pyfixest.utils.dev_utils import DataFrameType, _check_series_or_dataframe


class Fepois(Feols):
    """
    Estimate a Poisson regression model.

    Non user-facing class to estimate a Poisson regression model via Iterated
    Weighted Least Squares (IWLS).

    Inherits from the Feols class. Users should not directly instantiate this class,
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
            solver=solver,
            store_data=store_data,
            copy_data=copy_data,
            lean=lean,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            context=context,
            demeaner=demeaner,
            lookup_preconditioner=lookup_preconditioner,
        )

        # input checks
        _fepois_input_checks(
            drop_singletons=drop_singletons,
            tol=tol,
            maxiter=maxiter,
        )

        self.maxiter = maxiter
        self.tol = tol
        self._method = "fepois"
        self.convergence = False
        self.separation_check = separation_check
        self._offset_name = offset

        self._support_crv3_inference = True
        self._support_iid_inference = True
        self._supports_cluster_causal_variance = False
        self._support_decomposition = False

        self._Y_hat_response = np.array([])
        self.deviance: float | None = None
        self._Xbeta = np.array([])

    def prepare_model_matrix(self):
        "Prepare model inputs for estimation."
        super().prepare_model_matrix()

        # check that self._Y is a pandas Series or DataFrame
        self._Y = _check_series_or_dataframe(self._Y)

        # check that self._Y is a weakly positive number
        if np.any(self._Y < 0):
            raise ValueError("The dependent variable must be a weakly positive number.")

        # check for separation
        na_separation: list[int] = []
        if (
            self._fe is not None
            and self.separation_check is not None
            and self.separation_check  # not an empty list
        ):
            na_separation = _check_for_separation(
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
        "Turn estimation DataFrames to np arrays."
        self._Y, self._X, self._Z = (
            self._Y.to_numpy(),
            self._X.to_numpy(),
            self._X.to_numpy(),
        )
        if self._fe is not None:
            self._fe = self._fe.to_numpy()
            if self._fe.ndim == 1:
                self._fe = self._fe.reshape((self._N, 1))
        if self._offset_df is not None:
            self._offset = self._offset_df.to_numpy().reshape((-1, 1))
        else:
            self._offset = np.zeros((self._N, 1))

    def get_fit(self) -> None:
        "Fit a fixed-effects Poisson model via IRLS and write results onto self.*."
        self.to_array()
        assert self._offset is not None  # set in prepare_model_matrix

        def _demean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
            if self._fe is None:
                return x
            return self._demean_cache.demean_array(
                x=x,
                flist=self._fe,
                weights=w,
                na_index=self._na_index,
                demeaner=self._demeaner,
            )

        fit = fit_pois_irls(
            X=self._X,
            Y=self._Y,
            offset=self._offset,
            weights=self._weights,
            demean=_demean,
            coefnames=self._coefnames,
            collin_tol=self._collin_tol,
            solver=self._solver,
            maxiter=self.maxiter,
            tol=self.tol,
        )

        self._coefnames = fit.coefnames
        self._collin_vars = fit.collin_vars
        self._collin_index = fit.collin_index
        self._X = fit.X
        self._X_is_empty = self._X.shape[1] == 0
        self._k = self._X.shape[1]

        self._beta_hat = fit.beta
        self._Y_hat_response = fit.mu.flatten()
        self._Y_hat_link = fit.eta.flatten()

        user_weights = self._weights.flatten()
        self._irls_weights = fit.W.flatten()

        WX = fit.sqrt_W * fit.X_tilde
        WZ = fit.sqrt_W * fit.z_tilde
        beta_col = self._beta_hat.reshape((-1, 1))

        self._u_hat = (WZ - WX @ beta_col).flatten()
        self._u_hat_working = fit.z_tilde - fit.X_tilde @ beta_col
        self._u_hat_response = self._Y - fit.mu

        y = self._Y.flatten()
        self._y_hat_null = np.full_like(
            y, np.average(y, weights=user_weights), dtype=float
        )

        self._loglik = np.sum(
            user_weights
            * (y * np.log(self._Y_hat_response) - self._Y_hat_response - gammaln(y + 1))
        )

        # cant replicate fixest atm
        if self._has_weights:
            self._loglik_null = None
            self._pseudo_r2 = None
        else:
            self._loglik_null = np.sum(
                user_weights
                * (y * np.log(self._y_hat_null) - self._y_hat_null - gammaln(y + 1))
            )
            self._pseudo_r2 = 1 - (self._loglik / self._loglik_null)
        self._pearson_chi2 = np.sum(
            user_weights * (y - self._Y_hat_response) ** 2 / self._Y_hat_response
        )

        self._weights = fit.W
        self.deviance = pois_deviance(y, fit.mu, user_weights)

        self._Y = WZ
        self._X = WX
        self._Z = self._X

        self._tZX = self._Z.T @ self._X
        self._tZXinv = np.linalg.inv(self._tZX)
        self._Xbeta = fit.eta

        self._scores = self._u_hat[:, None] * self._X
        self._hessian = WX.T @ WX

        self.convergence = fit.converged
        if self.convergence:
            self._convergence = True

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


def _check_for_separation(
    fml: str,
    data: pd.DataFrame,
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: pd.DataFrame,
    methods: list[str] | None = None,
) -> list[int]:
    """
    Check for separation.

    Check for separation of Poisson Regression. For details, see the ppmlhdfe
    documentation on separation checks.

    Parameters
    ----------
    fml : str
        The formula used for estimation.
    data : pd.DataFrame
        The data used for estimation.
    Y : pd.DataFrame
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    fe : pd.DataFrame
        Fixed effects.
    methods: list[str], optional
        Methods used to check for separation. One of fixed effects ("fe") or
        iterative rectifier ("ir"). Executes all methods by default.

    Returns
    -------
    list
        List of indices of observations that are removed due to separation.
    """
    valid_methods: dict[str, _SeparationMethod] = {
        "fe": _check_for_separation_fe,
        "ir": _check_for_separation_ir,
    }
    if methods is None:
        methods = list(valid_methods)

    invalid_methods = [method for method in methods if method not in valid_methods]
    if invalid_methods:
        raise ValueError(
            f"Invalid separation method. Expecting {list(valid_methods)}. Received {invalid_methods}"
        )

    separation_na: set[int] = set()
    for method in methods:
        separation_na = separation_na.union(
            valid_methods[method](fml=fml, data=data, Y=Y, X=X, fe=fe)
        )

    if separation_na:
        warnings.warn(
            f"{len(separation_na)!s} observations removed because of separation."
        )

    return list(separation_na)


class _SeparationMethod(Protocol):
    def __call__(
        self,
        fml: str,
        data: pd.DataFrame,
        Y: pd.DataFrame,
        X: pd.DataFrame,
        fe: pd.DataFrame,
    ) -> set[int]:
        """
        Check for separation.

        Parameters
        ----------
        fml : str
            The formula used for estimation.
        data : pd.DataFrame
            The data used for estimation.
        Y : pd.DataFrame
            Dependent variable.
        X : pd.DataFrame
            Independent variables.
        fe : pd.DataFrame
            Fixed effects.

        Returns
        -------
        set
            Set of indices of separated observations.
        """
        ...


def _check_for_separation_fe(
    fml: str, data: pd.DataFrame, Y: pd.DataFrame, X: pd.DataFrame, fe: pd.DataFrame
) -> set[int]:
    """
    Check for separation using the "fe" check.

    Parameters
    ----------
    fml : str
        The formula used for estimation.
    data : pd.DataFrame
        The data used for estimation.
    Y : pd.DataFrame
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    fe : pd.DataFrame
        Fixed effects.

    Returns
    -------
    set
        Set of indices of separated observations.
    """
    separation_na: set[int] = set()
    if fe is not None and not (Y > 0).all(axis=0).all():
        Y_help = (Y.iloc[:, 0] > 0).astype(int)

        # loop over all elements of fe
        for x in fe.columns:
            ctab = pd.crosstab(Y_help, fe[x])
            null_column = ctab.xs(0)
            # sep_candidate if
            # fixed effect level has only observations with Y > 0
            sep_candidate = (np.sum(ctab > 0, axis=0).values == 1) & (
                null_column > 0
            ).to_numpy().flatten()
            # droplist: list of levels to drop
            droplist = ctab.xs(0)[sep_candidate].index.tolist()

            # dropset: list of indices to drop
            if len(droplist) > 0:
                fe_in_droplist = fe[x].isin(droplist)
                dropset = set(fe[x][fe_in_droplist].index)
                separation_na = separation_na.union(dropset)

    return separation_na


def _check_for_separation_ir(
    fml: str,
    data: pd.DataFrame,
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: pd.DataFrame,
    tol: float = 1e-4,
    maxiter: int = 100,
) -> set[int]:
    """
    Check for separation using the "iterative rectifier" algorithm
    proposed by Correia et al. (2021). For details see http://arxiv.org/abs/1903.01633.

    Parameters
    ----------
    fml : str
        The formula used for estimation.
    data : pd.DataFrame
        The data used for estimation.
    Y : pd.DataFrame
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    fe : pd.DataFrame
        Fixed effects.
    tol : float
        Tolerance to detect separated observation. Defaults to 1e-4.
    maxiter : int
        Maximum number of iterations. Defaults to 100.

    Returns
    -------
    set
        Set of indices of separated observations.
    """
    # lazy load to avoid circular import
    fixest_module = import_module("pyfixest.estimation")
    feols = fixest_module.feols
    # initialize
    separation_na: set[int] = set()
    tmp_suffix = "_separationTmp"
    # build formula
    name_dependent, rest = re.split(r"\s*~\s*", fml, maxsplit=1)
    name_dependent_separation = "U"
    if name_dependent_separation in data.columns:
        name_dependent_separation += tmp_suffix

    fml_separation = f"{name_dependent_separation} ~ {rest}"

    dependent: pd.Series = data[name_dependent]
    is_interior = dependent > 0
    if is_interior.all():
        # no boundary sample, can exit
        return separation_na

    # initialize variables
    tmp: pd.DataFrame = pd.DataFrame(index=data.index)
    tmp["U"] = (dependent == 0).astype(float).rename("U")
    # weights
    N0 = (dependent > 0).sum()
    K = N0 / tol**2
    tmp["omega"] = pd.Series(
        np.where(dependent > 0, K, 1), name="omega", index=data.index
    )
    # combine data
    # TODO: avoid create new object?
    tmp = data.join(tmp, how="left", validate="one_to_one", rsuffix=tmp_suffix)
    # TODO: need to ensure that join doesn't create duplicated columns
    # assert not tmp.columns.duplicated().any()

    iteration = 0
    has_converged = False
    while iteration < maxiter:
        iteration += 1
        # regress U on X
        # TODO: check acceleration in ppmlhdfe's implementation: https://github.com/sergiocorreia/ppmlhdfe/blob/master/src/ppmlhdfe_separation_relu.mata#L135
        fitted = feols(fml_separation, data=tmp, weights="omega")
        tmp["Uhat"] = pd.Series(
            data=fitted.predict(), index=fitted._data.index, name="Uhat"
        )
        Uhat = tmp["Uhat"]
        # update when within tolerance of zero
        # need to be more strict below zero to avoid false positives
        within_zero = (Uhat > -0.1 * tol) & (Uhat < tol)
        Uhat.where(~(is_interior | within_zero.fillna(True)), 0, inplace=True)
        if (Uhat >= 0).all():
            # all separated observations have been identified
            has_converged = True
            break
        tmp.loc[~is_interior, "U"] = np.fmax(
            Uhat[~is_interior], 0
        )  # rectified linear unit (ReLU)

    if has_converged:
        separation_na = set(dependent[Uhat > 0].index)
    else:
        warnings.warn(
            "iterative rectivier separation check: maximum number of iterations reached before convergence"
        )

    return separation_na


def _fepois_input_checks(drop_singletons: bool, tol: float, maxiter: int):
    if not isinstance(drop_singletons, bool):
        raise TypeError("drop_singletons must be logical.")
    if not isinstance(tol, (int, float)):
        raise TypeError("tol must be numeric.")
    if tol <= 0 or tol >= 1:
        raise AssertionError("tol must be between 0 and 1.")
    if not isinstance(maxiter, int):
        raise TypeError("maxiter must be integer.")
    if maxiter <= 0:
        raise AssertionError("maxiter must be greater than 0.")
