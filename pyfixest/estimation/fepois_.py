import warnings
from importlib import import_module
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from pyfixest.errors import (
    NonConvergenceError,
    NotImplementedError,
)
from pyfixest.estimation.demean_ import demean
from pyfixest.estimation.feols_ import Feols
from pyfixest.utils.dev_utils import DataFrameType, _to_integer


class Fepois(Feols):
    """
    Estimate a Poisson regression model.

    Non user-facing class to estimate a Poisson regression model via Iterated
    Weighted Least Squares (IWLS).

    Inherits from the Feols class. Users should not directly instantiate this class,
    but rather use the [fepois()](/reference/estimation.fepois.qmd) function.
    Note that no demeaning is performed in this class: demeaning is performed in the
    [FixestMulti](/reference/estimation.fixest_multi.qmd) class (to allow for caching
    of demeaned variables for multiple estimation).

    The method implements the algorithm from Stata's `ppmlhdfe` module.

    Attributes
    ----------
    Y : np.ndarray
        Dependent variable, a two-dimensional numpy array.
    X : np.ndarray
        Independent variables, a two-dimensional numpy array.
    fe : np.ndarray
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
    fixef_tol: float, default = 1e-08.
        Tolerance level for the convergence of the demeaning algorithm.
    weights_name : Optional[str]
        Name of the weights variable.
    weights_type : Optional[str]
        Type of weights variable.
    """

    def __init__(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        fe: Union[np.ndarray, None],
        weights: np.ndarray,
        coefnames: list[str],
        drop_singletons: bool,
        collin_tol: float,
        maxiter: int = 25,
        tol: float = 1e-08,
        fixef_tol: float = 1e-08,
        weights_name: Optional[str] = None,
        weights_type: Optional[str] = None,
    ):
        super().__init__(
            Y=Y,
            X=X,
            weights=weights,
            coefnames=coefnames,
            collin_tol=collin_tol,
            weights_name=weights_name,
            weights_type=weights_type,
        )

        # input checks
        _fepois_input_checks(fe, drop_singletons, tol, maxiter)

        self.fe = fe
        self.maxiter = maxiter
        self.tol = tol
        self.fixef_tol = fixef_tol
        self._drop_singletons = drop_singletons
        self._method = "fepois"
        self.convergence = False

        if self.fe is not None:
            self._has_fixef = True
        else:
            self._has_fixef = False

        # check if Y is a weakly positive integer
        self._Y = _to_integer(self._Y)
        # check that self._Y is a weakly positive integer
        if np.any(self._Y < 0):
            raise ValueError(
                "The dependent variable must be a weakly positive integer."
            )

        self._support_crv3_inference = True
        self._support_iid_inference = True
        self._supports_cluster_causal_variance = False

        self._Y_hat_response = np.array([])
        self.deviance = None
        self._Xbeta = np.array([])

    def get_fit(self) -> None:
        """
        Fit a Poisson Regression Model via Iterated Weighted Least Squares (IWLS).

        Returns
        -------
        None

        Attributes
        ----------
        beta_hat : np.ndarray
            Estimated coefficients.
        Y_hat : np.ndarray
            Estimated dependent variable.
        u_hat : np.ndarray
            Estimated residuals.
        weights : np.ndarray
            Weights (from the last iteration of the IRLS algorithm).
        X : np.ndarray
            Demeaned independent variables (from the last iteration of the IRLS
            algorithm).
        Z : np.ndarray
            Demeaned independent variables (from the last iteration of the IRLS
            algorithm).
        Y : np.ndarray
            Demeaned dependent variable (from the last iteration of the IRLS
            algorithm).
        """
        _Y = self._Y
        _X = self._X
        _fe = self.fe
        _N = self._N
        _drop_singletons = self._drop_singletons
        _convergence = self.convergence  # False
        _maxiter = self.maxiter
        _iwls_maxiter = 25
        _tol = self.tol
        _fixef_tol = self.fixef_tol

        def compute_deviance(_Y: np.ndarray, mu: np.ndarray):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                deviance = (
                    2 * np.sum(np.where(_Y == 0, 0, _Y * np.log(_Y / mu)) - (_Y - mu))
                ).flatten()
            return deviance

        stop_iterating = False
        crit = 1

        for i in range(_maxiter):
            if stop_iterating:
                _convergence = True
                break
            if i == _maxiter:
                raise NonConvergenceError(
                    f"""
                    The IRLS algorithm did not converge with {_iwls_maxiter}
                    iterations. Try to increase the maximum number of iterations.
                    """
                )

            if i == 0:
                _mean = np.mean(_Y)
                mu = (_Y + _mean) / 2
                eta = np.log(mu)
                Z = eta + _Y / mu - 1
                reg_Z = Z.copy()
                last = compute_deviance(_Y, mu)

            else:
                # update w and Z
                Z = eta + _Y / mu - 1  # eq (8)
                reg_Z = Z.copy()  # eq (9)

            # tighten HDFE tolerance - currently not possible with PyHDFE
            # if crit < 10 * inner_tol:
            #    inner_tol = inner_tol / 10

            # Step 1: weighted demeaning
            ZX = np.concatenate([reg_Z, _X], axis=1)

            if _fe is not None:
                # ZX_resid = algorithm.residualize(ZX, mu)
                ZX_resid, success = demean(
                    x=ZX, flist=_fe, weights=mu.flatten(), tol=_fixef_tol
                )
                if success is False:
                    raise ValueError("Demeaning failed after 100_000 iterations.")
            else:
                ZX_resid = ZX

            Z_resid = ZX_resid[:, 0].reshape((_N, 1))  # z_resid
            X_resid = ZX_resid[:, 1:]  # x_resid

            # Step 2: estimate WLS
            WX = np.sqrt(mu) * X_resid
            WZ = np.sqrt(mu) * Z_resid

            XWX = WX.transpose() @ WX
            XWZ = WX.transpose() @ WZ

            delta_new = np.linalg.solve(XWX, XWZ)  # eq (10), delta_new -> reg_z
            resid = Z_resid - X_resid @ delta_new

            mu_old = mu.copy()
            # more updating
            eta = Z - resid
            mu = np.exp(eta)

            # same criterion as fixest
            # https://github.com/lrberge/fixest/blob/6b852fa277b947cea0bad8630986225ddb2d6f1b/R/ESTIMATION_FUNS.R#L2746
            deviance = compute_deviance(_Y, mu)
            crit = np.abs(deviance - last) / (0.1 + np.abs(last))
            last = deviance.copy()

            stop_iterating = crit < _tol

        self._beta_hat = delta_new.flatten()
        self._Y_hat_response = mu
        self._Y_hat_link = eta
        # (Y - self._Y_hat)
        # needed for the calculation of the vcov

        # updat for inference
        self._weights = mu_old
        # if only one dim
        if self._weights.ndim == 1:
            self._weights = self._weights.reshape((self._N, 1))

        self._u_hat = resid.flatten()

        self._Y = Z_resid
        self._X = X_resid
        self._Z = self._X
        self.deviance = deviance

        self._tZX = np.transpose(self._Z) @ self._X
        self._tZXinv = np.linalg.inv(self._tZX)
        self._Xbeta = eta

        self._scores = self._u_hat[:, None] * self._weights * X_resid
        self._hessian = XWX
        self._T = self._weights * X_resid

        if _convergence:
            self._convergence = True

    def predict(
        self, newdata: Optional[DataFrameType] = None, type: str = "link"
    ) -> np.ndarray:
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
        type : str, optional
            The type of prediction to be computed.
            Can be either "response" (default) or "link".
            If type="response", the output is at the level of the response variable,
            i.e., it is the expected predictor E(Y|X).
            If "link", the output is at the level of the explanatory variables,
            i.e., the linear predictor X @ beta.

        Returns
        -------
        np.ndarray
            A flat array with the predicted values of the regression model.
        """
        _Xbeta = self._Xbeta
        _has_fixef = self._has_fixef

        if _has_fixef:
            raise NotImplementedError(
                "Prediction with fixed effects is not yet implemented for Poisson regression."
            )
        if newdata is not None:
            raise NotImplementedError(
                "Prediction with function argument `newdata` is not yet implemented for Poisson regression."
            )

        if type not in ["response", "link"]:
            raise ValueError("type must be one of 'response' or 'link'.")

        y_hat = super().predict(newdata=newdata)
        if type == "link":
            y_hat = np.exp(y_hat)

        return y_hat


def _check_for_separation(
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
    Y : pd.DataFrame
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    fe : pd.DataFrame
        Fixed effects.
    method: list[str], optional
        Method to check for separation. Executes ``fe`` and ``ir`` by default.

    Returns
    -------
    list
        List of indices of observations that are removed due to separation.
    """
    valid_methods: dict[
        str, Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], set[int]]
    ] = {
        "fe": _check_for_separation_fe,
        "ir": _check_for_separation_ir,
    }
    if methods is None:
        methods = list(valid_methods)
    invalid_methods = [method for method in methods if method not in valid_methods]
    if invalid_methods:
        raise ValueError(
            f"Invalid separation method. Expecting {valid_methods}. Got {invalid_methods}"
        )

    separation_na: set[int] = set()
    for method in methods:
        separation_na = separation_na.union(valid_methods[method](Y, X, fe))

    return list(separation_na)


def _check_for_separation_fe(
    Y: pd.DataFrame, X: pd.DataFrame, fe: pd.DataFrame
) -> set[int]:
    """
    Check for separation using the "fe" check.

    Parameters
    ----------
    Y : pd.DataFrame
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    fe : pd.DataFrame
        Fixed effects.

    Returns
    -------
    set
        Set of indices of observations that are removed due to separation.
    """
    separation_na: set[int] = set()
    if not (Y > 0).all(axis=0).all():
        Y_help = (Y > 0).astype(int).squeeze()

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
    Y: pd.DataFrame, X: pd.DataFrame, fe: pd.DataFrame
) -> set[int]:
    """
    Check for separation using the "iterative rectifier" check.

    For details see http://arxiv.org/abs/1903.01633

    Parameters
    ----------
    Y : pd.DataFrame
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    fe : pd.DataFrame
        Fixed effects.

    Returns
    -------
    set
        Set of indices of observations that are removed due to separation.
    """
    tol = 1e-5
    maxiter = 10_000
    # lazy loading to avoid circular import
    fixest_module = import_module("pyfixest.estimation")
    feols = getattr(fixest_module, "feols")

    U = (Y == 0).astype(int)
    N0 = (Y > 0).sum()
    K = N0 / tol**2
    omega = U.where(U > 0, K)
    data = pd.concat([U, X, fe], axis=1)
    fml = f"U ~ {'+'.join(X.columns)}"

    iter = 0
    while iter < maxiter:
        iter += 1
        # regress U on X
        # TODO: check acceleration in ppmlhdfe's implementation: https://github.com/sergiocorreia/ppmlhdfe/blob/master/src/ppmlhdfe_separation_relu.mata#L135
        Uhat = feols(fml, data=data, weights=omega).predict()
        Uhat = np.where(Uhat.abs() < tol, 0, Uhat)
        if (Uhat >= 0).all():
            # all separated observations have been identified
            break

        data["U"] = np.fmax(Uhat, 0)  # rectified linear unit (ReLU)

    separation_na = set(Y[Uhat > 0].index)
    return separation_na


def _fepois_input_checks(
    fe: Union[np.ndarray, None], drop_singletons: bool, tol: float, maxiter: int
):
    """
    Perform input checks for Fepois constructor arguments.

    Parameters
    ----------
    fe : Union[np.ndarray, None]
        Fixed effects. None if no fixed effects are used.
    drop_singletons : bool
        Whether to drop singleton fixed effects.
    tol : float
        Tolerance level for convergence check.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    None
    """
    # fe must be np.array of dimension 2 or None
    if fe is not None:
        if not isinstance(fe, np.ndarray):
            raise AssertionError("fe must be a numpy array.")
        if fe.ndim != 2:
            raise AssertionError("fe must be a numpy array of dimension 2.")
    # drop singletons must be logical
    if not isinstance(drop_singletons, bool):
        raise TypeError("drop_singletons must be logical.")
    # tol must be numeric and between 0 and 1
    if not isinstance(tol, (int, float)):
        raise TypeError("tol must be numeric.")
    if tol <= 0 or tol >= 1:
        raise AssertionError("tol must be between 0 and 1.")
    # maxiter must be integer and greater than 0
    if not isinstance(maxiter, int):
        raise TypeError("maxiter must be integer.")
    if maxiter <= 0:
        raise AssertionError("maxiter must be greater than 0.")
