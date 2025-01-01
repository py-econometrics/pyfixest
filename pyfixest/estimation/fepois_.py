import warnings
from importlib import import_module
from typing import Literal, Optional, Protocol, Union

import numpy as np
import pandas as pd

from pyfixest.errors import (
    NonConvergenceError,
)
from pyfixest.estimation.demean_ import demean
from pyfixest.estimation.feols_ import Feols, PredictionType
from pyfixest.estimation.FormulaParser import FixestFormula
from pyfixest.estimation.solvers import solve_ols
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
    solver: Literal["np.linalg.lstsq", "np.linalg.solve", "scipy.sparse.linalg.lsqr", "jax"],
        default is 'np.linalg.solve'. Solver to use for the estimation.
    demeaner_backend: Literal["numba", "jax"]
        The backend used for demeaning.
    fixef_tol: float, default = 1e-08.
        Tolerance level for the convergence of the demeaning algorithm.
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
        ssc_dict: dict[str, Union[str, bool]],
        drop_singletons: bool,
        drop_intercept: bool,
        weights: Optional[str],
        weights_type: Optional[str],
        collin_tol: float,
        fixef_tol: float,
        lookup_demeaned_data: dict[str, pd.DataFrame],
        tol: float,
        maxiter: int,
        solver: Literal[
            "np.linalg.lstsq", "np.linalg.solve", "scipy.sparse.linalg.lsqr", "jax"
        ] = "np.linalg.solve",
        demeaner_backend: Literal["numba", "jax"] = "numba",
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        sample_split_var: Optional[str] = None,
        sample_split_value: Optional[Union[str, int]] = None,
        separation_check: Optional[list[str]] = None,
    ):
        super().__init__(
            FixestFormula,
            data,
            ssc_dict,
            drop_singletons,
            drop_intercept,
            weights,
            weights_type,
            collin_tol,
            fixef_tol,
            lookup_demeaned_data,
            solver,
            demeaner_backend,
            store_data,
            copy_data,
            lean,
            sample_split_var,
            sample_split_value,
        )

        # input checks
        _fepois_input_checks(drop_singletons, tol, maxiter)

        self.maxiter = maxiter
        self.tol = tol
        self._method = "fepois"
        self.convergence = False
        self.separation_check = separation_check

        self._support_crv3_inference = True
        self._support_iid_inference = True
        self._supports_cluster_causal_variance = False
        self._support_decomposition = False

        self._Y_hat_response = np.array([])
        self.deviance = None
        self._Xbeta = np.array([])

    def prepare_model_matrix(self):
        "Prepare model inputs for estimation."
        super().prepare_model_matrix()

        # check if Y is a weakly positive integer
        self._Y = _to_integer(self._Y)
        # check that self._Y is a weakly positive integer
        if np.any(self._Y < 0):
            raise ValueError(
                "The dependent variable must be a weakly positive integer."
            )

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
            self._N = self._Y.shape[0]

            self.na_index = np.concatenate([self.na_index, np.array(na_separation)])
            self.n_separation_na = len(na_separation)

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
        _fe = self._fe
        _N = self._N
        _convergence = self.convergence  # False
        _maxiter = self.maxiter
        _tol = self.tol
        _fixef_tol = self._fixef_tol
        _solver = self._solver

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
                    The IRLS algorithm did not converge with {_maxiter}
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

            delta_new = solve_ols(XWX, XWZ, _solver).reshape(
                (-1, 1)
            )  # eq (10), delta_new -> reg_z
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
        self._Y_hat_response = mu.flatten()
        self._Y_hat_link = eta.flatten()
        # (Y - self._Y_hat)
        # needed for the calculation of the vcov

        # update for inference
        self._weights = mu_old
        self._irls_weights = mu
        # if only one dim
        if self._weights.ndim == 1:
            self._weights = self._weights.reshape((self._N, 1))

        self._u_hat = (WZ - WX @ delta_new).flatten()
        self._u_hat_working = resid
        self._u_hat_response = self._Y - np.exp(eta)

        self._Y = WZ
        self._X = WX
        self._Z = self._X
        self.deviance = deviance

        self._tZX = np.transpose(self._Z) @ self._X
        self._tZXinv = np.linalg.inv(self._tZX)
        self._Xbeta = eta

        self._scores = self._u_hat[:, None] * self._X
        self._hessian = XWX

        if _convergence:
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

    def predict(
        self,
        newdata: Optional[DataFrameType] = None,
        atol: float = 1e-6,
        btol: float = 1e-6,
        type: PredictionType = "link",
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

        Returns
        -------
        np.ndarray
            A flat array with the predicted values of the regression model.
        """
        if self._has_fixef:
            raise NotImplementedError(
                "Prediction with fixed effects is not yet implemented for Poisson regression."
            )
        if newdata is not None:
            raise NotImplementedError(
                "Prediction with function argument `newdata` is not yet implemented for Poisson regression."
            )
        return super().predict(newdata=newdata, type=type, atol=atol, btol=btol)


def _check_for_separation(
    fml: str,
    data: pd.DataFrame,
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: pd.DataFrame,
    methods: Optional[list[str]] = None,
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
    name_dependent, rest = fml.split("~")
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
        tmp["Uhat"] = pd.Series(fitted.predict(), index=fitted._data.index, name="Uhat")
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
    """
    Perform input checks for Fepois constructor arguments.

    Parameters
    ----------
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
