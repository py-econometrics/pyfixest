from typing import Optional, Union

import numpy as np
import pandas as pd

from pyfixest.estimation.feols_ import Feols, _drop_multicollinear_variables


class Feiv(Feols):
    """
    Non user-facing class to estimate an IV model using a 2SLS estimator.

    Inherits from the Feols class. Users should not directly instantiate this class,
    but rather use the [feols()](/reference/estimation.feols.qmd) function. Note that
    no demeaning is performed in this class: demeaning is performed in the
    [FixestMulti](/reference/estimation.fixest_multi.qmd) class (to allow for caching
    of demeaned variables for multiple estimation).

    Parameters
    ----------
    Y : np.ndarray
        Dependent variable, a two-dimensional np.array.
    X : np.ndarray
        Independent variables, a two-dimensional np.array.
    endgvar : np.ndarray
        Endogenous Indenpendent variables, a two-dimensional np.array.
    Z : np.ndarray
        Instruments, a two-dimensional np.array.
    weights : np.ndarray
        Weights, a one-dimensional np.array.
    coefnames_x : list
        Names of the coefficients of X.
    coefnames_z : list
        Names of the coefficients of Z.
    collin_tol : float
        Tolerance for collinearity check.
    solver: str, default is 'np.linalg.solve'
        Solver to use for the estimation. Alternative is 'np.linalg.lstsq'.
    weights_name : Optional[str]
        Name of the weights variable.
    weights_type : Optional[str]
        Type of the weights variable. Either "aweights" for analytic weights
        or "fweights" for frequency weights.

    Attributes
    ----------
    _Z : np.ndarray
        Processed instruments after handling multicollinearity.
    _coefnames_z : list
        Names of coefficients for Z after handling multicollinearity.
    _collin_vars_z : list
        Variables identified as collinear in Z.
    _collin_index_z : list
        Indices of collinear variables in Z.
    _is_iv : bool
        Indicator if instrumental variables are used.
    _support_crv3_inference : bool
        Indicator for supporting CRV3 inference.
    _support_iid_inference : bool
        Indicator for supporting IID inference.
    _tZX : np.ndarray
        Transpose of Z times X.
    _tXZ : np.ndarray
        Transpose of X times Z.
    _tZy : np.ndarray
        Transpose of Z times Y.
    _tZZinv : np.ndarray
        Inverse of transpose of Z times Z.
    _beta_hat : np.ndarray
        Estimated regression coefficients.
    _Y_hat_link : np.ndarray
        Predicted values of the regression model.
    _u_hat : np.ndarray
        Residuals of the regression model.
    _scores : np.ndarray
        Scores used in the regression.
    _hessian : np.ndarray
        Hessian matrix used in the regression.
    _bread : np.ndarray
        Bread matrix used in the regression.
    _pi_hat : np.ndarray
        Estimated coefficients from 1st stage regression
    _X_hat : np.ndarray
        Predicted values of the 1st stage regression
    _v_hat : np.ndarray
        Residuals of the 1st stage regression
    _model_1st_stage : Any
        feols object of 1st stage regression.
        It contains various results and diagnostics
        from the fixed effects OLS regression.
    _endogvar_1st_stage : np.ndarray
        Unweihgted Endogenous independent variable vector
    _Z_1st_stage : np.ndarray
        Unweighted instruments vector to be used for 1st stage
    _non_exo_instruments : list
        List of instruments name excluding exogenous independent vars.
    __p_iv : scalar
        Number of instruments listed in _non_exo_instruments
    _iv_loc_first  : scalar
        First index of insturments in 1st stage regression
    _f_stat_1st_stage : scalar
        F-statistics of First Stage regression for evaluation of IV weakness.
        The computed F-statistics test the following null hypothesis :
        # H0 : β_{z_1} = 0 & ... & β_{z_{p_iv}} = 0 where z_1, ..., z_{p_iv}
        # are the instrument variables
        # H1 : H0 does not hold
        Note that this F-statistics is adjusted to heteroskedasticity /
        clusters if users set specification of variance-covariance matrix type
    _eff_F : scalar
        Effective F-statistics of first stage regression as in Olea and Pflueger 2013


    Raises
    ------
    ValueError
        If Z is not a two-dimensional array.

    """

    # Constructor and methods implementation...

    def __init__(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        endogvar: np.ndarray,
        Z: np.ndarray,
        weights: np.ndarray,
        coefnames_x: list,
        coefnames_z: list,
        collin_tol: float,
        weights_name: Optional[str],
        weights_type: Optional[str],
        solver: str = "np.linalg.solve",
    ) -> None:
        super().__init__(
            Y=Y,
            X=X,
            weights=weights,
            coefnames=coefnames_x,
            collin_tol=collin_tol,
            weights_name=weights_name,
            weights_type=weights_type,
            solver=solver,
        )

        if self._has_weights:
            w = np.sqrt(weights)
            Z_1st_stage = Z
            endogvar_1st_stage = endogvar
            Z = Z * w
            endogvar = endogvar * w
        else:
            w = 1
            Z_1st_stage = Z
            endogvar_1st_stage = endogvar
            Z = Z * w
            endogvar = endogvar * w

        # check if Z is two dimensional array
        if len(Z.shape) != 2:
            raise ValueError("Z must be a two-dimensional array")

        # handle multicollinearity in Z
        (
            self._Z,
            self._coefnames_z,
            self._collin_vars_z,
            self._collin_index_z,
        ) = _drop_multicollinear_variables(Z, coefnames_z, self._collin_tol)

        self._is_iv = True

        self._support_crv3_inference = False
        self._support_iid_inference = True
        self._supports_cluster_causal_variance = False
        self._endogvar = endogvar
        self._endogvar_1st_stage = endogvar_1st_stage
        self._Z_1st_stage = Z_1st_stage

    def extract_indices_IVs(self):
        """Extract indices of instrument variales in the first stage regression."""
        indices = []
        for instrument in self._non_exo_instruments:
            if instrument in self._coefnames_z:
                index = self._coefnames_z.index(instrument)
                indices.append(index)
        return indices

    def get_fit(self) -> None:
        """Fit a IV model using a 2SLS estimator."""
        _X = self._X
        _Z = self._Z
        _Y = self._Y

        _solver = self._solver

        # Start Second Stage
        self._tZX = _Z.T @ _X
        self._tXZ = _X.T @ _Z
        self._tZy = _Z.T @ _Y
        self._tZZinv = np.linalg.inv(_Z.T @ _Z)

        H = self._tXZ @ self._tZZinv
        A = H @ self._tZX
        B = H @ self._tZy
        self._beta_hat = self.solve_ols(A, B, _solver)

        # Predicted values and residuals
        self._Y_hat_link = self._X @ self._beta_hat
        self._u_hat = self._Y.flatten() - self._Y_hat_link.flatten()

        # Compute scores and hessian
        self._scores = self._Z * self._u_hat[:, None]
        self._hessian = self._Z.T @ self._Z

        # Compute bread matrix
        D = np.linalg.inv(self._tXZ @ self._tZZinv @ self._tZX)
        self._bread = H.T @ D @ H

    def first_stage(self) -> None:
        """Implement First stage regression."""
        from pyfixest.estimation.estimation import feols

        #  Start First Stage
        _X = self._X
        _Z = self._Z
        _Y = self._Y

        # Store names of instruments from Z matrix
        self._non_exo_instruments = list(set(self._coefnames_z) - set(self._coefnames))

        # Prepare your data
        # Select the columns from self._data that match the variable names
        data = pd.DataFrame(self._Z_1st_stage, columns=self._coefnames_z)

        # Store instrument variable matrix for future use
        _Z_iv = data[self._non_exo_instruments]
        self._Z_iv = _Z_iv.to_numpy()
        # Dynamically create the formula string
        # Extract names of fixed effect

        data["Y"] = self._endogvar_1st_stage
        independent_vars = " + ".join(
            data.columns[:-1]
        )  # All columns except the last one ('Y')

        # Set fix effects, cluster, and weight options to be passed to feols.

        if self._has_fixef:
            FE_vars = self._fixef.split("+")
            data[FE_vars] = self._data[FE_vars]
            formula = f"Y ~ {independent_vars} | {self._fixef}"
        else:
            formula = f"Y ~ {independent_vars}"

        # Type hint to reflect that vcov_detail can be either a dict or a str
        vcov_detail: Union[dict[str, str], str]

        if self._is_clustered:
            a = self._clustervar[0]
            vcov_detail = {self._vcov_type_detail: a}
            data[a] = self._data[a]
        else:
            vcov_detail = self._vcov_type_detail

        if self._has_weights:
            data["weights"] = self._weights
            weight_detail = "weights"
        else:
            weight_detail = None

        # Do first stage regression

        model1 = feols(
            fml=formula,
            data=data,
            vcov=vcov_detail,
            weights=weight_detail,
            collin_tol=1e-10,
        )

        # Ensure model1 is of type Feols
        if isinstance(model1, Feols):
            # Store the first stage coefficients
            self._pi_hat = model1._beta_hat

            # Use fitted values from the first stage
            self._X_hat = model1._Y_hat_link

            # Residuals from the first stage
            self._v_hat = model1._u_hat

            # Store 1st stage model for further use
            self._model_1st_stage = model1

        else:
            raise TypeError("The first stage model must be of type Feols")

        self.IV_weakness_test(["f_stat"])

    def IV_Diag(self, statistics: Optional[list[str]] = None):
        """Implement IV diagnostic tests.

        Notes
        -----
        This method covers diagnostic tests related with IV regression.
        We currently have IV weak tests only. More test will be updated
        in future updates!

        Parameters
        ----------
        statistics : list[str], optional
            List of IV diagnostic statistics

        Example
        -------
        The following is an example usage of this method:

            ```{python}

            import numpy as np
            import pandas as pd
            from pyfixest.estimation.estimation import feols

            # Set random seed for reproducibility
            np.random.seed(1)

            # Number of observations
            n = 1000

            # Simulate the data
            # Instrumental variable
            z = np.random.binomial(1, 0.5, size=n)
            z2 = np.random.binomial(1, 0.5, size=n)

            # Endogenous variable
            d = 0.5 * z + 1.5 * z2 + np.random.normal(size=n)

            # Control variables
            c1 = np.random.normal(size=n)
            c2 = np.random.normal(size=n)

            # Outcome variable
            y = 1.0 + 1.5 * d + 0.8 * c1 + 0.5 * c2 + np.random.normal(size=n)

            # Cluster variable
            cluster = np.random.randint(1, 50, size=n)
            weights = np.random.uniform(1, 3, size=n)

            # Create a DataFrame
            data = pd.DataFrame({
                'd': d,
                'y': y,
                'z': z,
                'z2': z2,
                'c1': c1,
                'c2': c2,
                'cluster': cluster,
                'weights': weights
            })

            vcov_detail = "iid"

            # Fit OLS model
            fit_ols = feols("y ~ 1 + d + c1 + c2", data=data, vcov=vcov_detail)

            # Fit IV model
            fit_iv = feols("y ~ 1 + c1 + c2 | d ~ z", data=data,
                     vcov=vcov_detail,
                     weights="weights")
            fit_iv.first_stage()
            F_stat_pf = fit_iv._f_stat_1st_stage
            fit_iv.IV_Diag()
            F_stat_eff_pf = fit_iv._eff_F

            print("(Unadjusted) F stat :", F_stat_pf)
            print("Effective F stat :", F_stat_eff_pf)

            # The example above generates the following results
            # (Unadjusted) F stat : 52.81535560457482
            # Effective F stat : 48.661542741328205
        """
        # Set default statistics
        iv_diag_stat = ["f_stat", "effective_f"]

        # Set statistics allowed in the current version
        iv_diag_stat_allowed = ["f_stat", "effective_f"]

        # Check whether there is unsupported statistics.
        if statistics:
            invalid_stats = [
                stat for stat in statistics if stat not in iv_diag_stat_allowed
            ]

            if invalid_stats:
                raise ValueError(
                    f"Statistics not supported: {invalid_stats}."
                    f"You should specify from the following list of statistics {iv_diag_stat_allowed}"
                )

            iv_diag_stat += statistics

        self.IV_weakness_test(iv_diag_stat)

    def IV_weakness_test(self, iv_diag_statistics: Optional[list[str]] = None) -> None:
        """Implement IV weakness test (F-test).

        This method covers hetero-robust and clustered-robust F statistics.
        It produces two statistics:

        - self._f_stat_1st_stage: F statistics of first stage regression
        - self._eff_F: Effective F statistics (Olea and Pflueger 2013)
                       of first stage regression

        Notes
        -----
        "self._f_stat_1st_stage" is adjusted to the specification of vcov.
        If vcov_detail = "iid", F statistics is not adjusted,
        otherwise it is always adjusted.

        Parameters
        ----------
        iv_diag_statistics : list, optional
            List of IV weakness statistics

        """
        iv_diag_statistics = iv_diag_statistics or []

        if "f_stat" in iv_diag_statistics:
            self._p_iv = len(self._non_exo_instruments)

            # Create an identity matrix of size p_iv by p_iv
            # Pad the identity matrix with zeros to make it of size p_iv by k
            p_iv = self._p_iv  # number of IVs
            k = (
                self._model_1st_stage._k
            )  # number of estimated coefficients of 1st stage

            iv_loc = self.extract_indices_IVs()  # Extract indices of IVs

            self._iv_loc_first = np.min(iv_loc)
            iv_loc_first = self._iv_loc_first

            # Generate matrix R that tests the following;
            # H0 : \beta_{z_1} = 0 & ... & \beta_{z_{p_iv}} = 0
            #      where z_1, ..., z_{p_iv} are the instrument variables
            # H1 : H0 does not hold

            # Pad identity matrix to implement wald-test
            identity_matrix = np.eye(p_iv)
            R = np.zeros((p_iv, k))
            R[:, iv_loc_first : p_iv + iv_loc_first] = identity_matrix

            self._model_1st_stage.wald_test(R=R)
            self._f_stat_1st_stage = self._model_1st_stage._f_statistic
            self._p_value_1st_stage = self._model_1st_stage._p_value

        if "effective_f" in iv_diag_statistics:
            self.eff_F()

    def eff_F(self) -> None:
        """Compute Effective F stat (Olea and Pflueger 2013)."""
        # If vcov is iid, redo first stage regression
        if self._vcov_type_detail == "iid":
            self._vcov_type_detail = "hetero"
            self._model_1st_stage.vcov("hetero")

        # Compute Effective F stat by Olea and Pflueger 2013
        # 1. Extract First Stage Coefficients and Variance-Covariance Matrix:
        #   Extract the coefficients for the instrument z
        #   from the first stage regression.
        #   Extract the robust variance-covariance matrix of these coefficients.
        #   Extract the instrument matrix.
        # 2. Compute the Instrument Matrix:
        #   Construct the instrument matrix Q_{zz} = Z.T x Z
        # 3. Compute the Effective F-statistic:
        #   F_{eff} = π.T Q_{zz} π / trance(ΣQ_{zz})

        # Extract coefficients for the non-exogenous instruments

        pi_hat = np.array(
            [
                self._model_1st_stage.coef()[instrument]
                for instrument in self._non_exo_instruments
            ]
        )

        Z = self._Z_iv

        # Calculate the cross-product of the instrument matrix
        Q_zz = Z.T @ Z

        # Extract the robust variance-covariance matrix
        vcv = self._model_1st_stage._vcov

        # Map the instrument names to their indices in the parameter list
        # Number of rows/columns in vcv

        # Extract the submatrix

        iv_pos = range(self._iv_loc_first, self._p_iv + self._iv_loc_first)

        # iv_pos = range(0, self._p_iv) if self._has_fixef else range(1, self._p_iv + 1)

        Sigma = vcv[np.ix_(iv_pos, iv_pos)]

        # Calculate the effective F-statistic
        self._eff_F = (pi_hat.T @ Q_zz @ pi_hat) / np.sum(np.diag(Sigma @ Q_zz))
