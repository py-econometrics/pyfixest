import pandas as pd
import numpy as np

from pyfixest.estimation import feols
from pyfixest.exceptions import NotImplementedError
from pyfixest.model_matrix_fixest import model_matrix_fixest

from abc import ABC, abstractmethod
from formulaic import model_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, Union, List
import warnings


def event_study(
    data,
    yname,
    idname,
    tname,
    gname,
    xfml=None,
    estimator="twfe",
    att=True,
    cluster="idname",
):
    """
    Estimate a treatment effect using an event study design. If estimator is "twfe", then
    the treatment effect is estimated using the two-way fixed effects estimator. If estimator
    is "did2s", then the treatment effect is estimated using Gardner's two-step DID2S estimator.
    Other estimators are work in progress, please contact the package author if you are interested
    / need other estimators (i.e. Mundlak, Sunab, Imputation DID or Projections).

    Args:
        data: The DataFrame containing all variables.
        yname: The name of the dependent variable.
        idname: The name of the id variable.
        tname: Variable name for calendar period.
        gname: unit-specific time of initial treatment.
        xfml: The formula for the covariates.
        estimator: The estimator to use. Options are "did2s" and "twfe".
        att: Whether to estimate the average treatment effect on the treated (ATT) or the
            canonical event study design with all leads and lags. Default is True.
    Returns:
        A fitted model object of class feols.
    Examples:
        >>> from pyfixest.experimental.did import event_study, did2s
        >>> from pyfixest.estimation import feols
        >>> from pyfixest.summarize import etable, summary
        >>> import pandas as pd
        >>> import numpy as np
        >>> df_het = pd.read_csv("pyfixest/experimental/data/df_het.csv")
        >>> fit_twfe = event_study(
        >>>     data = df_het,
        >>>     yname = "dep_var",
        >>>     idname= "state",
        >>>     tname = "year",
        >>>     gname = "g",
        >>>     estimator = "twfe"
        >>> )
        >>> fit_did2s = event_study(
        >>>     data = df_het,
        >>>     yname = "dep_var",
        >>>     idname= "state",
        >>>     tname = "year",
        >>>     gname = "g",
        >>>     estimator = "did2s"
        >>> )
    """

    assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
    assert isinstance(yname, str), "yname must be a string"
    assert isinstance(idname, str), "idname must be a string"
    assert isinstance(tname, str), "tname must be a string"
    assert isinstance(gname, str), "gname must be a string"
    assert isinstance(xfml, str) or xfml is None, "xfml must be a string or None"
    assert isinstance(estimator, str), "estimator must be a string"
    assert isinstance(att, bool), "att must be a boolean"
    assert isinstance(cluster, str), "cluster must be a string"
    assert cluster == "idname", "cluster must be idname"

    if cluster == "idname":
        cluster = idname
    else:
        raise NotImplementedError(
            "Clustering by a variable of your choice is not yet supported."
        )

    if estimator == "did2s":
        did2s = DID2S(
            data=data,
            yname=yname,
            idname=idname,
            tname=tname,
            gname=gname,
            xfml=xfml,
            att=att,
            cluster=cluster,
        )
        fit, did2s._first_u, did2s._second_u = did2s.estimate()
        vcov, _G = did2s.vcov()
        fit._vcov = vcov
        fit._G = _G
        fit._vcov_type = "CRV1"
        fit._vcov_type_detail = "CRV1 (GMM)"
        fit._method = "did2s"

    elif estimator == "twfe":
        twfe = TWFE(
            data=data,
            yname=yname,
            idname=idname,
            tname=tname,
            gname=gname,
            xfml=xfml,
            att=att,
            cluster=cluster,
        )
        fit = twfe.estimate()
        vcov = fit.vcov(vcov={"CRV1": twfe._idname})
        fit._method = "twfe"

    else:
        raise Exception("Estimator not supported")

    # update inference with vcov matrix
    fit.get_inference()

    return fit


class DID(ABC):
    @abstractmethod
    def __init__(self, data, yname, idname, tname, gname, xfml, att, cluster):
        """
        Args:
            data: The DataFrame containing all variables.
            yname: The name of the dependent variable.
            idname: The name of the id variable.
            tname: Variable name for calendar period. Must be an integer in the format YYYYMMDDHHMMSS, i.e. it must be
                   possible to compare two dates via '>'. Date time variables are currently not accepted.
            gname: unit-specific time of initial treatment. Must be an integer in the format YYYYMMDDHHMMSS, i.e. it must be
                   possible to compare two dates via '>'. Date time variables are currently not accepted. Never treated units
                   must have a value of 0.
            xfml: The formula for the covariates.
            estimator: The estimator to use. Options are "did2s".
            att: Whether to estimate the average treatment effect on the treated (ATT) or the
                canonical event study design with all leads and lags. Default is True.
            cluster: The name of the cluster variable.
        Returns:
            None
        """

        # do some checks here

        self._data = data.copy()
        self._yname = yname
        self._idname = idname
        self._tname = tname
        self._gname = gname
        self._xfml = xfml
        self._att = att
        self._cluster = cluster

        if self._xfml is not None:
            raise NotImplementedError("Covariates are not yet supported.")
        if self._att is False:
            raise NotImplementedError(
                "Event study design with leads and lags is not yet supported."
            )

        # check if idname, tname and gname are in data
        # if self._idname not in self._data.columns:
        #    raise ValueError(f"The variable {self._idname} is not in the data.")
        # if self._tname not in self._data.columns:
        #    raise ValueError(f"The variable {self._tname} is not in the data.")
        # if self._gname not in self._data.columns:
        #    raise ValueError(f"The variable {self._gname} is not in the data.")

        # check if tname and gname are of type int (either int 64, 32, 8)
        if self._data[self._tname].dtype not in ["int64", "int32", "int8"]:
            raise ValueError(
                f"The variable {self._tname} must be of type int64, and more specifically, in the format YYYYMMDDHHMMSS."
            )
        if self._data[self._gname].dtype not in ["int64", "int32", "int8"]:
            raise ValueError(
                f"The variable {self._gname} must be of type int64, and more specifically, in the format YYYYMMDDHHMMSS."
            )

        # check if there is a never treated unit
        if 0 not in self._data[self._gname].unique():
            raise ValueError(f"There must be at least one unit that is never treated.")

        # create a treatment variable
        self._data["ATT"] = (self._data[self._tname] >= self._data[self._gname]) * (
            self._data[self._gname] > 0
        )

    @abstractmethod
    def estimate(self):
        pass

    @abstractmethod
    def vcov(self):
        pass

    # @abstractmethod
    # def aggregate(self):
    #    pass


class TWFE(DID):

    """
    Estimate a Difference-in-Differences model using the two-way fixed effects estimator.
    """

    def __init__(self, data, yname, idname, tname, gname, xfml, att, cluster):
        """
        Args:
            data: The DataFrame containing all variables.
            yname: The name of the dependent variable.
            idname: The name of the id variable.
            tname: Variable name for calendar period. Must be an integer in the format YYYYMMDDHHMMSS, i.e. it must be
                   possible to compare two dates via '>'. Date time variables are currently not accepted.
            gname: unit-specific time of initial treatment. Must be an integer in the format YYYYMMDDHHMMSS, i.e. it must be
                     possible to compare two dates via '>'. Date time variables are currently not accepted. Never treated units
                     must have a value of 0.
            xfml: The formula for the covariates.
            att: Whether to estimate the average treatment effect on the treated (ATT) or the
                canonical event study design with all leads and lags. Default is True.
            cluster (str): The name of the cluster variable.
        Returns:
            None
        """

        super().__init__(data, yname, idname, tname, gname, xfml, att, cluster)

        self._estimator = "twfe"

        if self._xfml is not None:
            self._fml = f"{yname} ~ ATT + {xfml} | {idname} + {tname}"
        else:
            self._fml = f"{yname} ~ ATT | {idname} + {tname}"

    def estimate(self):
        _fml = self._fml
        _data = self._data

        fit = feols(fml=_fml, data=_data)
        self._fit = fit

        return fit

    def vcov(self):
        """
        Method not needed. The vcov matrix is calculated via the `Feols` object.
        """

        pass


class DID2S(DID):

    """
    Difference-in-Differences estimation using Gardner's two-step DID2S estimator (2021).

    Multiple parts of this code are direct translations of Kyle Butt's R code `did2s`, published
    under MIT license: https://github.com/kylebutts/did2s/tree/main.
    """

    def __init__(self, data, yname, idname, tname, gname, xfml, att, cluster):
        """
        Args:
            data: The DataFrame containing all variables.
            yname: The name of the dependent variable.
            idname: The name of the id variable.
            tname: Variable name for calendar period. Must be an integer in the format YYYYMMDDHHMMSS, i.e. it must be
                   possible to compare two dates via '>'. Date time variables are currently not accepted.
            gname: unit-specific time of initial treatment. Must be an integer in the format YYYYMMDDHHMMSS, i.e. it must be
                     possible to compare two dates via '>'. Date time variables are currently not accepted. Never treated units
                     must have a value of 0.
            xfml: The formula for the covariates.
            att: Whether to estimate the average treatment effect on the treated (ATT) or the
                canonical event study design with all leads and lags. Default is True.
            cluster (str): The name of the cluster variable.
        Returns:
            None
        """

        super().__init__(data, yname, idname, tname, gname, xfml, att, cluster)

        self._estimator = "did2s"

        if self._xfml is not None:
            self._fml1 = f" ~ {xfml} | {idname} + {tname}"
            self._fml2 = f" ~ 0 + ATT + {xfml}"
        else:
            self._fml1 = f" ~ 0 | {idname} + {tname}"
            self._fml2 = f" ~ 0 + ATT"

    def estimate(self):
        """
        Args:
            data (pd.DataFrame): The DataFrame containing all variables.
            yname (str): The name of the dependent variable.
            _first_stage (str): The formula for the first stage.
            _second_stage (str): The formula for the second stage.
            treatment (str): The name of the treatment variable.
        Returns:
            tba
        """

        return _did2s_estimate(
            data=self._data,
            yname=self._yname,
            _first_stage=self._fml1,
            _second_stage=self._fml2,
            treatment="ATT",
        )  # returns triple Feols, first_u, second_u

    def vcov(self):
        return _did2s_vcov(
            data=self._data,
            yname=self._yname,
            first_stage=self._fml1,
            second_stage=self._fml2,
            treatment="ATT",
            first_u=self._first_u,
            second_u=self._second_u,
            cluster=self._cluster,
        )


def _did2s_estimate(
    data: pd.DataFrame,
    yname: str,
    _first_stage: str,
    _second_stage: str,
    treatment: str,
    i_ref1: Optional[Union[int, str, List]] = None,
    i_ref2: Optional[Union[int, str, List]] = None,
):
    """
    Args:
        data (pd.DataFrame): The DataFrame containing all variables.
        yname (str): The name of the dependent variable.
        _first_stage (str): The formula for the first stage.
        _second_stage (str): The formula for the second stage.
        treatment (str): The name of the treatment variable. Must be boolean.
        i_ref1 (int, str or list): The reference value(s) for the first variable used with "i()" syntax. Only applicable for the second stage formula.
        i_ref2 (int, str or list): The reference value(s) for the second variable used with "i()" syntax. Only applicable for the second stage formula.
    Returns:
        A fitted model object of class feols and the first and second stage residuals.
    """

    _first_stage_full = f"{yname} {_first_stage}"
    _second_stage_full = f"{yname}_hat {_second_stage}"

    if treatment is not None:
        if treatment not in data.columns:
            raise ValueError(f"The variable {treatment} is not in the data.")
        # check that treatment is boolean
        if data[treatment].dtype != "bool":
            if data[treatment].dtype in [
                "int64",
                "int32",
                "int8",
                "float64",
                "float32",
            ]:
                if data[treatment].nunique() == 2:
                    data[treatment] = data[treatment].astype(bool)
                    warnings.warn(
                        f"The treatment variable {treatment} was converted to boolean."
                    )
                else:
                    raise ValueError(
                        f"The treatment variable {treatment} must be boolean."
                    )
        _not_yet_treated_data = data[data[treatment] == False]
    else:
        _not_yet_treated_data = data[data["ATT"] == False]

    # check if first stage formulas has fixed effects
    if "|" not in _first_stage:
        raise ValueError("First stage formula must contain fixed effects.")
    # check if second stage formulas has fixed effects
    if "|" in _second_stage:
        raise ValueError("Second stage formula must not contain fixed effects.")

    # estimate first stage
    fit1 = feols(
        fml=_first_stage_full,
        data=_not_yet_treated_data,
        vcov="iid",
        i_ref1=None,
        i_ref2=None,
    )  # iid as it might be faster than CRV

    # obtain estimated fixed effects
    fit1.fixef()

    # demean data
    Y_hat = fit1.predict(newdata=data)
    _first_u = data[f"{yname}"].to_numpy().flatten() - Y_hat
    data[f"{yname}_hat"] = _first_u

    # intercept needs to be dropped by hand due to the presence of fixed effects in the first stage
    fit2 = feols(
        _second_stage_full, data=data, vcov="iid", drop_intercept=True, i_ref1=i_ref1, i_ref2=i_ref2
    )
    _second_u = fit2.resid()

    return fit2, _first_u, _second_u


def _did2s_vcov(
    data: pd.DataFrame,
    yname: str,
    first_stage: str,
    second_stage: str,
    treatment: str,
    first_u: np.ndarray,
    second_u: np.ndarray,
    cluster: str,
    i_ref1: Optional[Union[int, str, List]] = None,
    i_ref2: Optional[Union[int, str, List]] = None,
):
    """
    Compute a variance covariance matrix for Gardner's 2-stage Difference-in-Differences Estimator.
    Args:
        data (pd.DataFrame): The DataFrame containing all variables.
        yname (str): The name of the dependent variable.
        first_stage (str): The formula for the first stage.
        second_stage (str): The formula for the second stage.
        treatment (str): The name of the treatment variable.
        first_u (np.ndarray): The first stage residuals.
        second_u (np.ndarray): The second stage residuals.
        cluster (str): The name of the cluster variable.
        i_ref1 (int, str or list): The reference value(s) for the first variable used with "i()" syntax. Only applicable for the second stage formula.
        i_ref2 (int, str or list): The reference value(s) for the second variable used with "i()" syntax. Only applicable for the second stage formula.
    Returns:
        A variance covariance matrix.
    """

    cluster_col = data[cluster]
    _, clustid = pd.factorize(cluster_col)

    _G = clustid.nunique()  # actually not used here, neither in did2s

    # some formula parsing to get the correct formula for the first and second stage model matrix
    first_stage_x, first_stage_fe = first_stage.split("|")
    first_stage_fe = [f"C({i})" for i in first_stage_fe.split("+")]
    first_stage_fe = "+".join(first_stage_fe)
    first_stage = f"{first_stage_x}+{first_stage_fe}"

    second_stage = f"{second_stage}"

    # note for future Alex: intercept needs to be dropped! it is not as fixed effects are converted to
    # dummies, hence has_fixed checks are False
    _, X1, _, _, _, _, _, _, _ = model_matrix_fixest(
        fml=f"{yname} {first_stage}", data=data, drop_intercept = False, i_ref1=i_ref1, i_ref2=i_ref2
    )
    _, X2, _, _, _, _, _, _, _ = model_matrix_fixest(
        fml=f"{yname} {second_stage}", data=data, drop_intercept = True, i_ref1=i_ref1, i_ref2=i_ref2
    )  # reference values not dropped, multicollinearity error

    X1 = csr_matrix(X1.values)
    X2 = csr_matrix(X2.values)

    X10 = X1.copy().tocsr()
    treated_rows = np.where(data[treatment], 0, 1)
    X10 = X10.multiply(treated_rows[:, None])

    X10X10 = X10.T.dot(X10)
    X2X1 = X2.T.dot(X1)
    X2X2 = X2.T.dot(X2)

    V = spsolve(X10X10, X2X1.T).T

    k = X2.shape[1]
    vcov = np.zeros((k, k))

    X10 = X10.tocsr()
    X2 = X2.tocsr()

    for (
        _,
        g,
    ) in enumerate(clustid):
        X10g = X10[cluster_col == g, :]
        X2g = X2[cluster_col == g, :]
        first_u_g = first_u[cluster_col == g]
        second_u_g = second_u[cluster_col == g]

        W_g = X2g.T.dot(second_u_g) - V @ X10g.T.dot(first_u_g)
        score = spsolve(X2X2, W_g)
        if score.ndim == 1:
            score = score.reshape(-1, 1)
        cov_g = score.dot(score.T)

        vcov += cov_g

    return vcov, _G


def did2s(
    data: pd.DataFrame,
    yname: str,
    first_stage: str,
    second_stage: str,
    treatment: str,
    cluster: str,
    i_ref1: Optional[Union[int, str, List]] = None,
    i_ref2: Optional[Union[int, str, List]] = None,
):
    """
    Estimate a Difference-in-Differences model using Gardner's two-step DID2S estimator.
    Args:
        data (pd.DataFrame): The DataFrame containing all variables.
        yname (str): The name of the dependent variable.
        first_stage (str): The formula for the first stage. Must start with '~'.
        second_stage (str): The formula for the second stage. Must start with '~'.
        treatment (str): The name of the treatment variable.
        cluster (str): The name of the cluster variable.
        i_ref1 (int, str or list): The reference value(s) for the first variable used with "i()" syntax. Only applicable for the second stage formula.
        i_ref2 (int, str or list): The reference value(s) for the second variable used with "i()" syntax. Only applicable for the second stage formula.
    Returns:
        A fitted model object of class feols.
    Examples:
        >>> from pyfixest.experimental.did import event_study, did2s
        >>> from pyfixest.estimation import feols
        >>> from pyfixest.summarize import etable, summary
        >>> import pandas as pd
        >>> import numpy as np
        >>> df_het = pd.read_csv("pyfixest/experimental/data/df_het.csv")
        >>> fit = did2s(
        >>>     yname = "dep_var",
        >>>     first_stage = "~ X | state + year",
        >>>     second_stage = "~ 0 + treat",
        >>>     treatment = "treat",
        >>>     data = df_het,
        >>>     cluster = "state",
        >>> )
    """

    first_stage = first_stage.replace(" ", "")
    second_stage = second_stage.replace(" ", "")
    assert first_stage[0] == "~", "First stage must start with ~"
    assert second_stage[0] == "~", "Second stage must start with ~"

    data = data.copy()

    fit, first_u, second_u = _did2s_estimate(
        data=data,
        yname=yname,
        _first_stage=first_stage,
        _second_stage=second_stage,
        treatment=treatment,
        i_ref1=i_ref1,
        i_ref2=i_ref2,
    )

    vcov, _G = _did2s_vcov(
        data=data,
        yname=yname,
        first_stage=first_stage,
        second_stage=second_stage,
        treatment=treatment,
        first_u=first_u,
        second_u=second_u,
        cluster=cluster,
        i_ref1=i_ref1,
        i_ref2=i_ref2,
    )

    fit._vcov = vcov
    fit._G = _G
    fit.get_inference()  # update inference with correct vcov matrix

    fit._vcov_type = "CRV1"
    fit._vcov_type_detail = "CRV1 (GMM)"
    # fit._G = did2s._G
    fit._method = "did2s"

    return fit
