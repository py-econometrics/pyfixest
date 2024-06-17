from typing import Optional

import pandas as pd

from pyfixest.did.did2s import DID2S
from pyfixest.did.twfe import TWFE
from pyfixest.errors import NotImplementedError


def event_study(
    data: pd.DataFrame,
    yname: str,
    idname: str,
    tname: str,
    gname: str,
    xfml: Optional[str] = None,
    estimator: Optional[str] = "twfe",
    att: Optional[bool] = True,
    cluster: Optional[str] = "idname",
):
    """
    Estimate a treatment effect using an event study design.

    This function allows for the estimation of treatment effects using different
    estimators.
    Currently, it supports "twfe" for the two-way fixed effects estimator and
    "did2s" for Gardner's two-step DID2S estimator.
    Other estimators are in development.

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing all variables.
    yname : str
        The name of the dependent variable.
    idname : str
        The name of the id variable.
    tname : str
        Variable name for calendar period.
    gname : str
        Unit-specific time of initial treatment.
    xfml : str
        The formula for the covariates.
    estimator : str
        The estimator to use. Options are "did2s" and "twfe".
    att : bool, optional
        If True, estimates the average treatment effect on the treated (ATT).
        If False, estimates the canonical event study design with all leads and lags.
        Default is True.

    Returns
    -------
    object
        A fitted model object of class [Feols(/reference/Feols.qmd).

    Examples
    --------
    ```{python}
    import pandas as pd

    --------


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
        raise NotImplementedError("Estimator not supported")

    # update inference with vcov matrix
    fit.inference()

    return fit
