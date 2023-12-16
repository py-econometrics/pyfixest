import pandas as pd
from pyfixest.exceptions import NotImplementedError
from pyfixest.did.did2s import DID2S
from pyfixest.did.twfe import TWFE


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
        >>> from pyfixest.did.did import event_study, did2s
        >>> from pyfixest.estimation import feols
        >>> from pyfixest.summarize import etable, summary
        >>> import pandas as pd
        >>> import numpy as np
        >>> df_het = pd.read_csv("pyfixest/did/data/df_het.csv")
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
