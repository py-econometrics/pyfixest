from typing import Optional, Union

import pandas as pd

from pyfixest.did.did2s import DID2S, _did2s_estimate, _did2s_vcov
from pyfixest.did.lpdid import LPDID
from pyfixest.did.twfe import TWFE
from pyfixest.exceptions import NotImplementedError


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
    Estimate Event Study Model.

    This function allows for the estimation of treatment effects using different
    estimators. Currently, it supports "twfe" for the two-way fixed effects
    estimator and "did2s" for Gardner's two-step DID2S estimator. Other estimators
    are in development.

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
        If False, estimates the canonical event study design with all leads and
        lags. Default is True.

    Returns
    -------
    object
        A fitted model object of class [Feols(/reference/Feols.qmd).

    Examples
    --------
    ```{python}
    import pandas as pd
    from pyfixest.did.estimation import event_study

    url = "https://raw.githubusercontent.com/s3alfisc/pyfixest/master/pyfixest/did/data/df_het.csv"
    df_het = pd.read_csv(url)

    fit_twfe = event_study(
        df_het,
        yname="dep_var",
        idname="unit",
        tname="year",
        gname="g",
        estimator="twfe",
        att=True,
    )

    fit_twfe.tidy()
    ```
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
    fit.get_inference()

    return fit


def did2s(
    data: pd.DataFrame,
    yname: str,
    first_stage: str,
    second_stage: str,
    treatment: str,
    cluster: str,
    i_ref1: Optional[Union[int, str, list]] = None,
    i_ref2: Optional[Union[int, str, list]] = None,
):
    """
    Estimate a Difference-in-Differences model using Gardner's two-step DID2S estimator.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing all variables.
    yname : str
        The name of the dependent variable.
    first_stage : str
        The formula for the first stage, starting with '~'.
    second_stage : str
        The formula for the second stage, starting with '~'.
    treatment : str
        The name of the treatment variable.
    cluster : str
        The name of the cluster variable.
    i_ref1 : int, str, list, optional
        The reference value(s) for the first variable used with "i()" syntax in
        the second stage formula. Default is None.
    i_ref2 : int, str, list, optional
        The reference value(s) for the second variable used with "i()" syntax in
        the second stage formula. Default is None.

    Returns
    -------
    object
        A fitted model object of class [Feols(/reference/Feols.qmd).

    Examples
    --------
    ```{python}
    import pandas as pd
    import numpy as np
    from pyfixest.did.estimation import did2s

    url = "https://raw.githubusercontent.com/s3alfisc/pyfixest/master/pyfixest/did/data/df_het.csv"
    df_het = pd.read_csv(url)
    df_het.head()
    ```

    In a first step, we estimate a classical event study model:

    ```{python}
    # estimate the model
    fit = did2s(
        df_het,
        yname="dep_var",
        first_stage="~ 0 | unit + year",
        second_stage="~i(rel_year)",
        treatment="treat",
        cluster="state",
        i_ref1=[-1.0, np.inf],
    )

    fit.tidy().head()
    ```

    We can also inspect the model visually:

    ```{python}
    fit.iplot(figsize= [1200, 400], coord_flip=False).show()
    ```

    To estimate a pooled effect, we need to slightly update the second stage formula:

    ```{python}
    fit = did2s(
        df_het,
        yname="dep_var",
        first_stage="~ 0 | unit + year",
        second_stage="~i(treat)",
        treatment="treat",
        cluster="state"
    )
    fit.tidy().head()
    ```
    """
    first_stage = first_stage.replace(" ", "")
    second_stage = second_stage.replace(" ", "")
    assert first_stage[0] == "~", "First stage must start with ~"
    assert second_stage[0] == "~", "Second stage must start with ~"

    # assert that there is no 0, -1 or - 1 in the second stage formula
    if "0" in second_stage or "-1" in second_stage:
        raise ValueError(
            """
            The second stage formula should not contain '0' or '-1'. Note that
            the intercept is dropped automatically due to the presence of fixed
            effects in the first stage.
            """
        )

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


def lpdid(
    data: pd.DataFrame,
    yname: str,
    idname: str,
    tname: str,
    gname: str,
    vcov: Optional[Union[str, dict[str, str]]] = None,
    pre_window: Optional[int] = None,
    post_window: Optional[int] = None,
    never_treated: int = 0,
    att: bool = True,
    xfml=None,
) -> pd.DataFrame:
    """
    Local projections approach to estimation.

    Estimate a Difference-in-Differences / Event Study Model via the Local
    Projections Approach.

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
    vcov : str, dict, optional
        The type of inference to employ. Defaults to {"CRV1": idname}.
        Options include "iid", "hetero", or a dictionary like {"CRV1": idname}.
    pre_window : int, optional
        The number of periods before the treatment to include in the estimation.
        Default is the minimum relative year in the data.
    post_window : int, optional
        The number of periods after the treatment to include in the estimation.
        Default is the maximum relative year in the data.
    never_treated : int, optional
        Value in gname indicating units never treated. Default is 0.
    att : bool, optional
        If True, estimates the pooled average treatment effect on the treated (ATT).
        Default is False.
    xfml : str, optional
        Formula for the covariates. Not yet supported.

    Returns
    -------
    DataFrame
        A DataFrame with the estimated coefficients.

    Examples
    --------
    ```{python}
    import pandas as pd
    from pyfixest.did.estimation import lpdid

    url = "https://raw.githubusercontent.com/s3alfisc/pyfixest/master/pyfixest/did/data/df_het.csv"
    df_het = pd.read_csv(url)

    fit = lpdid(
        df_het,
        yname="dep_var",
        idname="unit",
        tname="year",
        gname="g",
        vcov={"CRV1": "state"},
        pre_window=-20,
        post_window=20,
        att=False
    )

    fit.tidy().head()
    fit.iplot(figsize= [1200, 400], coord_flip=False).show()
    ```

    To get the ATT, set `att=True`:

    ```{python}
    fit = lpdid(
        df_het,
        yname="dep_var",
        idname="unit",
        tname="year",
        gname="g",
        vcov={"CRV1": "state"},
        pre_window=-20,
        post_window=20,
        att=True
    )
    fit.tidy()
    ```
    """
    FIT = LPDID(
        data=data,
        yname=yname,
        idname=idname,
        tname=tname,
        gname=gname,
        cluster={"CRV1": idname},  # just something to pass DID input checks
        vcov=vcov,
        pre_window=pre_window,
        post_window=post_window,
        never_treated=never_treated,
        att=att,
        xfml=xfml,
    )

    FIT.estimate()

    return FIT
