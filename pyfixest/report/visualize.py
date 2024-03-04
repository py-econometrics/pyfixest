from typing import Optional, Union

import pandas as pd

# from lets_plot import *
from lets_plot import (
    LetsPlot,
    aes,
    coord_flip,
    element_text,
    geom_errorbar,
    geom_hline,
    geom_point,
    geom_vline,
    ggplot,
    ggsize,
    ggtitle,
    position_dodge,
    theme,
    ylab,
)

from pyfixest.report.summarize import _post_processing_input_checks
from pyfixest.utils.dev_utils import _select_order_coefs

LetsPlot.setup_html()


def iplot(
    models,
    alpha: float = 0.05,
    figsize: tuple = (500, 300),
    yintercept: Union[int, str, None] = None,
    xintercept: Union[int, str, None] = None,
    rotate_xticks: int = 0,
    title: Optional[str] = None,
    coord_flip: Optional[bool] = True,
    keep: Optional[Union[list, str]] = [],
    drop: Optional[Union[list, str]] = [],
    exact_match: Optional[bool] = False,
):
    r"""
    Plot model coefficients.

    Plot model coefficients for variables interacted via "i()" syntax, with
    confidence intervals.

    Parameters
    ----------
    models : list or object
        A list of fitted models of type `Feols` or
        `Fepois`, or just a single model.
    figsize : tuple
        The size of the figure.
    alpha : float
        The significance level for the confidence intervals.
    yintercept : int or None, optional
        The value at which to draw a horizontal line on the plot.
    xintercept : int or None, optional
        The value at which to draw a vertical line on the plot.
    rotate_xticks : float, optional
        The angle in degrees to rotate the xticks labels. Default is 0 (no rotation).
    title : str, optional
        The title of the plot.
    coord_flip : bool, optional
        Whether to flip the coordinates of the plot. Default is True.
    keep: str or list of str, optional
        The pattern for retaining coefficient names. You can pass a string (one
        pattern) or a list (multiple patterns). Default is keeping all coefficients.
        You should use regular expressions to select coefficients.
            "age",            # would keep all coefficients containing age
            r"^tr",           # would keep all coefficients starting with tr
            r"\\d$",          # would keep all coefficients ending with number
        Output will be in the order of the patterns.
    drop: str or list of str, optional
        The pattern for excluding coefficient names. You can pass a string (one
        pattern) or a list (multiple patterns). Syntax is the same as for `keep`.
        Default is keeping all coefficients. Parameter `keep` and `drop` can be
        used simultaneously.
    exact_match: bool, optional
        Whether to use exact match for `keep` and `drop`. Default is False.
        If True, the pattern will be matched exactly to the coefficient name
        instead of using regular expressions.

    Returns
    -------
    object
        A lets-plot figure.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    df = pf.get_data()
    fit1 = pf.feols("Y ~ i(f1)", data = df)
    fit2 = pf.feols("Y ~ i(f1) + X2", data = df)
    fit3 = pf.feols("Y ~ i(f1) + X2 | f2", data = df)

    pf.iplot([fit1, fit2, fit3])
    ```
    """
    models = _post_processing_input_checks(models)

    df_all = []
    all_icovars = []

    for x, _ in enumerate(models):
        fxst = models[x]
        if fxst._icovars is None:
            raise ValueError(
                f"The {x} th estimated model did not have ivars / 'i()' model syntax."
                "In consequence, the '.iplot()' method is not supported."
            )
        all_icovars += fxst._icovars
        df_model = fxst.tidy().reset_index()  # Coefficient -> simple column
        df_model["fml"] = fxst._fml
        df_model.set_index("fml", inplace=True)
        df_all.append(df_model)

    # drop duplicates
    all_icovars = list(set(all_icovars))

    df = pd.concat(df_all, axis=0)
    if keep or drop:
        idxs = _select_order_coefs(df["Coefficient"], keep, drop, exact_match)
    else:
        idxs = df["Coefficient"]
    df = df.loc[df["Coefficient"].isin(idxs), :]
    fml_list = df.index.unique()  # noqa: F841
    # keep only coefficients interacted via the i() syntax
    df = df[df["Coefficient"].isin(all_icovars)].reset_index()

    return _coefplot(
        df=df,
        figsize=figsize,
        alpha=alpha,
        yintercept=yintercept,
        xintercept=xintercept,
        rotate_xticks=rotate_xticks,
        title=title,
        flip_coord=coord_flip,
    )


def coefplot(
    models: list,
    alpha: int = 0.05,
    figsize: tuple = (500, 300),
    yintercept: float = 0,
    xintercept: float = None,
    rotate_xticks: int = 0,
    title: Optional[str] = None,
    coord_flip: Optional[bool] = True,
    keep: Optional[Union[list, str]] = [],
    drop: Optional[Union[list, str]] = [],
    exact_match: Optional[bool] = False,
):
    r"""
    Plot model coefficients with confidence intervals.

    Parameters
    ----------
    models : list or object
        A list of fitted models of type `Feols` or `Fepois`, or just a single model.
    figsize : tuple
        The size of the figure.
    alpha : float
        The significance level for the confidence intervals.
    yintercept : float or None, optional
        The value at which to draw a horizontal line on the plot. Default is 0.
    xintercept : float or None, optional
        The value at which to draw a vertical line on the plot. Default is None.
    rotate_xticks : float, optional
        The angle in degrees to rotate the xticks labels. Default is 0 (no rotation).
    title : str, optional
        The title of the plot.
    coord_flip : bool, optional
        Whether to flip the coordinates of the plot. Default is True.
    keep: str or list of str, optional
        The pattern for retaining coefficient names. You can pass a string (one
        pattern) or a list (multiple patterns). Default is keeping all coefficients.
        You should use regular expressions to select coefficients.
            "age",            # would keep all coefficients containing age
            r"^tr",           # would keep all coefficients starting with tr
            r"\\d$",          # would keep all coefficients ending with number
        Output will be in the order of the patterns.
    drop: str or list of str, optional
        The pattern for excluding coefficient names. You can pass a string (one
        pattern) or a list (multiple patterns). Syntax is the same as for `keep`.
        Default is keeping all coefficients. Parameter `keep` and `drop` can be
        used simultaneously.
    exact_match: bool, optional
        Whether to use exact match for `keep` and `drop`. Default is False.
        If True, the pattern will be matched exactly to the coefficient name
        instead of using regular expressions.

    Returns
    -------
    object
        A lets-plot figure.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    df = pf.get_data()
    fit1 = pf.feols("Y ~ X1", data = df)
    fit2 = pf.feols("Y ~ X1 + X2", data = df)
    fit3 = pf.feols("Y ~ X1 + X2 | f1", data = df)

    pf.coefplot([fit1, fit2, fit3])
    ```
    """
    models = _post_processing_input_checks(models)
    df_all = []
    for x, _ in enumerate(models):
        fxst = models[x]
        df_model = fxst.tidy().reset_index()
        df_model["fml"] = fxst._fml
        df_model.set_index("fml", inplace=True)
        df_all.append(df_model)

    df = pd.concat(df_all, axis=0)
    if keep or drop:
        idxs = _select_order_coefs(df.index, keep, drop, exact_match)
    else:
        idxs = df.index
    df = df.loc[idxs, :].reset_index()

    return _coefplot(
        df=df,
        figsize=figsize,
        alpha=alpha,
        yintercept=yintercept,
        xintercept=xintercept,
        rotate_xticks=rotate_xticks,
        title=title,
        flip_coord=coord_flip,
    )


def _coefplot(
    df: pd.DataFrame,
    figsize: tuple[int, int],
    alpha: float,
    yintercept: Optional[int] = None,
    xintercept: Optional[int] = None,
    rotate_xticks: float = 0,
    title: Optional[str] = None,
    flip_coord: Optional[bool] = True,
):
    """
    Plot model coefficients with confidence intervals.

    Parameters
    ----------
    models : list
        A list of fitted models indices.
    figsize : tuple
        The size of the figure.
    alpha : float
        The significance level for the confidence intervals.
    yintercept : int or None, optional
        The value at which to draw a horizontal line on the plot.
    xintercept : int or None, optional
        The value at which to draw a vertical line on the plot.
    df : pandas.DataFrame
        The dataframe containing the data used for the model fitting.
    rotate_xticks : float, optional
        The angle in degrees to rotate the xticks labels. Default is 0 (no rotation).
    title : str, optional
        The title of the plot.
    flip_coord : bool, optional
        Whether to flip the coordinates of the plot. Default is True.

    Returns
    -------
    object
        A lets-plot figure.
    """
    df.reset_index(inplace=True)
    df.rename(columns={"fml": "Model"}, inplace=True)

    plot = (
        ggplot(df, aes(x="Coefficient", y="Estimate", color="Model"))
        + geom_point(position=position_dodge(0.5))
        + geom_errorbar(
            aes(ymin="2.5%", ymax="97.5%"), width=0.05, position=position_dodge(0.5)
        )
        + ylab("Estimate and 95% Confidence Interval")
    )

    if flip_coord:
        plot += coord_flip()
    if yintercept is not None:
        plot += geom_hline(yintercept=yintercept, linetype="dashed", color="black")
    if xintercept is not None:
        plot += geom_vline(xintercept=xintercept, linetype="dashed", color="black")
    if figsize is not None:
        plot += ggsize(figsize[0], figsize[1])
    if rotate_xticks is not None:
        plot += theme(axis_text_x=element_text(angle=rotate_xticks))
    if title is not None:
        plot += ggtitle(title)

    return plot
