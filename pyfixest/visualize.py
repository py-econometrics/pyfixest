import pandas as pd
from typing import List, Tuple, Optional
from pyfixest.summarize import _post_processing_input_checks
from typing import Union
from lets_plot import (
    ggplot,
    aes,
    geom_point,
    geom_errorbar,
    geom_hline,
    geom_vline,
    ggsize,
    theme,
    element_text,
    position_dodge,
    coord_flip,
    ggtitle,
    ylab,
)
from lets_plot import *

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
):
    """

    # iplot

    Plot model coefficients for variables interacted via "i()" syntax, with confidence intervals.
    Args:
        models (list): A list of fitted models of type `Feols` or `Fepois`, or just a single model.
        figsize (tuple): The size of the figure.
        alpha (float): The significance level for the confidence intervals.
        yintercept (int or None): The value at which to draw a horizontal line on the plot.
        xintercept (int or None): The value at which to draw a vertical line on the plot.
        rotate_xticks (float): The angle in degrees to rotate the xticks labels. Default is 0 (no rotation).
        title (str): The title of the plot.
        coord_flip (bool): Whether to flip the coordinates of the plot. Default is True.
    Returns:
        A lets-plot figure.
    """

    import pdb; pdb.set_trace()

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
    fml_list = df.index.unique()
    # keep only coefficients interacted via the i() syntax
    df = df[df.Coefficient.isin(all_icovars)].reset_index()

    plot = _coefplot(
        df=df,
        figsize=figsize,
        alpha=alpha,
        yintercept=yintercept,
        xintercept=xintercept,
        rotate_xticks=rotate_xticks,
        title=title,
        flip_coord=coord_flip,
    )

    return plot


def coefplot(
    models: List,
    alpha: int = 0.05,
    figsize: tuple = (500, 300),
    yintercept: float = 0,
    xintercept: float = None,
    rotate_xticks: int = 0,
    coefficients: Optional[List[str]] = None,
    title: Optional[str] = None,
    coord_flip: Optional[bool] = True,
):
    """

    # coefplot

    Plot model coefficients with confidence intervals.

    Args:
        models (list): A list of fitted models of type `Feols` or `Fepois`, or just a single model.
        figsize (tuple): The size of the figure.
        alpha (float): The significance level for the confidence intervals.
        yintercept (float or None): The value at which to draw a horizontal line on the plot. Default is 0.
        xintercept (float or None): The value at which to draw a vertical line on the plot. Default is None.
        rotate_xticks (float): The angle in degrees to rotate the xticks labels. Default is 0 (no rotation).
        coefficients (list): A list of coefficients to plot. If None, all coefficients are plotted.
        title (str): The title of the plot.
        coord_flip (bool): Whether to flip the coordinates of the plot. Default is True.
    Returns:
        A lets-plot figure.
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

    if coefficients is not None:
        df = df[df.Coefficient.isin(coefficients)].reset_index()

    plot = _coefplot(
        df=df,
        figsize=figsize,
        alpha=alpha,
        yintercept=yintercept,
        xintercept=xintercept,
        rotate_xticks=rotate_xticks,
        title=title,
        flip_coord=coord_flip,
    )

    return plot


def _coefplot(
    df: pd.DataFrame,
    figsize: Tuple[int, int],
    alpha: float,
    yintercept: Optional[int] = None,
    xintercept: Optional[int] = None,
    rotate_xticks: float = 0,
    title: Optional[str] = None,
    flip_coord: Optional[bool] = True,
):
    """
    Plot model coefficients with confidence intervals.
    Args:
        models (list): A list of fitted models indices.
        figsize (tuple): The size of the figure.
        alpha (float): The significance level for the confidence intervals.
        yintercept (int or None): The value at which to draw a horizontal line on the plot.
        xintercept (int or None): The value at which to draw a vertical line on the plot.
        df (pandas.DataFrame): The dataframe containing the data used for the model fitting.
        rotate_xticks (float): The angle in degrees to rotate the xticks labels. Default is 0 (no rotation).
        title (str): The title of the plot.
        flip_coord (bool): Whether to flip the coordinates of the plot. Default is True.
    Returns:
        A lets-plot figure.
    """

    df.reset_index(inplace=True)
    df.rename(columns={"fml": "Model"}, inplace=True)

    plot = (
        ggplot(df, aes(x="Coefficient", y="Estimate", color="Model"))
        + geom_point(position=position_dodge(0.5))
        + geom_errorbar(
            aes(ymin="2.5 %", ymax="97.5 %"), width=0.05, position=position_dodge(0.5)
        )
        + ylab("Estimate and 95% Confidence Interval")
    )

    if flip_coord:
        plot += coord_flip()
    if yintercept is not None:
        plot += geom_hline(yintercept=yintercept, linetype="dashed", color="red")
    if xintercept is not None:
        plot += geom_vline(xintercept=xintercept, linetype="dashed")
    if figsize is not None:
        plot += ggsize(figsize[0], figsize[1])
    if rotate_xticks is not None:
        plot += theme(axis_text_x=element_text(angle=rotate_xticks))
    if title is not None:
        plot += ggtitle(title)

    return plot
