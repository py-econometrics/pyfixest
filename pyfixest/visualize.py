import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from scipy.stats import norm
import re
from pyfixest.summarize import _post_processing_input_checks
from typing import Union


def iplot(
    models,
    alpha: float = 0.05,
    figsize: tuple = (10, 10),
    yintercept: Union[int, str, None] = None,
    xintercept: Union[int, str, None] = None,
    rotate_xticks: int = 0,
) -> None:
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
    Returns:
        None
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
    fml_list = df.index.unique()
    # keep only coefficients interacted via the i() syntax
    df = df[df.Coefficient.isin(all_icovars)].reset_index()

    plot = _coefplot(
        models=fml_list,
        df=df,
        figsize=figsize,
        alpha=alpha,
        yintercept=yintercept,
        xintercept=xintercept,
        is_iplot=True,
        rotate_xticks=rotate_xticks,
    )

    return plot


def coefplot(
    models,
    alpha=0.05,
    figsize=(10, 10),
    yintercept=None,
    xintercept=None,
    rotate_xticks=0,
    coefficients: Optional[List[str]] = None,
):
    """

    # coefplot

    Plot model coefficients with confidence intervals.
    Args:
        models (list): A list of fitted models of type `Feols` or `Fepois`, or just a single model.
        figsize (tuple): The size of the figure.
        alpha (float): The significance level for the confidence intervals.
        yintercept (int or None): The value at which to draw a horizontal line on the plot.
        xintercept (int or None): The value at which to draw a vertical line on the plot.
        rotate_xticks (float): The angle in degrees to rotate the xticks labels. Default is 0 (no rotation).
        coefficients (list): A list of coefficients to plot. If None, all coefficients are plotted.
    Returns:
        None
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
    fml_list = df.index.unique()

    if coefficients is not None:
        df = df[df.Coefficient.isin(coefficients)].reset_index()

    plot = _coefplot(
        models=fml_list,
        df=df,
        figsize=figsize,
        alpha=alpha,
        yintercept=yintercept,
        xintercept=xintercept,
        rotate_xticks=rotate_xticks,
        is_iplot=False,
    )

    return plot


def _coefplot(
    models: List,
    df: pd.DataFrame,
    figsize: Tuple[int, int],
    alpha: float,
    yintercept: Optional[int] = None,
    xintercept: Optional[int] = None,
    is_iplot: bool = False,
    rotate_xticks: float = 0,
) -> None:
    """
    Plot model coefficients with confidence intervals.
    Args:
        models (list): A list of fitted models indices.
        figsize (tuple): The size of the figure.
        alpha (float): The significance level for the confidence intervals.
        yintercept (int or None): The value at which to draw a horizontal line on the plot.
        xintercept (int or None): The value at which to draw a vertical line on the plot.
        df (pandas.DataFrame): The dataframe containing the data used for the model fitting.
        is_iplot (bool): If True, plot variable interactions specified via the `i()` syntax.
        rotate_xticks (float): The angle in degrees to rotate the xticks labels. Default is 0 (no rotation).
    Returns:
    None
    """

    if len(models) > 1:
        fig, ax = plt.subplots(
            len(models), gridspec_kw={"hspace": 0.5}, figsize=figsize
        )

        for x, model in enumerate(models):
            df_model = df.reset_index().set_index("fml").xs(model)
            coef = df_model["Estimate"]
            conf_l = coef - df_model["Std. Error"] * norm.ppf(1 - alpha / 2)
            conf_u = coef + df_model["Std. Error"] * norm.ppf(1 - alpha / 2)
            coefnames = df_model["Coefficient"].values.tolist()

            # could be moved out of the for loop, as the same ivars for all
            # models.

            if is_iplot == True:
                fig.suptitle("iplot")
                coefnames = [
                    (i)
                    for string in coefnames
                    for i in re.findall(r"\[T\.([\d\.\-]+)\]", string)
                ]

            ax[x].scatter(coefnames, coef, color="b", alpha=0.8)
            ax[x].scatter(coefnames, conf_u, color="b", alpha=0.8, marker="_", s=100)
            ax[x].scatter(coefnames, conf_l, color="b", alpha=0.8, marker="_", s=100)
            ax[x].vlines(coefnames, ymin=conf_l, ymax=conf_u, color="b", alpha=0.8)
            if yintercept is not None:
                ax[x].axhline(yintercept, color="red", linestyle="--", alpha=0.5)
            if xintercept is not None:
                ax[x].axvline(xintercept, color="red", linestyle="--", alpha=0.5)
            ax[x].set_ylabel("Coefficients")
            ax[x].set_title(model)
            ax[x].tick_params(axis="x", rotation=rotate_xticks)

    else:
        fig, ax = plt.subplots(figsize=figsize)

        model = models[0]

        df_model = df.reset_index().set_index("fml").xs(model)

        coef = df_model["Estimate"].values
        conf_l = coef - df_model["Std. Error"].values * norm.ppf(1 - alpha / 2)
        conf_u = coef + df_model["Std. Error"].values * norm.ppf(1 - alpha / 2)
        coefnames = df_model["Coefficient"].values.tolist()

        if is_iplot == True:
            fig.suptitle("iplot")
            coefnames = [
                (i)
                for string in coefnames
                for i in re.findall(r"\[T\.([\d\.\-]+)\]", string)
            ]

        ax.scatter(coefnames, coef, color="b", alpha=0.8)
        ax.scatter(coefnames, conf_u, color="b", alpha=0.8, marker="_", s=100)
        ax.scatter(coefnames, conf_l, color="b", alpha=0.8, marker="_", s=100)
        ax.vlines(coefnames, ymin=conf_l, ymax=conf_u, color="b", alpha=0.8)
        if yintercept is not None:
            ax.axhline(yintercept, color="red", linestyle="--", alpha=0.5)
        if xintercept is not None:
            ax.axvline(xintercept, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel("Coefficients")
        ax.set_title(model)
        ax.tick_params(axis="x", rotation=rotate_xticks)

        plt.show()
        plt.close()
