import math
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make lets-plot an optional dependency
try:
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

    _HAS_LETS_PLOT = True
except ImportError:
    _HAS_LETS_PLOT = False

from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.models.feiv_ import Feiv
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.estimation.quantreg.quantreg_ import Quantreg
from pyfixest.report.utils import (
    _check_label_keys_in_covars,
    _post_processing_input_checks,
    _relabel_expvar,
)
from pyfixest.utils.dev_utils import _select_order_coefs

ModelInputType = Union[
    FixestMulti, Feols, Fepois, Feiv, list[Union[Feols, Fepois, Feiv]]
]

# Only setup lets-plot if it's available
if _HAS_LETS_PLOT:
    LetsPlot.setup_html()


def set_figsize(
    figsize: Optional[tuple[int, int]], plot_backend: str
) -> tuple[int, int]:
    """
    Set the figure size based on the plot backend.

    Parameters
    ----------
    figsize: tuple[int, int], optional
        The size of the figure. Default is None.
    plot_backend: str
        The plot backend. Must be one of 'matplotlib' or 'lets_plot'.

    Returns
    -------
    tuple[int, int]
        The size of the figure.
    """
    if figsize is not None:
        return figsize

    if plot_backend == "matplotlib":
        return (10, 6)
    elif plot_backend == "lets_plot":
        if not _HAS_LETS_PLOT:
            raise ImportError(
                "The 'lets_plot' package is required for the 'lets_plot' backend. "
                "Please install it with 'pip install lets-plot' or use the 'matplotlib' backend."
            )
        return (500, 300)
    else:
        raise ValueError("plot_backend must be either 'lets_plot' or 'matplotlib'.")


def iplot(
    models: ModelInputType,
    alpha: float = 0.05,
    figsize: Optional[tuple[int, int]] = None,
    yintercept: Union[int, str, None] = None,
    xintercept: Union[int, str, None] = None,
    rotate_xticks: int = 0,
    title: Optional[str] = None,
    coord_flip: bool = True,
    keep: Optional[Union[list, str]] = None,
    drop: Optional[Union[list, str]] = None,
    exact_match: bool = False,
    plot_backend: str = "lets_plot" if _HAS_LETS_PLOT else "matplotlib",
    labels: Optional[dict] = None,
    cat_template: Optional[str] = None,
    rename_models: Optional[dict[str, str]] = None,
    ax: Optional[plt.Axes] = None,
    joint: Optional[Union[str, bool]] = None,
    seed: Optional[int] = None,
):
    r"""
    Plot model coefficients for variables interacted via "i()" syntax, with
    confidence intervals.

    Parameters
    ----------
    models : A supported model object (Feols, Fepois, Feiv, FixestMulti) or a list of
            Feols, Fepois & Feiv models.
    figsize : tuple or None, optional
        The size of the figure. If None, the default size is used.
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
    plot_backend: str, optional
        The plotting backend to use. Options are "lets_plot" (default if installed) and "matplotlib".
        If "lets_plot" is specified but not installed, an ImportError will be raised with instructions
        to install it or use "matplotlib" instead.
    rename_models : dict, optional
        A dictionary to rename the models. The keys are the original model names and the values the new names.
    labels: dict, optional
        A dictionary to relabel the variables. The keys in this dictionary are the original variable names, which correspond to the names stored in the `_coefnames` attribute of the model. The values in the dictionary are the new  names you want to assign to these variables.
        Note that interaction terms will also be relabeled using the labels of the individual variables.
        The renaming is applied after the selection of the coefficients via `keep` and `drop`.
    cat_template: str, optional
        Template to relabel categorical variables. None by default, which applies no relabeling.
        Other options include combinations of "{variable}" and "{value}", e.g. "{variable}::{value}"
        to mimic fixest encoding. But "{variable}--{value}" or "{variable}{value}" or just "{value}"
        are also possible.
    joint: str or bool, optional
        Whether to plot simultaneous confidence bands for the coefficients. If True, simultaneous confidence bands
        are plotted. If False, "standard" confidence intervals are plotted. If "both", both are plotted in
        one figure. Default is None, which returns the standard confidence intervals. Note that this option is
        not available for objects of type `FixestMulti`, i.e. multiple estimation.
    seed: int, optional
        The seed for the random number generator. Default is None. Only required / used when `joint` is True.

    Returns
    -------
    object
        A plot figure from the specified backend.

    Examples
    --------
    ```{python}
    import pyfixest as pf
    from pyfixest.report.utils import rename_categoricals

    df = pf.get_data()
    fit1 = pf.feols("Y ~ i(f1)", data = df)
    fit2 = pf.feols("Y ~ i(f1) + X2", data = df)
    fit3 = pf.feols("Y ~ i(f1) + X2 | f2", data = df)

    pf.iplot([fit1, fit2, fit3], labels = rename_categoricals(fit1._coefnames))
    pf.iplot(
        models = [fit1, fit2, fit3],
        labels = rename_categoricals(fit1._coefnames)
    )
    pf.iplot(
        models = [fit1, fit2, fit3],
        rename_models = {
            fit1._model_name_plot: "Model 1",
            fit2._model_name_plot: "Model 2",
            fit3._model_name_plot: "Model 3"
        },
    )
    pf.iplot(
        models = [fit1, fit2, fit3],
        rename_models = {
            "Y~i(f1)": "Model 1",
            "Y~i(f1)+X2": "Model 2",
            "Y~i(f1)+X2|f2": "Model 3"
        },
    )
    pf.iplot([fit1], joint = "both")
    ```
    """
    models = _post_processing_input_checks(
        models, check_duplicate_model_names=True, rename_models=rename_models
    )
    if joint not in [False, None] and len(models) > 1:
        raise ValueError(
            "The 'joint' parameter is only available for a single model, i.e. objects of type FixestMulti are not supported."
        )

    df_all: list[pd.DataFrame] = []
    all_icovars: list[str] = []

    if keep is None:
        keep = []

    if drop is None:
        drop = []

    if rename_models is None:
        rename_models = {}

    for x, fxst in enumerate(list(models)):
        if fxst._icovars is None:
            raise ValueError(
                f"The {x} th estimated model did not have ivars / 'i()' model syntax."
                "In consequence, the '.iplot()' method is not supported."
            )
        all_icovars += fxst._icovars

        df_model = _get_model_df(
            fxst=fxst, alpha=alpha, joint=joint, seed=seed, rename_models=rename_models
        )
        df_all.append(df_model)

    # drop duplicates
    all_icovars = list(set(all_icovars))

    df = pd.concat(df_all, axis=0)
    if keep or drop:
        idxs = _select_order_coefs(df["Coefficient"].tolist(), keep, drop, exact_match)
    else:
        idxs = df["Coefficient"]
    df = df.loc[df["Coefficient"].isin(idxs), :]
    # keep only coefficients interacted via the i() syntax
    df = df[df["Coefficient"].isin(all_icovars)].reset_index()

    # check that labels match the coef names
    if labels is not None:
        _check_label_keys_in_covars(
            label_keys=list(labels.keys()),
            covariate_names=df["Coefficient"].unique().tolist(),
        )

    return _coefplot(
        plot_backend=plot_backend,
        df=df,
        figsize=figsize,
        alpha=alpha,
        yintercept=yintercept,
        xintercept=xintercept,
        rotate_xticks=rotate_xticks,
        title=title,
        flip_coord=coord_flip,
        labels=labels,
        cat_template=cat_template,
        ax=ax,
    )


def coefplot(
    models: ModelInputType,
    alpha: float = 0.05,
    figsize: Optional[tuple[int, int]] = None,
    yintercept: float = 0,
    xintercept: Union[float, None] = None,
    rotate_xticks: int = 0,
    title: Optional[str] = None,
    coord_flip: bool = True,
    keep: Optional[Union[list, str]] = None,
    drop: Optional[Union[list, str]] = None,
    exact_match: bool = False,
    plot_backend: str = "lets_plot" if _HAS_LETS_PLOT else "matplotlib",
    labels: Optional[dict] = None,
    joint: Optional[Union[str, bool]] = None,
    seed: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    rename_models: Optional[dict[str, str]] = None,
):
    r"""
    Plot model coefficients with confidence intervals.

    Parameters
    ----------
    models : A supported model object (Feols, Fepois, Feiv, FixestMulti) or a list of
            Feols, Fepois & Feiv models.
    figsize : tuple or None, optional
        The size of the figure. If None, the default size is used.
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
    plot_backend: str, optional
        The plotting backend to use. Options are "lets_plot" (default if installed) and "matplotlib".
        If "lets_plot" is specified but not installed, an ImportError will be raised with instructions
        to install it or use "matplotlib" instead.
    rename_models : dict, optional
        A dictionary to rename the models. The keys are the original model names and the values the new names.
    labels: dict, optional
        A dictionary to relabel the variables. The keys in this dictionary are the original variable names, which correspond to the names stored in the `_coefnames` attribute of the model. The values in the dictionary are the new  names you want to assign to these variables.
        Note that interaction terms will also be relabeled using the labels of the individual variables.
        The renaming is applied after the selection of the coefficients via `keep` and `drop`.
    joint: str or bool, optional
        Whether to plot simultaneous confidence bands for the coefficients. If True, simultaneous confidence bands
        are plotted. If False, "standard" confidence intervals are plotted. If "both", both are plotted in
        one figure. Default is None, which returns the standard confidence intervals. Note that this option is
        not available for objects of type `FixestMulti`, i.e. multiple estimation.
    seed: int, optional
        The seed for the random number generator. Default is None. Only required / used when `joint` is True.

    Returns
    -------
    object
        A plot figure from the specified backend.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    df = pf.get_data()
    fit1 = pf.feols("Y ~ i(f1)", data = df)
    fit2 = pf.feols("Y ~ i(f1) + X2", data = df)
    fit3 = pf.feols("Y ~ i(f1) + X2 | f1", data = df)

    pf.iplot([fit1, fit2, fit3])
    pf.iplot(
        models = [fit1, fit2, fit3],
        rename_models = {
            fit1._model_name_plot: "Model 1",
            fit2._model_name_plot: "Model 2",
            fit3._model_name_plot: "Model 3"
        },
    )
    pf.iplot(
        models = [fit1, fit2, fit3],
        rename_models = {
            "Y~i(f1)": "Model 1",
            "Y~i(f1)+X2": "Model 2",
            "Y~i(f1)+X2|f1": "Model 3"
        },
    )
    pf.iplot([fit1], joint = "both")

    ```
    """
    models = _post_processing_input_checks(
        models, check_duplicate_model_names=True, rename_models=rename_models
    )
    if joint not in [False, None] and len(models) > 1:
        raise ValueError(
            "The 'joint' parameter is only available for a single model, i.e. objects of type FixestMulti are not supported."
        )

    if keep is None:
        keep = []

    if drop is None:
        drop = []

    if rename_models is None:
        rename_models = {}

    df_all = []
    for fxst in models:
        df_model = _get_model_df(
            fxst=fxst, alpha=alpha, joint=joint, seed=seed, rename_models=rename_models
        )
        df_all.append(df_model)

    df = pd.concat(df_all, axis=0).reset_index().set_index("Coefficient")
    if keep or drop:
        idxs = _select_order_coefs(df.index.tolist(), keep, drop, exact_match)
    else:
        idxs = df.index
    df = df.loc[idxs, :].reset_index()

    # check that labels match the coef names
    if labels is not None:
        _check_label_keys_in_covars(
            label_keys=list(labels.keys()),
            covariate_names=df["Coefficient"].unique().tolist(),
        )

    return _coefplot(
        plot_backend=plot_backend,
        df=df,
        figsize=figsize,
        alpha=alpha,
        yintercept=yintercept,
        xintercept=xintercept,
        rotate_xticks=rotate_xticks,
        title=title,
        flip_coord=coord_flip,
        labels=labels,
        ax=ax,
    )


def qplot(
    models: ModelInputType,
    rename_models: Optional[dict] = None,
    figsize: Optional[tuple] = None,
    ncol: Optional[int] = None,
    nrow: Optional[int] = None,
):
    """
    Plot regression quantiles.

    Parameters
    ----------
    models : A supported model object (Feols, Fepois, Feiv, FixestMulti) or a list of
            Feols, Fepois & Feiv models.
    figsize : tuple or None, optional
        The size of the figure. If None, the default size is (10, 6).
    rename_models : dict, optional
        A dictionary to rename the models. The keys are the original model names and the values the new names.
    ncol : int, optional
        Number of columns of subplots. Default is None. Note: cannot be set jointly with nrow argument.
    nrow : int, optional
        Number of rows of subplots. Default is None. Note: cannot be set jointly with ncol argument.

    Returns
    -------
    object
        A matplotplit figure.
    """
    if rename_models is None:
        rename_models = {}

    if figsize is None:
        figsize = (10, 6)

    models = _post_processing_input_checks(
        models, check_duplicate_model_names=True, rename_models=rename_models
    )

    df_all = pd.DataFrame()
    for model in models:
        if not isinstance(model, Quantreg):
            raise TypeError(
                "The 'qplot' function is only supported for objects of type Quantreg."
            )

        df = model.tidy()
        df["quantile"] = model._quantile
        df["model"] = model._model_name_plot

        df_all = pd.concat([df_all, df], axis=0)

    df_all.reset_index(inplace=True)

    return _qplot(
        data=df_all,
        figsize=figsize,
        nrow=nrow,
        ncol=ncol,
    )


def _coefplot(plot_backend, *, figsize, **plot_kwargs):
    """Coefplot function that dispatches to the correct plotting backend."""
    figsize = set_figsize(figsize, plot_backend)
    if plot_backend == "lets_plot":
        if not _HAS_LETS_PLOT:
            raise ImportError(
                "The 'lets_plot' package is required for the 'lets_plot' backend. "
                "Please install it with 'pip install lets-plot' or use the 'matplotlib' backend."
            )
        return _coefplot_lets_plot(figsize=figsize, **plot_kwargs)
    elif plot_backend == "matplotlib":
        return _coefplot_matplotlib(figsize=figsize, **plot_kwargs)
    else:
        raise ValueError("plot_backend must be either 'lets_plot' or 'matplotlib'.")


def _coefplot_lets_plot(
    df: pd.DataFrame,
    figsize: tuple[int, int],
    alpha: float,
    yintercept: Optional[int] = None,
    xintercept: Optional[int] = None,
    rotate_xticks: float = 0,
    title: Optional[str] = None,
    flip_coord: Optional[bool] = True,
    labels: Optional[dict] = None,
    cat_template: Optional[str] = None,
    ax=None,  # for compatibility with matplotlib backend
):
    """
    Plot model coefficients with confidence intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data used for the model fitting.
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
    flip_coord : bool, optional
        Whether to flip the coordinates of the plot. Default is True.
    labels : dict, optional
        A dictionary to relabel the variables. The keys are the original variable names and the values the new names.
    cat_template : str, optional
        Template to relabel categorical variables. None by default, which applies no relabeling.
        Other options include combinations of "{variable}" and "{value}", e.g. "{variable}::{value}"
        to mimic fixest encoding. But "{variable}--{value}" or "{variable}{value}" or just "{value}"
        are also possible.
    ax : None, optional
        Not used. Only for compatibility with the matplotlib backend.

    Returns
    -------
    object
        A lets-plot figure.
    """
    df.reset_index(inplace=True)
    df.rename(columns={"fml": "Model"}, inplace=True)
    ub, lb = 1 - alpha / 2, alpha / 2

    labels_dict = {} if labels is None else labels

    if not labels_dict or cat_template is not None:
        interactionSymbol = ":"
        df["Coefficient"] = df["Coefficient"].apply(
            lambda x: _relabel_expvar(
                x,
                labels_dict,
                interactionSymbol,
                cat_template if cat_template is not None else "",
            )
        )

    plot = (
        ggplot(df, aes(x="Coefficient", y="Estimate", color="Model"))
        + geom_point(position=position_dodge(0.5))
        + geom_errorbar(
            aes(ymin=str(round(lb * 100, 1)) + "%", ymax=str(round(ub * 100, 1)) + "%"),
            width=0.05,
            position=position_dodge(0.5),
        )
        + ylab(rf"Estimate and {round((1 - alpha) * 100, 1)}% Confidence Interval")
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


def _coefplot_matplotlib(
    df: pd.DataFrame,
    figsize: tuple[int, int],
    alpha: float,
    yintercept: Optional[int] = None,
    xintercept: Optional[int] = None,
    rotate_xticks: float = 0,
    title: Optional[str] = None,
    flip_coord: Optional[bool] = True,
    labels: Optional[dict] = None,
    cat_template: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    dodge: float = 0.5,
    **fig_kwargs,
) -> plt.Figure:
    """
    Plot model coefficients with confidence intervals, supporting multiple models.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data used for the model fitting.
        Must include a 'fml' column identifying different models.
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
    flip_coord : bool, optional
        Whether to flip the coordinates of the plot. Default is True.
    labels : dict, optional
        A dictionary to relabel the variables. The keys are the original variable names and the values the new names.
    cat_template : str, optional
        Template to relabel categorical variables. None by default, which applies no relabeling.
        Other options include combinations of "{variable}" and "{value}", e.g. "{variable}::{value}"
        to mimic fixest encoding. But "{variable}--{value}" or "{variable}{value}" or just "{value}"
        are also possible.
    dodge : float, optional
        The amount to dodge each model's points by. Default is 0.1.
    fig_kwargs : dict
        Additional keyword arguments to pass to the matplotlib figure.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib Figure object.
    """
    labels_dict = {} if labels is None else labels

    if not labels_dict or cat_template is not None:
        interactionSymbol = ":"
        df["Coefficient"] = df["Coefficient"].apply(
            lambda x: _relabel_expvar(
                x,
                labels_dict,
                interactionSymbol,
                cat_template if cat_template is not None else "",
            )
        )

    ub, lb = (f"{round(x * 100, 1)}%" for x in [1 - alpha / 2, alpha / 2])

    yintercept = yintercept if yintercept is not None else 0
    title = title if title is not None else "Coefficient Plot"

    if ax is None:
        f, ax = plt.subplots(figsize=figsize, **fig_kwargs)
    else:
        f = ax.get_figure()

    # Check if we have multiple models
    models = df["fml"].unique()
    is_multi_model = len(models) > 1

    colors = plt.cm.jet(np.linspace(0, 1, len(models)))
    color_dict = dict(zip(models, colors))

    # Calculate the positions for dodging
    unique_coefficients = df["Coefficient"].unique()
    if is_multi_model:
        coef_positions = {coef: i for i, coef in enumerate(unique_coefficients)}
    dodge_start = -(len(models) - 1) * dodge / 2

    for i, (model, group) in enumerate(df.groupby("fml")):
        color = color_dict[model]

        if is_multi_model:
            dodge_val = dodge_start + i * dodge
            x_pos = [coef_positions[coef] + dodge_val for coef in group["Coefficient"]]
        else:
            x_pos = list(map(float, range(len(group))))

        err = [group["Estimate"] - group[lb], group[ub] - group["Estimate"]]

        if flip_coord:
            ax.errorbar(
                x=group["Estimate"],
                y=x_pos,
                xerr=err,
                fmt="o",
                capsize=5,
                color=color,
                label=model if is_multi_model else "Estimates",
            )
        else:
            ax.errorbar(
                y=group["Estimate"],
                x=x_pos,
                # yerr=group["Std. Error"] * critval,
                yerr=err,
                fmt="o",
                capsize=5,
                color=color,
                label=model if is_multi_model else "Estimates",
            )

    if flip_coord:
        ax.axvline(x=yintercept, color="black", linestyle="--")
        if xintercept is not None:
            ax.axhline(y=xintercept, color="black", linestyle="--")
        ax.set_xlabel(
            rf"Estimate and {round((1 - alpha) * 100, 1)}% Confidence Interval"
        )
        ax.set_ylabel("Coefficient")
        ax.set_yticks(range(len(unique_coefficients)))
        ax.set_yticklabels(unique_coefficients)
        ax.tick_params(axis="y", rotation=rotate_xticks)
    else:
        ax.axhline(y=yintercept, color="black", linestyle="--")
        if xintercept is not None:
            ax.axvline(x=xintercept, color="black", linestyle="--")
        ax.set_ylabel(
            rf"Estimate and {round((1 - alpha) * 100, 1)}% Confidence Interval"
        )
        ax.set_xlabel("Coefficient")
        ax.set_xticks(range(len(unique_coefficients)))
        ax.set_xticklabels(unique_coefficients)
        ax.tick_params(axis="x", rotation=rotate_xticks)

    ax.set_title(title)
    if is_multi_model:
        ax.legend()
    plt.tight_layout()
    plt.close()
    return f


def _qplot(
    data: pd.DataFrame,
    nrow: Optional[int] = None,
    ncol: Optional[int] = None,
    figsize: tuple[int, int] = (10, 6),
):
    """
    Plot coefficient estimates by quantile with shaded 95 % CIs.

    Parameters
    ----------
    data : DataFrame
        Columns required: ['Coefficient', 'quantile',
                          'Estimate', '2.5%', '97.5%'].
    nrow, ncol : int, optional
        Subplot layout.  Pass exactly one of them, or neither
        (defaults to a single row).
    figsize : tuple, optional
        Passed straight to ``plt.subplots``.

    Returns
    -------
    (Figure, ndarray[Axes])
        Handle to the created figure and axes.
    """
    if nrow is None and ncol is None:
        nrow = 1
    if (nrow is not None) and (ncol is not None):
        raise ValueError("Specify only one of nrow or ncol, not both.")

    coeffs = data["Coefficient"].unique().tolist()
    k = len(coeffs)

    if nrow is not None:
        assert nrow is not None  # for mypy, do people really do this?
        rows, cols = nrow, math.ceil(k / nrow)
    else:
        assert ncol is not None
        cols, rows = ncol, math.ceil(k / ncol)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, sharey="all")
    axes = axes.ravel()

    cmp = plt.get_cmap("Set1")

    for i, coef in enumerate(coeffs):
        ax = axes[i]
        sub = data.loc[data["Coefficient"] == coef].sort_values("quantile")

        q = sub["quantile"].to_numpy(float)
        est = sub["Estimate"].to_numpy(float)
        lo = sub["2.5%"].to_numpy(float)
        hi = sub["97.5%"].to_numpy(float)

        color = cmp(0)
        ax.plot(q, est, marker="o", label="Estimate", color=color)
        ax.fill_between(q, lo, hi, alpha=0.3, color=color)

        ax.axhline(0, color="black", lw=1, ls="--")
        ax.set_title(coef)
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Coefficient (95 % CI)")

    for j in range(k, rows * cols):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig, axes


def _get_model_df(
    fxst: Union[Feols, Fepois, Feiv],
    alpha: float,
    joint: Optional[Union[str, bool]],
    seed: Optional[int] = None,
    rename_models: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Get a tidy model frame as input to the _coefplot function.

    Parameters
    ----------
    fxst : Union[Feols, Fepois, Feiv]
        The fitted model.
    alpha : float
        The significance level for the confidence intervals.
    joint : Optional[Union[str, bool]]
        Whether to plot simultaneous confidence bands for the coefficients. If True, simultaneous confidence bands
        are plotted. If False, "standard" confidence intervals are plotted. If "both", both are plotted in
        one figure. Default is None, which returns the standard confidence intervals. Note that this option is
        not available for objects of type `FixestMulti`, i.e. multiple estimation.
    seed : int, optional
        The seed for the random number generator. Default is None. Only required / used when `joint` is True.
    rename_models : dict, optional
        A dictionary to rename the models. The keys are the original model names and the values the new names.

    Returns
    -------
    pd.DataFrame
        A tidy model frame.
    """
    if rename_models is None:
        rename_models = {}

    df_model = fxst.tidy(alpha=alpha).reset_index()  # Coefficient -> simple column

    df_model["fml"] = fxst._model_name_plot
    df_model["fml"] = df_model["fml"].apply(lambda x: rename_models.get(x, x))

    if joint in ["both", True]:
        lb, ub = f"{alpha / 2 * 100:.1f}%", f"{(1 - alpha / 2) * 100:.1f}%"
        df_joint = fxst.confint(joint=True, alpha=alpha, seed=seed)
        df_joint.reset_index(inplace=True)
        df_joint = df_joint.rename(columns={"index": "Coefficient"})
        df_joint_full = (
            df_model.copy()
            .drop([lb, ub], axis=1)
            .merge(df_joint, on="Coefficient", how="left")
        )

        df_joint_full["fml"] += " (joint CIs)"  # type: ignore[operator]

        if joint == "both":
            df_model = pd.concat([df_model, df_joint_full], axis=0)
        else:
            df_model = df_joint_full

    return df_model
