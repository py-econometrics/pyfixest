import warnings
from collections import Counter
from collections.abc import ValuesView
from typing import Optional, Union

import maketables
import numpy as np
import pandas as pd

from pyfixest.estimation.feiv_ import Feiv
from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.fepois_ import Fepois
from pyfixest.estimation.FixestMulti_ import FixestMulti

ModelInputType = Union[
    FixestMulti, Feols, Fepois, Feiv, list[Union[Feols, Fepois, Feiv]]
]


def etable(
    models: ModelInputType,
    type: str = "gt",
    signif_code: Optional[list] = None,
    coef_fmt: str = "b \n (se)",
    custom_stats: Optional[dict] = None,
    custom_model_stats: Optional[dict] = None,
    keep: Optional[Union[list, str]] = None,
    drop: Optional[Union[list, str]] = None,
    exact_match: Optional[bool] = False,
    labels: Optional[dict] = None,
    cat_template: Optional[str] = None,
    show_fe: Optional[bool] = True,
    felabels: Optional[dict] = None,
    fe_present: str = "x",
    fe_absent: str = "-",
    notes: str = "",
    model_heads: Optional[list] = None,
    head_order: Optional[str] = "dh",
    file_name: Optional[str] = None,
    **kwargs,
) -> Union[pd.DataFrame, str, None]:
    r"""
    Generate a table summarizing the results of multiple regression models.

    This function uses the maketables package internally to create publication-ready
    regression tables. It supports various output formats including HTML (via Great Tables),
    markdown, and LaTeX.

    Parameters
    ----------
    models : A supported model object (Feols, Fepois, Feiv, FixestMulti) or a list of
            Feols, Fepois & Feiv models.
        The models to be summarized in the table.
    type : str, optional
        Type of output. Either "df" for pandas DataFrame, "md" for markdown,
        "gt" for great_tables, or "tex" for LaTeX table. Default is "gt".
    signif_code : list, optional
        Significance levels for the stars. Default is None, which sets [0.001, 0.01, 0.05].
        If None, no stars are printed.
    coef_fmt : str, optional
        The format of the coefficient (b), standard error (se), t-stats (t), and
        p-value (p). Default is `"b \n (se)"`.
        Spaces ` `, parentheses `()`, brackets `[]`, newlines `\n` are supported.
    custom_stats: dict, optional
        A dictionary of custom statistics that can be used in the coef_fmt string to be displayed
        in the coefficuent cells analogously to "b", "se" etc. The keys are the names of the custom
        statistics, and the values are lists of lists, where each inner list contains the custom
        statistic values for all coefficients each model.
        Note that "b", "se", "t", or "p" are reserved and cannot be used as keys.
    custom_model_stats: dict, optional
        A dictionary of custom model statistics or model information displayed in a new line in the
        bottom panel of the table. The keys are the names of the statistics (i.e. entry in the first column)
        and the values are a lists of the same length as the number of models. Default is None.
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
    labels: dict, optional
        A dictionary to relabel the variables. The keys in this dictionary are the
        original variable names, which correspond to the names stored in the `_coefnames`
        attribute of the model. The values in the dictionary are the new names you want
        to assign to these variables.
        Note that interaction terms will also be relabeled using the labels of the individual variables.
        The command is applied after the `keep` and `drop` commands.
    cat_template: str, optional
        Template to relabel categorical variables. None by default, which applies no relabeling.
        Other options include combinations of "{variable}" and "{value}", e.g. "{variable}::{value}"
        to mimic fixest encoding. But "{variable}--{value}" or "{variable}{value}" or just "{value}"
        are also possible.
    show_fe: bool, optional
        Whether to show the rows with fixed effects markers. Default is True.
    felabels: dict, optional
        A dictionary to relabel the fixed effects. Only needed if you want to relabel
        the FE lines with a different label than the one specied for the respective
        variable in the labels dictionary.
        The command is applied after the `keep` and `drop` commands.
    fe_present: str, optional
        Symbol to use when a fixed effect is present in a model. Default is "x".
        Common alternatives include "Y", "YES", "✓", "✅", or any custom string.
    fe_absent: str, optional
        Symbol to use when a fixed effect is absent from a model. Default is "-".
        Common alternatives include "N", "NO", "✗", "", or any custom string.
    digits: int
        The number of digits to round to.
    thousands_sep: bool, optional
        The thousands separator. Default is False.
    scientific_notation: bool, optional
        Whether to use scientific notation. Default is True.
    scientific_notation_threshold: int, optional
        The threshold for using scientific notation. Default is 10_000.
    notes: str, optional
        Custom table notes. Default shows the significance levels and the format of
        the coefficient cell.
    model_heads: list, optional
        Add custom headlines to models when output as df or latex. Length of list
        must correspond to number of models. Default is None.
    head_order: str, optional
        String to determine the display of the table header when output as df or latex.
        Allowed values are "dh", "hd", "d", "h", or "". When head_order is "dh",
        the dependent variable is displayed first, followed by the custom model_heads
        (provided the user has specified them). With "hd" it is the other way around.
        When head_order is "d", only the dependent variable and model numbers are displayed
        and with "" only the model numbers. Default is "dh".
    file_name: str, optional
        The name/path of the file to save the LaTeX table to. Default is None.

    Returns
    -------
    pandas.DataFrame
        A styled DataFrame with the coefficients and standard errors of the models.
        When output is "tex", the LaTeX code is returned as a string.

    Examples
    --------
    For more examples, take a look at the [regression tables and summary statistics vignette](https://py-econometrics.github.io/pyfixest/table-layout.html).

    ```{python}
    import pyfixest as pf

    # load data
    df = pf.get_data()
    fit1 = pf.feols("Y~X1 + X2 | f1", df)
    fit2 = pf.feols("Y~X1 + X2 | f1 + f2", df)

    pf.etable([fit1, fit2])
    ```
    """
    # Apply pyfixest default for signif_code (different from maketables default)
    if signif_code is None:
        signif_code = [0.001, 0.01, 0.05]

    assert isinstance(signif_code, list) and len(signif_code) == 3, (
        "signif_code must be a list of length 3"
    )
    if signif_code:
        assert all([0 < i < 1 for i in signif_code]), (
            "All values of signif_code must be between 0 and 1"
        )
    if signif_code:
        assert signif_code[0] < signif_code[1] < signif_code[2], (
            "signif_code must be in increasing order"
        )

    assert type in [
        "df",
        "tex",
        "md",
        "html",
        "gt",
    ], "type must be either 'df', 'md', 'html', 'gt' or 'tex'"

    models_list = _post_processing_input_checks(models)

    if model_heads is not None:
        assert len(model_heads) == len(models_list), (
            "model_heads must have the same length as models"
        )

    assert head_order in [
        "dh",
        "hd",
        "d",
        "h",
        "",
    ], "head_order must be one of 'd', 'h', 'dh', 'hd', ''"

    if custom_model_stats is not None:
        assert isinstance(custom_model_stats, dict), "custom_model_stats must be a dict"
        for stat, values in custom_model_stats.items():
            assert isinstance(stat, str), "custom_model_stats keys must be strings"
            assert isinstance(values, list), "custom_model_stats values must lists"
            assert len(values) == len(models_list), (
                "lists in custom_model_stats values must have the same length as models"
            )

    table = maketables.ETable(
        models=models_list,
        signif_code=signif_code,
        coef_fmt=coef_fmt,
        custom_stats=custom_stats,
        custom_model_stats=custom_model_stats,
        keep=keep if keep else [],
        drop=drop if drop else [],
        exact_match=exact_match,
        labels=labels,
        cat_template=cat_template,
        show_fe=show_fe,
        felabels=felabels,
        fe_present=fe_present,
        fe_absent=fe_absent,
        notes=notes,
        model_heads=model_heads,
        head_order=head_order,
        **kwargs,
    )

    if type == "df":
        return table.df
    elif type == "md":
        result = table.df.to_markdown()
        print(result)
        return None
    elif type == "tex":
        result = table.make(type="tex")
        if file_name is not None:
            with open(file_name, "w") as f:
                f.write(result)
        return result
    elif type == "html":
        return table.make(type="html")
    elif type == "gt":
        result = table.make(type="gt")
        if file_name is not None:
            with open(file_name, "w") as f:
                f.write(result.as_raw_html())
        return result

    return None


def summary(models: ModelInputType, digits: int = 3) -> None:
    """
    Print a summary of estimation results for each estimated model.

    For each model, this method prints a header indicating the fixed-effects and the
    dependent variable, followed by a table of coefficient estimates with standard
    errors, t-values, and p-values.

    Parameters
    ----------
    models : A supported model object (Feols, Fepois, Feiv, FixestMulti) or a list of
            Feols, Fepois & Feiv models.
    digits : int, optional
        The number of decimal places to round the summary statistics to. Default is 3.

    Returns
    -------
    None

    Examples
    --------
    ```{python}
    import pyfixest as pf

    # load data
    df = pf.get_data()
    fit1 = pf.feols("Y~X1 + X2 | f1", df)
    fit2 = pf.feols("Y~X1 + X2 | f1 + f2", df)
    fit3 = pf.feols("Y~X1 + X2 | f1 + f2 + f3", df)

    pf.summary([fit1, fit2, fit3])
    ```
    """
    models = _post_processing_input_checks(models)

    for fxst in list(models):
        depvar = fxst._depvar

        df = fxst.tidy().round(digits)

        if fxst._method == "feols":
            estimation_method = "IV" if fxst._is_iv else "OLS"
        elif fxst._method == "fepois":
            estimation_method = "Poisson"
        elif fxst._method == "twfe":
            estimation_method = "TWFE"
        elif fxst._method == "did2s":
            estimation_method = "DID2S"
        elif "quantreg" in fxst._method:
            estimation_method = f"quantreg: q = {fxst._quantile}"  # type: ignore
        else:
            raise ValueError("Unknown estimation method.")
        print("###")
        print("")
        print("Estimation: ", estimation_method)
        depvar_fixef = f"Dep. var.: {depvar}"
        if fxst._fixef is not None:
            if not fxst._use_mundlak:
                depvar_fixef += f", Fixed effects: {fxst._fixef}"
            else:
                depvar_fixef += f", Mundlak: by {fxst._fixef}"
        print(depvar_fixef)
        if fxst._sample_split_value != "all":
            split = f"sample: {fxst._sample_split_var} = {fxst._sample_split_value}"
            print(split)
        print("Inference: ", fxst._vcov_type_detail)
        print("Observations: ", fxst._N)
        print("")
        print(df.to_markdown(floatfmt=f".{digits}f"))
        print("---")

        to_print = ""

        if not np.isnan(fxst._rmse):
            to_print += f"RMSE: {np.round(fxst._rmse, digits)} "
        if not np.isnan(fxst._r2):
            to_print += f"R2: {np.round(fxst._r2, digits)} "
        if not np.isnan(fxst._r2_within):
            to_print += f"R2 Within: {np.round(fxst._r2_within, digits)} "
        if fxst.deviance is not None:
            deviance_value = np.asarray(fxst.deviance).squeeze()
            to_print += f"Deviance: {np.round(deviance_value, digits)} "

        print(to_print)


def _post_processing_input_checks(
    models: ModelInputType,
    check_duplicate_model_names: bool = False,
    rename_models: Optional[dict[str, str]] = None,
) -> list[Union[Feols, Fepois, Feiv]]:
    """
    Perform input checks for post-processing models.

    Parameters
    ----------
        models : Union[List[Union[Feols, Fepois, Feiv]], FixestMulti]
                The models to be checked. This can either be a list of models
                (Feols, Fepois, Feiv) or a single FixestMulti object.
        check_duplicate_model_names : bool, optional
                Whether to check for duplicate model names. Default is False.
                Mostly used to avoid overlapping models in plots created via
                pf.coefplot() and pf.iplot().
        rename_models : dict, optional
                A dictionary to rename the models. The keys are the original model names
                and the values are the new model names.

    Returns
    -------
        List[Union[Feols, Fepois]]
            A list of checked and validated models. The returned list contains only
            Feols and Fepois types.

    Raises
    ------
        TypeError: If the models argument is not of the expected type.

    """
    models_list: list[Union[Feols, Fepois, Feiv]] = []

    if isinstance(models, (Feols, Fepois, Feiv)):
        models_list = [models]
    elif isinstance(models, FixestMulti):
        models_list = models.to_list()
    elif isinstance(models, (list, ValuesView)):
        if all(isinstance(m, (Feols, Fepois, Feiv)) for m in models):
            models_list = models
        else:
            raise TypeError(
                "All elements in the models list must be instances of Feols, Feiv, or Fepois."
            )
    else:
        raise TypeError("Invalid type for models argument.")

    if check_duplicate_model_names or rename_models is not None:
        all_model_names = [model._model_name for model in models_list]

    if check_duplicate_model_names:
        # create model_name_plot attribute to differentiate between models with the
        # same model_name / model formula
        for model in models_list:
            model._model_name_plot = model._model_name

        counter = Counter(all_model_names)
        duplicate_model_names = [item for item, count in counter.items() if count > 1]

        for duplicate_model in duplicate_model_names:
            duplicates = [
                model for model in models_list if model._model_name == duplicate_model
            ]
            for i, model in enumerate(duplicates):
                model._model_name_plot = f"Model {i}: {model._model_name}"
                warnings.warn(
                    f"The _model_name attribute {model._model_name}' is duplicated for models in the `models` you provided. To avoid overlapping model names / plots, the _model_name_plot attribute has been changed to '{model._model_name_plot}'."
                )

        if rename_models is not None:
            model_name_diff = set(rename_models.keys()) - set(all_model_names)
            if model_name_diff:
                warnings.warn(
                    f"""
                    The following model names specified in rename_models are not found in the models:
                    {model_name_diff}
                    """
                )

    return models_list


def dtable(
    df: pd.DataFrame,
    vars: list,
    stats: Optional[list] = None,
    bycol: Optional[list[str]] = None,
    byrow: Optional[str] = None,
    type: str = "gt",
    labels: dict | None = None,
    stats_labels: dict | None = None,
    digits: int = 2,
    notes: str = "",
    counts_row_below: bool = False,
    hide_stats: bool = False,
    **kwargs,
):
    r"""
    Generate descriptive statistics tables and create a booktab style table in
    the desired format (gt or tex).

    .. deprecated:: 0.41.0
        This function is deprecated and will be removed in a future version.
        Please use `maketables.DTable()` directly instead.
        See https://py-econometrics.github.io/maketables/ for documentation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the table to be displayed.
    vars : list
        List of variables to be included in the table.
    stats : list, optional
        List of statistics to be calculated. The default is None, that sets ['count','mean', 'std'].
        All pandas aggregation functions are supported.
    bycol : list, optional
        List of variables to be used to group the data by columns. The default is None.
    byrow : str, optional
        Variable to be used to group the data by rows. The default is None.
    type : str, optional
        Type of table to be created. The default is 'gt'.
        Type can be 'gt' for great_tables, 'tex' for LaTeX or 'df' for dataframe.
    labels : dict, optional
        Dictionary containing the labels for the variables. The default is None.
    stats_labels : dict, optional
        Dictionary containing the labels for the statistics. The default is None.
        The function uses a default labeling which will be replaced by the labels
        in the dictionary.
    digits : int, optional
        Number of decimal places to round the statistics to. The default is 2.
    notes : str
        Table notes to be displayed at the bottom of the table.
    counts_row_below : bool
        Whether to display the number of observations at the bottom of the table.
        Will only be carried out when each var has the same number of obs and when
        byrow is None. The default is False
    hide_stats : bool
        Whether to hide the names of the statistics in the table header. When stats
        are hidden and the user provides no notes string the labels of the stats are
        listed in the table notes. The default is False.
    kwargs : dict
        Additional arguments to be passed to maketables.DTable.

    Returns
    -------
    A table in the specified format.

    Examples
    --------
    For more examples, take a look at the [regression tables and summary statistics vignette](https://py-econometrics.github.io/pyfixest/table-layout.html).

    ```{python}
    import pyfixest as pf

    # load data
    df = pf.get_data()
    pf.dtable(df, vars = ["Y", "X1", "X2", "f1"])
    ```
    """
    warnings.warn(
        "pf.dtable() is deprecated and will be removed in a future version. "
        "Please use maketables.DTable() directly. "
        "See https://py-econometrics.github.io/maketables/ for documentation.",
        FutureWarning,
        stacklevel=2,
    )

    table = maketables.DTable(
        df=df,
        vars=vars,
        stats=stats,
        bycol=bycol,
        byrow=byrow,
        labels=labels,
        stats_labels=stats_labels,
        digits=digits,
        notes=notes,
        counts_row_below=counts_row_below,
        hide_stats=hide_stats,
        **kwargs,
    )

    # Handle output based on type parameter
    if type == "df":
        return table.df
    elif type == "gt":
        return table.make(type="gt")
    elif type == "tex":
        return table.make(type="tex")
    elif type == "html":
        return table.make(type="html")

    return table.make(type="gt")
