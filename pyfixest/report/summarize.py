import math
import re
import warnings
from collections import Counter
from collections.abc import ValuesView
from typing import Optional, Union

import numpy as np
import pandas as pd
from tabulate import tabulate

from pyfixest.estimation.feiv_ import Feiv
from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.fepois_ import Fepois
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.report.make_table import make_table
from pyfixest.report.utils import _relabel_expvar
from pyfixest.utils.dev_utils import _select_order_coefs

ModelInputType = Union[
    FixestMulti, Feols, Fepois, Feiv, list[Union[Feols, Fepois, Feiv]]
]


def etable(
    models: ModelInputType,
    type: str = "gt",
    signif_code: Optional[list] = None,
    coef_fmt: str = "b \n (se)",
    model_stats: Optional[list[str]] = None,
    model_stats_labels: Optional[dict[str, str]] = None,
    custom_stats: Optional[dict] = None,
    custom_model_stats: Optional[dict] = None,
    keep: Optional[Union[list, str]] = None,
    drop: Optional[Union[list, str]] = None,
    exact_match: Optional[bool] = False,
    labels: Optional[dict] = None,
    cat_template: Optional[str] = None,
    show_fe: Optional[bool] = True,
    show_se_type: Optional[bool] = True,  # legacy (ignored when model_stats provided)
    felabels: Optional[dict] = None,
    notes: str = "",
    model_heads: Optional[list] = None,
    head_order: Optional[str] = "dh",
    file_name: Optional[str] = None,
    **kwargs,
) -> Union[pd.DataFrame, str, None]:
    r"""
    Generate a table summarizing the results of multiple regression models.
    It supports various output formats including html (via great tables),  markdown, and LaTeX.

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
    model_stats: Optional[list[str]] = None,
        A list of model statistics to include in the table which will be displayed in the determined order. Names must match the model's respective attribute names (without leading "_") such as "r2", "adj_r2", "N", ...
    model_stats_labels: Optional[dict[str, str]] = None,
        A dictionary mapping model statistic names to display labels. If None, the default names are used.
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
    show_se_type: bool, optional
        Whether to show the rows with standard error type. Default is True.
    felabels: dict, optional
        A dictionary to relabel the fixed effects. Only needed if you want to relabel
        the FE lines with a different label than the one specied for the respective
        variable in the labels dictionary.
        The command is applied after the `keep` and `drop` commands.
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

    cat_template = "" if cat_template is None else cat_template

    models = _post_processing_input_checks(models)

    if labels is None:
        labels = {}
    if custom_stats is None:
        custom_stats = dict()
    if keep is None:
        keep = []
    if drop is None:
        drop = []

    if custom_stats:
        assert isinstance(custom_stats, dict), "custom_stats must be a dict"
        for key in custom_stats:
            assert isinstance(custom_stats[key], list), (
                "custom_stats values must be a list"
            )
            assert len(custom_stats[key]) == len(models), (
                f"custom_stats {key} must have the same number as models"
            )

    assert type in [
        "df",
        "tex",
        "md",
        "html",
        "gt",
    ], "type must be either 'df', 'md', 'html', 'gt' or 'tex'"

    if model_heads is not None:
        assert len(model_heads) == len(models), (
            "model_heads must have the same length as models"
        )

    # Check if head_order is allowed string & remove h when no model_heads provided
    assert head_order in [
        "dh",
        "hd",
        "d",
        "h",
        "",
    ], "head_order must be one of 'd', 'h', 'dh', 'hd', ''"
    if model_heads is None and "h" in head_order:
        head_order = head_order.replace("h", "")

    # Check if custom_model_stats is a dictionary and the provided lists have the same length as models
    if custom_model_stats is not None:
        assert isinstance(custom_model_stats, dict), "custom_model_stats must be a dict"
        for stat, values in custom_model_stats.items():
            assert isinstance(stat, str), "custom_model_stats keys must be strings"
            assert isinstance(values, list), "custom_model_stats values must lists"
            assert len(values) == len(models), (
                "lists in custom_model_stats values must have the same length as models"
            )

    # Collect info needed for coefficients & fixed effects
    dep_var_list: list[str] = []
    fixef_list: list[str] = []

    # Output-type dependent symbols
    if type in ["gt", "html"]:
        interactionSymbol = " &#215; "
        lbcode = "<br>"
    elif type == "tex":
        interactionSymbol = " $\\times$ "
        lbcode = r"\\"
    else:
        interactionSymbol = " x "
        lbcode = "\n"

    # Pre-scan models (only once)
    for model in models:
        dep_var_list.append(model._depvar)
        if model._fixef is not None and model._fixef != "0":
            fixef_list += model._fixef.split("+")

    # Fixed effects set
    if show_fe:
        fixef_list = [x for x in fixef_list if x]
        fixef_list = list(set(fixef_list))
        n_fixef = len(fixef_list)
    else:
        fixef_list = []
        n_fixef = 0

    # Determine default model stats (legacy emulation) if user did not provide any
    if model_stats is None:
        any_within = any(
            hasattr(m, "_r2_within")
            and not math.isnan(getattr(m, "_r2_within", float("nan")))
            for m in models
        )
        # Legacy order
        model_stats = ["N"]
        if show_se_type:
            model_stats.append("se_type")
        model_stats += ["r2", "r2_within" if any_within else "adj_r2"]

    assert isinstance(model_stats, (list, tuple)), "model_stats must be list-like"
    model_stats = list(model_stats)
    assert all(isinstance(s, str) for s in model_stats), (
        "model_stats entries must be strings"
    )
    # Assert that there are no duplicates in model_stats
    assert len(model_stats) == len(set(model_stats)), (
        "model_stats contains duplicate entries"
    )

    # Default labels by output type
    def _default_label(stat: str) -> str:
        if type in ("gt", "html"):
            mapping = {
                "N": "Observations",
                "se_type": "S.E. type",
                "r2": "R<sup>2</sup>",
                "adj_r2": "Adj. R<sup>2</sup>",
                "r2_within": "R<sup>2</sup> Within",
            }
        elif type == "tex":
            mapping = {
                "N": "Observations",
                "se_type": "S.E. type",
                "r2": "$R^2$",
                "adj_r2": "Adj. $R^2$",
                "r2_within": "$R^2$ Within",
            }
        else:
            mapping = {
                "N": "Observations",
                "se_type": "S.E. type",
                "r2": "R2",
                "adj_r2": "Adj. R2",
                "r2_within": "R2 Within",
            }
        return mapping.get(stat, stat)

    model_stats_rows: dict[str, list[str]] = {}
    for stat in model_stats:
        values = [_extract(m, stat) for m in models]
        label = _default_label(stat)
        if model_stats_labels and stat in model_stats_labels:
            label = model_stats_labels[stat]
        model_stats_rows[label] = values

    # Build custom model stats first (if any)
    if custom_model_stats is not None and len(custom_model_stats) > 0:
        # Values already validated for correct length earlier
        custom_df = pd.DataFrame.from_dict(custom_model_stats, orient="index")
    else:
        custom_df = pd.DataFrame()

    # Builtin / attribute stats
    if model_stats_rows:
        builtin_df = pd.DataFrame.from_dict(model_stats_rows, orient="index")
    else:
        builtin_df = pd.DataFrame()

    # Combine (custom first)
    if not custom_df.empty and not builtin_df.empty:
        model_stats_df = pd.concat([custom_df, builtin_df], axis=0)
    elif not custom_df.empty:
        model_stats_df = custom_df
    else:
        model_stats_df = builtin_df

    # Ensure index name consistency
    if model_stats_df.index.name is None:
        model_stats_df.index.name = None

    n_model_stats = model_stats_df.shape[0]

    # Create a dataframe for the Fixed Effects markers (fixed implementation)
    if show_fe and fixef_list:
        fe_rows = {}
        for fixef in fixef_list:
            row = []
            for model in models:
                has = (
                    model._fixef is not None
                    and fixef in model._fixef.split("+")
                    and not model._use_mundlak
                )
                row.append("x" if has else "-")
            fe_rows[fixef] = row
        fe_df = pd.DataFrame.from_dict(fe_rows, orient="index")
    else:
        fe_df = pd.DataFrame()
        show_fe = False

    # Finally, collect & format estimated coefficients and standard errors etc.
    coef_fmt_elements, coef_fmt_title = _parse_coef_fmt(coef_fmt, custom_stats)
    etable_list = []
    for i, model in enumerate(models):
        model_tidy_df = model.tidy()
        model_tidy_df.reset_index(
            inplace=True
        )  # If rounding here and p = 0.0499, it will be rounded to 0.05 and miss threshold.
        model_tidy_df["stars"] = (
            np.where(
                model_tidy_df["Pr(>|t|)"] < signif_code[0],
                "***",
                np.where(
                    model_tidy_df["Pr(>|t|)"] < signif_code[1],
                    "**",
                    np.where(model_tidy_df["Pr(>|t|)"] < signif_code[2], "*", ""),
                ),
            )
            if signif_code
            else ""
        )
        model_tidy_df[coef_fmt_title] = ""
        for element in coef_fmt_elements:
            if element == "b":
                model_tidy_df[coef_fmt_title] += (
                    model_tidy_df["Estimate"].apply(_number_formatter, **kwargs)
                    + model_tidy_df["stars"]
                )
            elif element == "se":
                model_tidy_df[coef_fmt_title] += model_tidy_df["Std. Error"].apply(
                    _number_formatter, **kwargs
                )
            elif element == "t":
                model_tidy_df[coef_fmt_title] += model_tidy_df["t value"].apply(
                    _number_formatter, **kwargs
                )
            elif element == "p":
                model_tidy_df[coef_fmt_title] += model_tidy_df["Pr(>|t|)"].apply(
                    _number_formatter, **kwargs
                )
            elif element in custom_stats:
                assert len(custom_stats[element][i]) == len(
                    model_tidy_df["Estimate"]
                ), (
                    f"custom_stats {element} has unequal length to the number of coefficients in model_tidy_df {i}"
                )
                model_tidy_df[coef_fmt_title] += pd.Series(
                    custom_stats[element][i]
                ).apply(_number_formatter, **kwargs)
            elif element == "\n":  # Replace output specific code for newline
                model_tidy_df[coef_fmt_title] += lbcode
            else:
                model_tidy_df[coef_fmt_title] += element
        model_tidy_df[coef_fmt_title] = pd.Categorical(model_tidy_df[coef_fmt_title])
        model_tidy_df = model_tidy_df[["Coefficient", coef_fmt_title]]
        model_tidy_df = pd.melt(
            model_tidy_df,
            id_vars=["Coefficient"],
            var_name="Metric",
            value_name=f"est{i + 1}",
        )
        model_tidy_df = model_tidy_df.drop("Metric", axis=1).set_index("Coefficient")
        etable_list.append(model_tidy_df)

    res = pd.concat(etable_list, axis=1)
    if keep or drop:
        idxs = _select_order_coefs(res.index.tolist(), keep, drop, exact_match)
    else:
        idxs = res.index
    res = res.loc[idxs, :].reset_index()
    # a lot of work to replace the NaNs with empty strings
    # reason: "" not a level of the category, might lead to a pandas error
    for column in res.columns:
        if (
            isinstance(res[column].dtype, pd.CategoricalDtype)
            and "" not in res[column].cat.categories
        ):
            res[column] = res[column].cat.add_categories([""])

        # Replace NA values with the empty string
        res[column] = res[column].fillna("")

    res.rename(columns={"Coefficient": "index"}, inplace=True)
    res.set_index("index", inplace=True)

    # Move the intercept row (if there is one) to the bottom of the table
    if "Intercept" in res.index:
        intercept_row = res.loc["Intercept"]
        res = res.drop("Intercept")
        res = pd.concat([res, pd.DataFrame([intercept_row])])

    # Relabel variables
    if (labels != {}) or (cat_template != ""):
        # Relabel dependent variables
        dep_var_list = [labels.get(k, k) for k in dep_var_list]

        # Relabel explanatory variables
        res_index = res.index.to_series()
        res_index = res_index.apply(
            lambda x: _relabel_expvar(x, labels or {}, interactionSymbol, cat_template)
        )
        res.set_index(res_index, inplace=True)

    # Relabel fixed effects
    if show_fe:
        if felabels is None:
            felabels = dict()
        if labels is None:
            labels = dict()
        fe_index = fe_df.index.to_series()
        fe_index = fe_index.apply(lambda x: felabels.get(x, labels.get(x, x)))
        fe_df.set_index(fe_index, inplace=True)

    # Ensure model_stats_df columns align after coefficient construction:
    # Allow user to pass model_stats = [] (no model stats displayed).
    # In that case model_stats_df is (0, 0) and assigning columns would raise a length mismatch.
    if model_stats_df.shape[1] == 0:
        # Create an empty frame with the correct columns so later concatenation works.
        model_stats_df = pd.DataFrame(
            index=pd.Index([], name=res.index.name), columns=res.columns
        )
    else:
        model_stats_df.columns = res.columns
    # Also align fixed effects dataframe columns
    if show_fe and not fe_df.empty:
        fe_df.columns = res.columns

    depvars = pd.DataFrame({"depvar": dep_var_list}).T
    depvars.columns = res.columns

    if type == "df":
        res_all = pd.concat([depvars, res, fe_df, model_stats_df])
        return res_all
    elif type == "md":
        res_all = pd.concat([depvars, res, fe_df, model_stats_df]).reset_index()
        # Generate notes string if user has not provided any
        if notes is None:
            if signif_code:
                notes = f"Significance levels: * p < {signif_code[2]}, ** p < {signif_code[1]}, *** p < {signif_code[0]}"
            else:
                notes = f"Format of coefficient cell: {coef_fmt_title}"
        res_all = _tabulate_etable_md(
            df=res_all,
            n_coef=res.shape[0],
            n_fixef=n_fixef,
            n_models=len(models),
            n_model_stats=n_model_stats,
        )
        print(res_all)
        print(notes)
        return None

    elif type in ["tex", "gt"]:
        # Prepare Multiindex for columns
        id_dep = dep_var_list  # depvars
        id_head = [""] * len(models) if model_heads is None else model_heads
        id_num = [f"({s})" for s in range(1, len(models) + 1)]  # model numbers

        # Concatenate the dataframes for coefficients, fixed effects, and model stats
        # and add keys identifying the three parts which will allow make_table
        # to format the table correctly inserting line between the parts
        res_all = pd.concat([res, fe_df, model_stats_df], keys=["coef", "fe", "stats"])

        # When no depvars & headlines should be displayed then use simple index
        # otherwise generate MultiIndex & determine order of index levels as specified by head_order
        if head_order == "":
            res_all.columns = pd.Index(id_num)
        else:
            cindex = [{"h": id_head, "d": id_dep}[c] for c in head_order] + [id_num]
            res_all.columns = pd.MultiIndex.from_arrays(cindex)

        # Generate generic note string if none is provided
        if notes == "":
            if type == "gt":
                notes = (
                    f"Significance levels: * p < {signif_code[2]}, ** p < {signif_code[1]}, *** p < {signif_code[0]}. "
                    + f"Format of coefficient cell:\n{coef_fmt_title}"
                )
            elif type == "tex":
                notes = (
                    f"Significance levels: $*$ p $<$ {signif_code[2]}, $**$ p $<$ {signif_code[1]}, $***$ p $<$ {signif_code[0]}. "
                    + f"Format of coefficient cell: {coef_fmt_title}"
                )
        return make_table(
            res_all,
            type=type,
            notes=notes,
            rgroup_display=False,
            file_name=file_name,
            **kwargs,
        )

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
            to_print += f"Deviance: {np.round(fxst.deviance[0], digits)} "

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


def _tabulate_etable_md(df, n_coef, n_fixef, n_models, n_model_stats):
    """
    Format and tabulate a DataFrame.

    Parameters
    ----------
    - df (pandas.DataFrame): The DataFrame to be formatted and tabulated.
    - n_coef (int): The number of coefficients.
    - n_fixef (int): The number of fixed effects.
    - n_models (int): The number of models.
    - n_model_stats (int): The number of rows with model statistics.

    Returns
    -------
    - formatted_table (str): The formatted table as a string.
    """
    # Format the DataFrame for tabulate
    table = tabulate(
        df,
        headers="keys",
        showindex=False,
        colalign=["left"] + n_models * ["right"],
    )

    # Split the table into header and body
    header, body = table.split("\n", 1)

    # Add separating line after the third row
    body_lines = body.split("\n")
    body_lines.insert(2, "-" * len(body_lines[0]))
    if n_fixef > 0:
        body_lines.insert(-n_model_stats - n_fixef, "-" * len(body_lines[0]))
    body_lines.insert(-n_model_stats, "-" * len(body_lines[0]))
    body_lines.append("-" * len(body_lines[0]))

    # Join the lines back together
    formatted_table = "\n".join([header, "\n".join(body_lines)])

    # Print the formatted table
    return formatted_table


def _parse_coef_fmt(coef_fmt: str, custom_stats: dict):
    """
    Parse the coef_fmt string.

    Parameters
    ----------
    coef_fmt: str
        The coef_fmt string.
    custom_stats: dict
        A dictionary of custom statistics. Key should be lowercased (e.g., simul_intv).
        If you provide "b", "se", "t", or "p" as a key, it will overwrite the default
        values.

    Returns
    -------
    coef_fmt_elements: str
        The parsed coef_fmt string.
    coef_fmt_title: str
        The title for the coef_fmt string.
    """
    custom_elements = list(custom_stats.keys())
    if any([x in ["b", "se", "t", "p"] for x in custom_elements]):
        raise ValueError(
            "You cannot use 'b', 'se', 't', or 'p' as a key in custom_stats."
        )

    title_map = {
        "b": "Coefficient",
        "se": "Std. Error",
        "t": "t-stats",
        "p": "p-value",
    }

    allowed_elements = [
        "b",
        "se",
        "t",
        "p",
        " ",
        "\n",
        r"\(",
        r"\)",
        r"\[",
        r"\]",
        ",",
        *custom_elements,
    ]
    allowed_elements.sort(key=len, reverse=True)

    coef_fmt_elements = re.findall("|".join(allowed_elements), coef_fmt)
    coef_fmt_title = "".join([title_map.get(x, x) for x in coef_fmt_elements])

    return coef_fmt_elements, coef_fmt_title


def _number_formatter(x: float, **kwargs) -> str:
    """
    Format a number.

    Parameters
    ----------
    x: float
        The series to be formatted.
    digits: int
        The number of digits to round to.
    thousands_sep: bool, optional
        The thousands separator. Default is False.
    scientific_notation: bool, optional
        Whether to use scientific notation. Default is True.
    scientific_notation_threshold: int, optional
        The threshold for using scientific notation. Default is 10_000.
    integer: bool, optional
        Whether to format the number as an integer. Default is False.

    Returns
    -------
    formatted_x: pd.Series
        The formatted series.
    """
    digits = kwargs.get("digits", 3)
    thousands_sep = kwargs.get("thousands_sep", False)
    scientific_notation = kwargs.get("scientific_notation", True)
    scientific_notation_threshold = kwargs.get("scientific_notation_threshold", 10_000)
    integer = kwargs.get("integer", False)

    assert digits >= 0, "digits must be a positive integer"

    if integer:
        digits = 0
    x = np.round(x, digits)

    if scientific_notation and x > scientific_notation_threshold:
        return f"%.{digits}E" % x

    x_str = f"{x:,}" if thousands_sep else str(x)

    if "." not in x_str:
        x_str += ".0"  # Add a decimal point if it's an integer
    _int, _float = str(x_str).split(".")
    _float = _float.ljust(digits, "0")
    return _int if digits == 0 else f"{_int}.{_float}"


def _extract(model, key: str, **kwargs):
    """
    Extract the value of a model statistics from a model.

    Parameters
    ----------
    model: Any
        The model from which to extract the value.
    key: str
        The name of the statistic to extract. The method adds _ to the key and calls getattr on the model.

    Returns
    -------
    value: Any
        The extracted and formatted value.
    """
    if key == "se_type":
        if getattr(model, "_vcov_type", "") == "CRV":
            return "by: " + "+".join(getattr(model, "_clustervar", []))
        return getattr(model, "_vcov_type", None)
    attr_name = f"_{key}"
    val = getattr(model, attr_name, None)
    if val is None:
        return "-"
    if isinstance(val, (int, np.integer)):
        return _number_formatter(float(val), integer=True, **kwargs)
    if isinstance(val, (float, np.floating)):
        if math.isnan(val):
            return "-"
        return _number_formatter(float(val), **kwargs)
    if isinstance(val, bool):
        return str(val)
    return str(val)


def _relabel_index(index, labels=None, stats_labels=None):
    if stats_labels is None:
        if isinstance(index, pd.MultiIndex):
            index = pd.MultiIndex.from_tuples(
                [tuple(labels.get(k, k) for k in i) for i in index]
            )
        else:
            index = [labels.get(k, k) for k in index]
    else:
        # if stats_labels is provided, we relabel the lowest level of the index with it
        if isinstance(index, pd.MultiIndex):
            new_index = []
            for i in index:
                new_index.append(
                    tuple(
                        [labels.get(k, k) for k in i[:-1]]
                        + [stats_labels.get(i[-1], i[-1])]
                    )
                )
            index = pd.MultiIndex.from_tuples(new_index)
        else:
            index = [stats_labels.get(k, k) for k in index]
    return index


def _format_mean_std(
    data: pd.Series, digits: int = 2, newline: bool = True, type=str
) -> str:
    """
    Calculate the mean and standard deviation of a pandas Series and return as a string of the format "mean /n (std)".

    Parameters
    ----------
    data : pd.Series
        The pandas Series for which to calculate the mean and standard deviation.
    digits : int, optional
        The number of decimal places to round the mean and standard deviation to. The default is 2.
    newline : bool, optional
        Whether to add a newline character between the mean and standard deviation. The default is True.
    type : str, optional
        The type of the table output.

    Returns
    -------
    _format_mean_std : str
        The mean and standard deviation of the pandas Series formated as a string.

    """
    mean = data.mean()
    std = data.std()
    if newline:
        if type == "gt":
            return f"{mean:.{digits}f}<br>({std:.{digits}f})"
        elif type == "tex":
            return f"{mean:.{digits}f}\\\\({std:.{digits}f})"
    return f"{mean:.{digits}f} ({std:.{digits}f})"


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
        Additional arguments to be passed to the make_table function.

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
    if stats is None:
        stats = ["count", "mean", "std"]
    if labels is None:
        labels = {}
    if stats_labels is None:
        stats_labels = {}
    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame."
    assert all(pd.api.types.is_numeric_dtype(df[var]) for var in vars), (
        "Variables must be numerical."
    )
    assert type in ["gt", "tex", "df"], "type must be either 'gt' or 'tex' or 'df'."
    assert byrow is None or byrow in df.columns, (
        "byrow must be a column in the DataFrame."
    )
    assert bycol is None or all(col in df.columns for col in bycol), (
        "bycol must be a list of columns in the DataFrame."
    )

    # Default stats labels dictionary
    stats_dict = {
        "count": "N",
        "mean": "Mean",
        "std": "Std. Dev.",
        "mean_std": "Mean (Std. Dev.)",
        "mean_newline_std": "Mean (Std. Dev.)",
        "min": "Min",
        "max": "Max",
        "var": "Variance",
        "median": "Median",
    }
    stats_dict.update(stats_labels or {})

    # Define custom aggregation functions
    def mean_std(x):
        return _format_mean_std(x, digits=digits, newline=False, type=type)

    def mean_newline_std(x):
        return _format_mean_std(x, digits=digits, newline=True, type=type)

    # Create a dictionary to map stat names to custom functions
    custom_funcs = {"mean_std": mean_std, "mean_newline_std": mean_newline_std}

    # Prepare the aggregation dictionary allowing custom functions
    agg_funcs = {var: [custom_funcs.get(stat, stat) for stat in stats] for var in vars}

    # Calculate the desired statistics
    if (byrow is not None) and (bycol is not None):
        bylist = [byrow, *bycol]
        res = df.groupby(bylist).agg(agg_funcs)
    if (byrow is None) and (bycol is None):
        res = df.agg(agg_funcs)
    elif (byrow is not None) and (bycol is None):
        res = df.groupby(byrow).agg(agg_funcs)
    elif (byrow is None) and (bycol is not None):
        res = df.groupby(bycol).agg(agg_funcs)

    # Set counts_row_below to false when byrow is not None
    # or when 'count' is not in stats
    if (byrow is not None) or ("count" not in stats):
        counts_row_below = False

    # Round all floats to required decimal places
    # Convert to string to preserve the formatting
    format_string = ",." + str(digits) + "f"

    # Reshaping of table (just transpose when no multiindex)
    if res.columns.nlevels == 1:
        # Check whether number of obs should be displayed at the bottom
        if counts_row_below:
            # Only when all counts are the same within each row
            if res.loc["count"].nunique() == 1:
                # collect the number of obs
                nobs = res.loc["count"].iloc[0]
                # Drop the count row
                res = res.drop("count", axis=0)
                if "count" in stats:
                    stats.remove("count")
            else:
                counts_row_below = False

        # Transpose
        res = res.transpose(copy=True)

        # print(res)
        # Format the statistics
        for col in res.columns:
            # Format the statistics
            # for some reason count stats are displayed as floats when no multiindex,
            # so we need to convert them to integers
            if res[col].name == "count":
                res[col] = res[col].apply(lambda x: f"{x:.0f}")
            elif res[col].dtype == float:
                res[col] = res[col].apply(lambda x: f"{x:{format_string}}")

        # Add the number of observations at the bottom of the table
        if counts_row_below:
            obs_row = [str(int(nobs))] + [""] * (len(res.columns) - 1)
            res.loc[stats_dict["count"]] = obs_row

    else:
        # When there is a multiindex in the columns
        # First check whether number of obs should be displayed at the bottom
        if counts_row_below:
            # collect the number of obs for each row
            count_columns = res.xs("count", axis=1, level=-1)
            # Ensure count_columns is always a DataFrame
            if isinstance(count_columns, pd.Series):
                count_columns = count_columns.to_frame()
            # when all counts are the same within each row,
            # generate a vector with the counts
            if count_columns.nunique(axis=1).eq(1).all():
                nobs = count_columns.iloc[:, 0]
                # Drop the count column
                res = res.drop("count", axis=1, level=-1)
                if "count" in stats:
                    stats.remove("count")
                # And append the counts as an additional column
                # with the value being assigned to the column of the first stat
                # and labeled as defined in the stats_dict
                res[stats_dict["count"], stats[0]] = nobs
            else:
                counts_row_below = False

        # Format the statistics
        for col in res.columns:
            if res[col].dtype == float:
                res[col] = res[col].apply(lambda x: f"{x:{format_string}}")

        # Now some reshaping to bring the multiindex dataframe in the form of a typical descriptive statistics table
        res = pd.DataFrame(res.stack(level=0, future_stack=True))

        # First bring the variables to the rows:
        # Assign name to the column index
        res.columns.names = ["Statistics"]
        if bycol is not None:
            # Then bring the column objects to the columns:
            res = pd.DataFrame(res.unstack(level=tuple(bycol)))
            # Finally we want to have the objects first and then the statistics
            if not isinstance(res.columns, pd.MultiIndex):
                res.columns = pd.MultiIndex.from_tuples(res.columns)  # type: ignore
            res.columns = res.columns.reorder_levels([*bycol, "Statistics"])
            # And sort it properly by the variables
            # (we want to preserve the order of the lowest level for the stats)
            levels_to_sort = list(range(res.columns.nlevels - 1))
            res = res.sort_index(axis=1, level=levels_to_sort, sort_remaining=False)

        # When hide_stats is True, we remove the names of the statistics
        # And add a note to the table listing the statistics when the user
        # has not provided a notes string
        if hide_stats:
            res.columns = res.columns.droplevel(-1)
            if notes == "":
                notes = (
                    "Note: Displayed statistics are "
                    + ", ".join([stats_dict.get(k, k) for k in stats])
                    + "."
                )

    # Replace all NaNs with empty strings
    res = res.fillna("")

    # Relabel Variable names in row and column indices
    res.columns = _relabel_index(res.columns, labels, stats_dict)
    res.index = _relabel_index(res.index, labels)

    # When counts_row_below: Turn row index into a multiindex
    # to set up a second panel for the number of observations
    # that make_table will thus separate by a line
    if counts_row_below:
        res.index = pd.MultiIndex.from_tuples([("stats", i) for i in res.index])
        # Modify the last tuple in the MultiIndex
        new_index = list(res.index)
        new_index[-1] = ("nobs", stats_dict["count"])
        res.index = pd.MultiIndex.from_tuples(new_index)

    # Show row groups iff byrow is not None
    rgroup_display = byrow is not None

    # Generate the table
    if type in ["gt", "tex"]:
        # And make a booktab
        return make_table(
            res, type=type, notes=notes, rgroup_display=rgroup_display, **kwargs
        )
    else:
        return res
