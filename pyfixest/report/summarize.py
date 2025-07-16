import re
import warnings
from collections import Counter
from collections.abc import ValuesView
from typing import Optional, Union

import numpy as np
import pandas as pd
from great_tables import GT
from tabulate import tabulate

from pyfixest.estimation.feiv_ import Feiv
from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.fepois_ import Fepois
from pyfixest.estimation.FixestMulti_ import FixestMulti
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
    custom_stats: Optional[dict] = None,
    custom_model_stats: Optional[dict] = None,
    keep: Optional[Union[list, str]] = None,
    drop: Optional[Union[list, str]] = None,
    exact_match: Optional[bool] = False,
    labels: Optional[dict] = None,
    cat_template: Optional[str] = None,
    show_fe: Optional[bool] = True,
    show_se_type: Optional[bool] = True,
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

    dep_var_list = []
    nobs_list = []
    fixef_list: list[str] = []
    n_coefs = []
    se_type_list = []
    r2_list = []
    adj_r2_list = []
    r2_within_list = []

    # Define code for R2, interaction & line break depending on output type
    if type in ["gt", "html"]:
        interactionSymbol = " &#215; "
        R2code = "R<sup>2</sup>"
        adj_R2_code = "Adj. R<sup>2</sup>"
        R2_within_code = "R<sup>2</sup> Within"
        lbcode = "<br>"
    elif type == "tex":
        interactionSymbol = " $\\times$ "
        R2code = "$R^2$"
        adj_R2_code = "Adj. $R^2$"
        R2_within_code = "$R^2$ Within"
        lbcode = r"\\"
    else:
        interactionSymbol = " x "
        R2code = "R2"
        adj_R2_code = "Adj. R2"
        R2_within_code = "R2 Within"
        lbcode = "\n"

    for model in models:
        dep_var_list.append(model._depvar)
        n_coefs.append(len(model._coefnames))

        _nobs_kwargs = kwargs.copy()
        _nobs_kwargs["integer"] = True
        _nobs_kwargs["scientific_notation"] = False
        nobs_list.append(_number_formatter(model._N, **_nobs_kwargs))

        if not np.isnan(model._r2):
            r2_list.append(_number_formatter(model._r2, **kwargs))
        else:
            r2_list.append("-")

        if not np.isnan(model._adj_r2):
            adj_r2_list.append(_number_formatter(model._adj_r2, **kwargs))
        else:
            adj_r2_list.append("-")

        if not np.isnan(model._r2_within):
            r2_within_list.append(_number_formatter(model._r2_within, **kwargs))
        else:
            r2_within_list.append("-")

        if model._vcov_type == "CRV":
            se_type_list.append("by: " + "+".join(model._clustervar))
        else:
            se_type_list.append(model._vcov_type)

        if model._fixef is not None and model._fixef != "0":
            fixef_list += model._fixef.split("+")

    # find all fixef variables when the user does not want to hide the FE rows
    if show_fe:
        # drop "" from fixef_list
        fixef_list = [x for x in fixef_list if x]
        # keep only unique values
        fixef_list = list(set(fixef_list))
        n_fixef = len(fixef_list)
    else:
        fixef_list = []
        n_fixef = 0

    # First create a dataframe for the model stats such as R2, nobs, etc.
    model_stats_df = pd.DataFrame()
    if custom_model_stats is not None:
        for stat, values in custom_model_stats.items():
            model_stats_df[stat] = values
    model_stats_df["Observations"] = nobs_list
    if show_se_type:
        model_stats_df["S.E. type"] = se_type_list
    model_stats_df[R2code] = r2_list
    n_model_stats = model_stats_df.shape[1]
    if any(x != "-" for x in r2_within_list):
        model_stats_df[R2_within_code] = r2_within_list
    else:
        model_stats_df[adj_R2_code] = adj_r2_list
    # Transpose
    model_stats_df = model_stats_df.T

    # Create a dataframe for the Fixed Effects markers
    fe_df = pd.DataFrame()
    # when at least one model has a fixed effect & the user wants to show them
    if fixef_list:
        for fixef in fixef_list:
            # check if not empty string
            if fixef:
                for i, model in enumerate(models):
                    if (
                        model._fixef is not None
                        and fixef in model._fixef.split("+")
                        and not model._use_mundlak
                    ):
                        fe_df.loc[i, fixef] = "x"
                    else:
                        fe_df.loc[i, fixef] = "-"
        # Sort by model
        fe_df.sort_index(inplace=True)
        # Transpose
        fe_df = fe_df.T
    else:
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
        # When the user provides a dictionary for fixed effects, then use it
        # When a corresponsing variable is not in the felabel dictionary, then use the labels dictionary
        # When in neither then just use the original variable name
        fe_index = fe_df.index.to_series()
        fe_index = fe_index.apply(lambda x: felabels.get(x, labels.get(x, x)))
        fe_df.set_index(fe_index, inplace=True)

    model_stats_df.columns = res.columns
    if show_fe:
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


def make_table(
    df: pd.DataFrame,
    type: str = "gt",
    notes: str = "",
    rgroup_sep: str = "tb",
    rgroup_display: bool = True,
    caption: Optional[str] = None,
    tab_label: Optional[str] = None,
    texlocation: str = "htbp",
    full_width: bool = False,
    file_name: Optional[str] = None,
    **kwargs,
):
    r"""
    Create a booktab style table in the desired format (gt or tex) from a DataFrame.
    The DataFrame can have a multiindex. Column index used to generate horizonal
    table spanners. Row index used to generate row group names and
    row names. The table can have multiple index levels in columns and up to
    two levels in rows.


    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the table to be displayed.
    type : str, optional
        Type of table to be created. The default is 'gt'.
    notes : str
        Table notes to be displayed at the bottom of the table.
    rgroup_sep : str
        Whether group names are separated by lines. The default is "tb".
        When output type = 'gt', the options are 'tb', 't', 'b', or '', i.e.
        you can specify whether to have a line above, below, both or none.
        When output type = 'tex' no line will be added between the row groups
        when rgroup_sep is '' and otherwise a line before the group name will be added.
    rgroup_display : bool
        Whether to display row group names. The default is
        True.
    caption : str
        Table caption to be displayed at the top of the table. The default is None.
        When either caption or label is provided the table will be wrapped in a
        table environment.
    tab_label : str
        LaTex label of the table. The default is None. When either caption or label
        is provided the table will be wrapped in a table environment.
    texlocation : str
        Location of the table. The default is 'htbp'.
    full_width : bool
        Whether to expand the table to the full width of the page. The default is False.
    file_name : str
        Name of the file to save the table to. The default is None.
        gt tables will be saved as html files and latex tables as tex files.

    Returns
    -------
    A table in the specified format.
    """
    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame."
    assert not isinstance(df.index, pd.MultiIndex) or df.index.nlevels <= 2, (
        "Row index can have at most two levels."
    )
    assert type in ["gt", "tex"], "type must be either 'gt' or 'tex'."
    assert rgroup_sep in [
        "tb",
        "t",
        "b",
        "",
    ], "rgroup_sep must be either 'tb', 't', 'b', or ''."
    assert file_name is None or (
        isinstance(file_name, str) and file_name.endswith((".html", ".tex"))
    ), "file_name must end with '.html' or '.tex'."

    # Make a copy of the DataFrame to avoid modifying the original
    dfs = df.copy()

    # Produce LaTeX code if either type is 'tex' or the
    # user has passed a file_name which ends with '.tex'
    if type == "tex" or (isinstance(file_name, str) and file_name.endswith(".tex")):
        # First wrap all cells which contain a line break in a makecell command
        dfs = dfs.map(
            lambda x: f"\\makecell{{{x}}}" if isinstance(x, str) and "\\\\" in x else x
        )
        row_levels = dfs.index.nlevels
        # when the row index has more than one level, we will store
        # the top level to use later to add clines and row group titles
        # and then remove it
        if row_levels > 1:
            # Store the top level of the row index
            top_row_id = dfs.index.get_level_values(0).to_list()
            # Generate a list of the distinct values
            row_groups = list(dict.fromkeys(top_row_id))
            # Generate a list containing the number of rows for each group
            row_groups_len = [top_row_id.count(group) for group in row_groups]
            # Drop the top level of the row index:
            dfs.index = dfs.index.droplevel(0)

        # Style the table
        styler = dfs.style
        # if caption is not None:
        #     styler.set_caption(caption)

        # Generate LaTeX code
        latex_res = styler.to_latex(
            hrules=True,
            multicol_align="c",
            multirow_align="t",
            column_format="l" + "c" * (dfs.shape[1] + dfs.index.nlevels),
        )

        # # Now perform post-processing of the LaTeX code
        # # First split the LaTeX code into lines
        lines = latex_res.splitlines()
        # Find the line number of the \midrule
        line_at = next(i for i, line in enumerate(lines) if "\\midrule" in line)
        # Add space after this \midrule:
        lines.insert(line_at + 1, "\\addlinespace")
        line_at += 1

        # When there are row groups then insert midrules and groupname
        if row_levels > 1 and len(row_groups) > 1:
            # Insert a midrule after each row group
            for i in range(len(row_groups)):
                if rgroup_display:
                    # Insert a line with the row group name & same space around it
                    # lines.insert(line_at+1, "\\addlinespace")
                    lines.insert(line_at + 1, "\\emph{" + row_groups[i] + "} \\\\")
                    lines.insert(line_at + 2, "\\addlinespace")
                    lines.insert(line_at + 3 + row_groups_len[i], "\\addlinespace")
                    line_at += 3
                if (rgroup_sep != "") and (i < len(row_groups) - 1):
                    # For tex output we only either at a line between the row groups or not
                    # And we don't add a line after the last row group
                    line_at += row_groups_len[i] + 1
                    lines.insert(line_at, "\\midrule")
                    lines.insert(line_at + 1, "\\addlinespace")
                    line_at += 1
        else:
            # Add line space before the end of the table
            lines.insert(line_at + dfs.shape[0] + 1, "\\addlinespace")

        # Insert cmidrules (equivalent to column spanners in gt)
        # First find the first line with an occurrence of "multicolumn"
        cmidrule_line_number = None
        for i, line in enumerate(lines):
            if "multicolumn" in line:
                cmidrule_line_number = i + 1
                # Regular expression to find \multicolumn{number}
                pattern = r"\\multicolumn\{(\d+)\}"
                # Find all matches (i.e. values of d) in the LaTeX string & convert to integers
                ncols = [int(match) for match in re.findall(pattern, line)]

                cmidrule_string = ""
                leftcol = 2
                for n in ncols:
                    cmidrule_string += (
                        r"\cmidrule(lr){"
                        + str(leftcol)
                        + "-"
                        + str(leftcol + n - 1)
                        + "} "
                    )
                    leftcol += n
                lines.insert(cmidrule_line_number, cmidrule_string)

        # # Put the lines back together
        latex_res = "\n".join(lines)

        # Wrap in threeparttable to allow for table notes
        if notes is not None:
            latex_res = (
                "\\begin{threeparttable}\n"
                + latex_res
                + "\n\\footnotesize "
                + notes
                + "\n\\end{threeparttable}"
            )
        else:
            latex_res = (
                "\\begin{threeparttable}\n" + latex_res + "\n\\end{threeparttable}"
            )

        # If caption or label specified then wrap in table environment
        if (caption is not None) or (tab_label is not None):
            latex_res = (
                "\\begin{table}["
                + texlocation
                + "]\n"
                + "\\centering\n"
                + ("\\caption{" + caption + "}\n" if caption is not None else "")
                + ("\\label{" + tab_label + "}\n" if tab_label is not None else "")
                + latex_res
                + "\n\\end{table}"
            )

        # Set cell aligment to top
        latex_res = "\\renewcommand\\cellalign{t}\n" + latex_res

        # Set table width to full page width if full_width is True
        # This is done by changing the tabular environment to tabular*
        if full_width:
            latex_res = latex_res.replace(
                "\\begin{tabular}{l", "\\begin{tabularx}{\\linewidth}{X"
            )
            latex_res = latex_res.replace(
                "\\end{tabular}", "\\end{tabularx}\n \\vspace{3pt}"
            )
            # with tabular*
            # latex_res = latex_res.replace("\\begin{tabular}{", "\\begin{tabular*}{\linewidth}{@{\extracolsep{\\fill}}")
            # latex_res = latex_res.replace("\\end{tabular}", "\\end{tabular*}")

        if file_name is not None:
            with open(file_name, "w") as f:
                f.write(latex_res)  # Write the latex code to a file

        if type == "tex":
            return latex_res

    if type == "gt":
        # GT does not support MultiIndex columns, so we need to flatten the columns
        if isinstance(dfs.columns, pd.MultiIndex):
            # Store labels of the last level of the column index (to use as column names)
            col_names = dfs.columns.get_level_values(-1)
            nl = dfs.columns.nlevels
            # As GT does not accept non-unique column names: so to allow for them
            # we just assign column numbers to the lowest index level
            col_numbers = list(map(str, range(len(dfs.columns))))
            # Save the whole column index in order to generate table spanner labels later
            dfcols = dfs.columns.to_list()
            # Then flatten the column index just numbering the columns
            dfs.columns = pd.Index(col_numbers)
            # Store the mapping of column numbers to column names
            col_dict = dict(zip(col_numbers, col_names))
            # Modify the last elements in each tuple in dfcols
            dfcols = [(t[:-1] + (col_numbers[i],)) for i, t in enumerate(dfcols)]
            # And drop the first column as we don't want table spanners on top of the variables
            # WE DON'T NEED THIS WITH ROW INDEX dfcols = dfcols[1:]
        else:
            nl = 1

        rowindex = dfs.index

        # Now reset row index to have the index as columns to be displayed in the table
        dfs.reset_index(inplace=True)

        # And specify the rowname_col and groupname_col
        if isinstance(rowindex, pd.MultiIndex):
            rowname_col = dfs.columns[1]
            groupname_col = dfs.columns[0]
        else:
            rowname_col = dfs.columns[0]
            groupname_col = None

        # Generate the table with GT
        gt = GT(dfs, auto_align=False)

        # When caption is provided, add it to the table
        if caption is not None:
            gt = (
                gt.tab_header(title=caption).tab_options(
                    table_border_top_style="hidden",
                )  # Otherwise line above caption
            )

        if nl > 1:
            # Add column spanners based on multiindex
            # Do this for every level in the multiindex (except the one with the column numbers)
            for i in range(nl - 1):
                col_spanners: dict[str, list[str | int]] = {}
                # Iterate over columns and group them by the labels in the respective level
                for c in dfcols:
                    key = c[i]
                    if key not in col_spanners:
                        col_spanners[key] = []
                    col_spanners[key].append(c[-1])
                for label, columns in col_spanners.items():
                    gt = gt.tab_spanner(label=label, columns=columns, level=nl - 1 - i)
            # Restore column names
            gt = gt.cols_label(**col_dict)

        # Customize the table layout
        gt = (
            gt.tab_source_note(notes)
            .tab_stub(rowname_col=rowname_col, groupname_col=groupname_col)
            .tab_options(
                table_border_bottom_style="hidden",
                stub_border_style="hidden",
                column_labels_border_top_style="solid",
                column_labels_border_top_color="black",
                column_labels_border_bottom_style="solid",
                column_labels_border_bottom_color="black",
                column_labels_border_bottom_width="0.5px",
                column_labels_vlines_color="white",
                column_labels_vlines_width="0px",
                table_body_border_top_style="solid",
                table_body_border_top_width="0.5px",
                table_body_border_top_color="black",
                table_body_hlines_style="none",
                table_body_vlines_color="white",
                table_body_vlines_width="0px",
                table_body_border_bottom_color="black",
                row_group_border_top_style="solid",
                row_group_border_top_width="0.5px",
                row_group_border_top_color="black",
                row_group_border_bottom_style="solid",
                row_group_border_bottom_width="0.5px",
                row_group_border_bottom_color="black",
                row_group_border_left_color="white",
                row_group_border_right_color="white",
                data_row_padding="4px",
                column_labels_padding="4px",
            )
            .cols_align(align="center")
        )

        # Full page width
        if full_width:
            gt = gt.tab_options(table_width="100%")

        # Customize row group display
        if "t" not in rgroup_sep:
            gt = gt.tab_options(row_group_border_top_style="none")
        if "b" not in rgroup_sep:
            gt = gt.tab_options(row_group_border_bottom_style="none")
        if not rgroup_display:
            gt = gt.tab_options(
                row_group_font_size="0px",
                row_group_padding="0px",
            )
        # Save the html code of the table to a file
        if file_name is not None:
            with open(file_name, "w") as f:
                f.write(gt.as_raw_html())

        return gt


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
