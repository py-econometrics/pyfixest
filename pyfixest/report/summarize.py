import re
from typing import Optional, Union

import numpy as np
import pandas as pd
from tabulate import tabulate

from pyfixest.estimation.feiv_ import Feiv
from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.fepois_ import Fepois
from pyfixest.utils.dev_utils import _select_order_coefs


def etable(
    models: Union[Feols, Fepois, Feiv, list],
    type: Optional[str] = "md",
    signif_code: Optional[list] = [0.001, 0.01, 0.05],
    coef_fmt: Optional[str] = "b (se)",
    custom_stats: Optional[dict] = dict(),
    keep: Optional[Union[list, str]] = [],
    drop: Optional[Union[list, str]] = [],
    exact_match: Optional[bool] = False,
    **kwargs,
) -> Union[pd.DataFrame, str]:
    r"""
    Create an esttab-like table from a list of models.

    Parameters
    ----------
    models : list
        A list of models of type Feols, Feiv, Fepois.
    type : str, optional
        Type of output. Either "df" for pandas DataFrame, "md" for markdown,
        or "tex" for LaTeX table. Default is "md".
    signif_code : list, optional
        Significance levels for the stars. Default is [0.001, 0.01, 0.05].
        If None, no stars are printed.
    coef_fmt : str, optional
        The format of the coefficient (b), standard error (se), t-stats (t), and
        p-value (p). Default is `"b (se)"`.
        Spaces ` `, parentheses `()`, brackets `[]`, newlines `\n` are supported.
        Newline is not support for LaTeX output.
    custom_stats: dict, optional
        A dictionary of custom statistics. "b", "se", "t", or "p" are reserved.
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
    digits: int
        The number of digits to round to.
    thousands_sep: bool, optional
        The thousands separator. Default is False.
    scientific_notation: bool, optional
        Whether to use scientific notation. Default is True.
    scientific_notation_threshold: int, optional
        The threshold for using scientific notation. Default is 10_000.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the coefficients and standard errors of the models.
    """  # noqa: D301
    assert (
        signif_code is None or len(signif_code) == 3
    ), "signif_code must be a list of length 3 or None"
    if signif_code:
        assert all(
            [0 < i < 1 for i in signif_code]
        ), "All values of signif_code must be between 0 and 1"
    if signif_code:
        assert (
            signif_code[0] < signif_code[1] < signif_code[2]
        ), "signif_code must be in increasing order"
    models = _post_processing_input_checks(models)

    if custom_stats:
        assert isinstance(custom_stats, dict), "custom_stats must be a dict"
        for key in custom_stats:
            assert isinstance(
                custom_stats[key], list
            ), "custom_stats values must be a list"
            assert len(custom_stats[key]) == len(
                models
            ), f"custom_stats {key} must have the same number as models"

    assert type in [
        "df",
        "tex",
        "md",
        "html",
    ], "type must be either 'df', 'md', 'html' or 'tex'"

    dep_var_list = []
    nobs_list = []
    fixef_list = []
    nobs_list = []
    n_coefs = []
    se_type_list = []
    r2_list = []
    r2_within_list = []  # noqa: F841

    for i, model in enumerate(models):
        dep_var_list.append(model._depvar)
        n_coefs.append(len(model._coefnames))
        nobs_list.append(_number_formatter(model._N, integer=True, **kwargs))
        if model._method == "feols" and not model._is_iv and not model._has_weights:
            r2_list.append(_number_formatter(model._r2, **kwargs))
        else:
            r2_list.append("-")

        if model._vcov_type == "CRV":
            se_type_list.append("by: " + "+".join(model._clustervar))
        else:
            se_type_list.append(model._vcov_type)

        if model._fixef is not None:
            fixef_list += model._fixef.split("+")

    # find all fixef variables
    fixef_list = list(set(fixef_list))
    n_fixef = len(fixef_list)

    # create a pd.dataframe with the depvar, nobs, and fixef as keys
    nobs_fixef_df = pd.DataFrame(
        {"Observations": nobs_list, "S.E. type": se_type_list, "R2": r2_list}
    )

    if fixef_list:  # only when at least one model has a fixed effect
        for fixef in fixef_list:
            nobs_fixef_df[fixef] = "-"

            for i, model in enumerate(models):
                if model._fixef is not None and fixef in model._fixef.split("+"):
                    nobs_fixef_df.loc[i, fixef] = "x"

    colnames = nobs_fixef_df.columns.tolist()
    colnames.reverse()
    nobs_fixef_df = nobs_fixef_df[colnames].T.reset_index()

    coef_fmt_elements, coef_fmt_title = _parse_coef_fmt(coef_fmt, custom_stats)

    etable_list = []
    for i, model in enumerate(models):
        model = (
            model.tidy().reset_index()
        )  # If rounding here and p = 0.0499, it will be rounded to 0.05 and miss threshold.
        model["stars"] = (
            np.where(
                model["Pr(>|t|)"] < signif_code[0],
                "***",
                np.where(
                    model["Pr(>|t|)"] < signif_code[1],
                    "**",
                    np.where(model["Pr(>|t|)"] < signif_code[2], "*", ""),
                ),
            )
            if signif_code
            else ""
        )
        model[coef_fmt_title] = ""
        for element in coef_fmt_elements:
            if element == "b":
                model[coef_fmt_title] += (
                    model["Estimate"].apply(_number_formatter, **kwargs)
                    + model["stars"]
                )
            elif element == "se":
                model[coef_fmt_title] += model["Std. Error"].apply(
                    _number_formatter, **kwargs
                )
            elif element == "t":
                model[coef_fmt_title] += model["t value"].apply(
                    _number_formatter, **kwargs
                )
            elif element == "p":
                model[coef_fmt_title] += model["Pr(>|t|)"].apply(
                    _number_formatter, **kwargs
                )
            elif element in custom_stats:
                assert len(custom_stats[element][i]) == len(
                    model["Estimate"]
                ), f"custom_stats {element} has unequal length to the number of coefficients in model {i}"
                model[coef_fmt_title] += pd.Series(custom_stats[element][i]).apply(
                    _number_formatter, **kwargs
                )
            elif element == "\n" and type == "tex":
                raise ValueError("Newline is currently not supported for LaTeX output.")
            else:
                model[coef_fmt_title] += element
        model[coef_fmt_title] = pd.Categorical(model[coef_fmt_title])
        model = model[["Coefficient", coef_fmt_title]]
        model = pd.melt(
            model, id_vars=["Coefficient"], var_name="Metric", value_name=f"est{i+1}"
        )
        model = model.drop("Metric", axis=1).set_index("Coefficient")
        etable_list.append(model)

    res = pd.concat(etable_list, axis=1)
    if keep or drop:
        idxs = _select_order_coefs(res.index, keep, drop, exact_match)
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
    nobs_fixef_df.columns = res.columns

    depvars = pd.DataFrame({"depvar": dep_var_list}).T.reset_index()
    depvars.columns = res.columns

    res_all = pd.concat([depvars, res, nobs_fixef_df], ignore_index=True)
    res_all.columns = [""] + list(res_all.columns[1:])

    if type == "tex":
        return res_all.to_latex()
    elif type == "md":
        res_all = _tabulate_etable(res_all, len(models), n_fixef)
        print(res_all)
        if signif_code:
            print(
                f"Significance levels: * p < {signif_code[2]}, ** p < {signif_code[1]}, *** p < {signif_code[0]}"
            )
        print(f"Format of coefficient cell:\n{coef_fmt_title}")
    else:
        return res_all


def summary(
    models: Union[Feols, Fepois, Feiv, list], digits: Optional[int] = 3
) -> None:
    """
    Print a summary of estimation results for each estimated model.

    For each model, this method prints a header indicating the fixed-effects and the
    dependent variable, followed by a table of coefficient estimates with standard
    errors, t-values, and p-values.

    Parameters
    ----------
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
        else:
            raise ValueError("Unknown estimation method.")

        print("###")
        print("")
        print("Estimation: ", estimation_method)
        depvar_fixef = f"Dep. var.: {depvar}"
        if fxst._fixef is not None:
            depvar_fixef += f", Fixed effects: {fxst._fixef}"
        print(depvar_fixef)
        print("Inference: ", fxst._vcov_type_detail)
        print("Observations: ", fxst._N)
        print("")
        print(df.to_markdown(floatfmt=f".{digits}f"))
        print("---")
        if fxst._method == "feols":
            if not fxst._is_iv:
                if fxst._has_fixef:
                    print(
                        f"RMSE: {np.round(fxst._rmse, digits)}   R2: {np.round(fxst._r2, digits)}   R2 Within: {np.round(fxst._r2_within, digits)}"
                    )
                else:
                    print(
                        f"RMSE: {np.round(fxst._rmse, digits)}   R2: {np.round(fxst._r2, digits)}"
                    )
        elif fxst._method == "fepois":
            print(f"Deviance: {np.round(fxst.deviance[0], digits)}")
        else:
            pass


def _post_processing_input_checks(models):
    """
    Perform input checks for post-processing models.

    Parameters
    ----------
        models (Feols, Fepois, list, dict): The models to be checked.

    Returns
    -------
        models (Feols, Fepois, list, dict): The checked models.

    Raises
    ------
        TypeError: If the models argument is not of the expected type.

    """
    # check if models instance of Feols or Fepois
    if isinstance(models, (Feols, Fepois)):
        models = [models]

    else:
        if isinstance(models, list):
            for model in models:
                if not isinstance(model, (Feols, Fepois)):
                    raise TypeError(
                        f"""
                        Each element of the passed list needs to be of type Feols
                        or Fepois, but {type(model)} was passed. If you want to
                        summarize a FixestMulti object, please use FixestMulti.to_list()
                        to convert it to a list of Feols or Fepois instances.
                        """
                    )

        else:
            raise TypeError(
                """
                The models argument must be either a list of Feols or Fepois instances, or
                simply a single Feols or Fepois instance. The models argument does not accept instances
                of type FixestMulti - please use models.to_list() to convert the FixestMulti
                instance to a list of Feols or Fepois instances.
                """
            )

    return models


def _tabulate_etable(df, n_models, n_fixef):
    """
    Format and tabulate a DataFrame.

    Parameters
    ----------
    - df (pandas.DataFrame): The DataFrame to be formatted and tabulated.
    - n_models (int): The number of models.
    - n_fixef (int): The number of fixed effects.

    Returns
    -------
    - formatted_table (str): The formatted table as a string.
    """
    # Format the DataFrame for tabulate
    table = tabulate(
        df, headers="keys", showindex=False, colalign=["left"] + n_models * ["right"]
    )

    # Split the table into header and body
    header, body = table.split("\n", 1)

    # Add separating line after the third row
    body_lines = body.split("\n")
    body_lines.insert(2, "-" * len(body_lines[0]))
    body_lines.insert(-3 - n_fixef, "-" * len(body_lines[0]))
    body_lines.insert(-3, "-" * len(body_lines[0]))
    body_lines.append("-" * len(body_lines[0]))

    # Join the lines back together
    formatted_table = "\n".join([header, "\n".join(body_lines)])

    # Print the formatted table
    return formatted_table


def _parse_coef_fmt(coef_fmt: str, custom_stats: Optional[dict] = None):
    """
    Parse the coef_fmt string.

    Parameters
    ----------
    coef_fmt: str
        The coef_fmt string.
    custom_stats: dict, optional
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
    ] + custom_elements
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

    x = f"{x:,}" if thousands_sep else str(x)

    if "." not in x:
        x += ".0"  # Add a decimal point if it's an integer
    _int, _float = str(x).split(".")
    _float = _float.ljust(digits, "0")
    return _int if digits == 0 else f"{_int}.{_float}"
