from __future__ import annotations

from typing import TYPE_CHECKING

import maketables
import numpy as np
import pandas as pd

from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.internals.literals import InferenceType
from pyfixest.estimation.models.feiv_ import Feiv
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.report.utils import _post_processing_input_checks

if TYPE_CHECKING:
    from docx.document import Document
    from great_tables import GT

ModelInputType = FixestMulti | Feols | Fepois | Feiv | list[Feols | Fepois | Feiv]

_METHOD_DISPLAY_NAMES: dict[str, str] = {
    "fepois": "Poisson",
    "feglm-logit": "Logit",
    "feglm-probit": "Probit",
    "feglm-gaussian": "Gaussian",
    "twfe": "TWFE",
    "did2s": "DID2S",
}


def _get_estimation_method_name(fxst: Feols) -> str:
    """Get the display name for an estimation method."""
    method = fxst.model.method
    if method == "feols":
        return "IV" if fxst.model.is_iv else "OLS"
    if "quantreg" in method:
        return f"quantreg: q = {fxst.options.quantile}"  # type: ignore
    if method in _METHOD_DISPLAY_NAMES:
        return _METHOD_DISPLAY_NAMES[method]
    raise ValueError(f"Unknown estimation method: {method}")


def etable(
    models: ModelInputType,
    type: str = "gt",
    signif_code: list | None = None,
    coef_fmt: str | None = None,
    custom_stats: dict | None = None,
    custom_model_stats: dict | None = None,
    keep: list | str | None = None,
    drop: list | str | None = None,
    exact_match: bool | None = False,
    labels: dict | None = None,
    cat_template: str | None = None,
    show_fe: bool | None = True,
    felabels: dict | None = None,
    fe_present: str = "x",
    fe_absent: str = "-",
    notes: str = "",
    model_heads: list | None = None,
    head_order: str | None = "dh",
    file_name: str | None = None,
    docx_style: dict[str, object] | None = None,
    **kwargs,
) -> pd.DataFrame | GT | Document | str | None:
    r"""
    Generate a table summarizing the results of multiple regression models.

    This function uses the maketables package internally to create publication-ready
    regression tables. It supports various output formats including HTML (via Great Tables),
    markdown, LaTeX, Typst, and Word.

    Parameters
    ----------
    models : Feols, Fepois, Feiv, Feglm, Quantreg, FixestMulti, or list
        A fitted model, a list of fitted models, or the `FixestMulti`
        result of multiple estimation. GLM and quantile results use the
        same interface through their fitted-model classes.
        The models to be summarized in the table.
    type : str, optional
        Type of output. Either `df` for pandas `DataFrame`, `md` for markdown,
        `html` for an HTML string, `gt` for `great_tables`, `tex` for LaTeX,
        `typst` for Typst, or `docx` for an editable Word document.
        Default is `gt`.
    signif_code : list, optional
        Significance levels for the stars. Default is `None`, which sets
        `[0.001, 0.01, 0.05]`. Must be a list of three increasing values between
        0 and 1; an empty list is not accepted. Note that stars are only
        rendered if `coef_fmt` includes `*` on a statistic token, for example
        `b*` or `b:.3f*`.
    coef_fmt : str, optional
        The format of the coefficient (b), standard error (se), t-stats (t), and
        p-value (p). Default is inherited from maketables.
        To render significance stars, include `*` on the relevant token, following
        maketables syntax, for example `"b:.3f* \n (se:.3f)"` with 3 decimal places.
        Spaces ` `, parentheses `()`, brackets `[]`, newlines `\n` are supported.
        For further available formatting rules, see `maketables.ETable`.
    custom_stats: dict, optional
        A dictionary of custom statistics that can be used in the `coef_fmt` string to be displayed
        in the coefficient cells analogously to `b`, `se` etc. The keys are the names of the custom
        statistics, and the values are lists of lists, where each inner list contains the custom
        statistic values for all coefficients each model.
        Note that `b`, `se`, `t`, or `p` are reserved and cannot be used as keys.
    custom_model_stats: dict, optional
        A dictionary of custom model statistics or model information displayed in a new line in the
        bottom panel of the table. The keys are the names of the statistics (i.e. entry in the first column)
        and the values are a lists of the same length as the number of models. Default is `None`.
    keep : str or list of str, optional
        Coefficient names or regular-expression patterns to retain, in the
        requested pattern order. For example, "X1" matches names containing
        X1 and "^X" matches names starting with X. `None` retains all coefficients.
        Use `exact_match=True` to match complete coefficient names.
    drop: str or list of str, optional
        The pattern for excluding coefficient names. You can pass a string (one
        pattern) or a list (multiple patterns). Syntax is the same as for `keep`.
        Default is keeping all coefficients. Parameter `keep` and `drop` can be
        used simultaneously.
    exact_match: bool, optional
        Whether to use exact match for `keep` and `drop`. Default is `False`.
        If `True`, the pattern will be matched exactly to the coefficient name
        instead of using regular expressions.
    labels: dict, optional
        A dictionary to relabel the variables. The keys in this dictionary are the
        original coefficient names, available from `fit.coef().index`;
        values are the names to display.
        Note that interaction terms will also be relabeled using the labels of the individual variables.
        The command is applied after the `keep` and `drop` commands.
    cat_template: str, optional
        Template to relabel categorical variables. `None` by default, which uses
        the maketables default "{variable}={value}" (e.g. `job[T.Admin]` is
        shown as `job=Admin`). Other options include combinations of
        `{variable}` and `{value}`, e.g. `{variable}::{value}` to mimic fixest
        encoding. But `{variable}--{value}` or `{variable}{value}` or just
        `{value}` are also possible.
    show_fe: bool, optional
        Whether to show the rows with fixed effects markers. Default is `True`.
    felabels: dict, optional
        A dictionary to relabel the fixed effects. Only needed if you want to relabel
        the FE lines with a different label than the one specified for the respective
        variable in the labels dictionary.
        The command is applied after the `keep` and `drop` commands.
    fe_present: str, optional
        Symbol to use when a fixed effect is present in a model. Default is `x`.
        Common alternatives include `Y`, `YES`, `✓`, `✅`, or any custom string.
    fe_absent: str, optional
        Symbol to use when a fixed effect is absent from a model. Default is `-`.
        Common alternatives include `N`, `NO`, `✗`, `""`, or any custom string.
    notes: str, optional
        Custom table notes. Default shows the significance levels and the format of
        the coefficient cell.
    model_heads: list, optional
        Add custom headlines to models. Length of list must correspond to
        number of models. Default is `None`.
    head_order: str, optional
        String to determine the display of the table header. Allowed values
        are `dh`, `hd`, `d`, `h`, or `""`. When `head_order` is `dh`, the dependent
        variable is displayed first, followed by the custom `model_heads`
        (provided the user has specified them). With `hd` it is the other way
        around. When `head_order` is `d`, only the dependent variable and model
        numbers are displayed, with `h` only the custom `model_heads` and model
        numbers, and with `""` only the model numbers. Default is `dh`.
    file_name: str, optional
        The name/path of the file to save when type is `gt`, `tex`, `typst`,
        or `docx`. Default is `None`, which returns the output without saving it.
    docx_style: dict, optional
        Word style overrides, used only when type is `docx`. Passed to
        `maketables.MTable.make` as `docx_style`. For example,
        `{"font_name": "Arial", "font_size_pt": 12, "notes_font_size_pt": 9}`.
        See `maketables.MTable.DEFAULT_DOCX_STYLE` for supported options.
        Default is `None`, which uses maketables' Word style defaults.
    **kwargs
        Further keyword arguments forwarded to `maketables.ETable`. Commonly
        used ones are `caption` (table caption), `tab_label` (label or anchor
        for LaTeX and HTML output), `model_stats` (list of statistics shown in
        the bottom panel, in order, e.g. `["N", "r2", "se_type"]`; include
        `"se_type"` to show the type of standard errors), `model_stats_labels`
        (dict of display labels for `model_stats` keys, e.g.
        `{"N": "Observations"}`), and `order` (explicit coefficient order,
        applied after `keep` and `drop`). `digits` (number of decimal places)
        is only applied when `coef_fmt` has no format specifiers such as
        `b:.3f`; the default `coef_fmt` has them, so `digits` is ignored with a
        warning unless you also pass a `coef_fmt` without specifiers, e.g.
        `"b \n (se)"`. Keyword arguments that maketables does not know are
        silently ignored.

    Returns
    -------
    pandas.DataFrame, great_tables.GT, docx.document.Document, str, or None
        A `DataFrame` for `df`, a GT object for `gt`, a string for `html`,
        `tex`, or `typst`, or a Word Document for `docx`. Markdown output
        is printed and returns `None`. Word output is returned even when saved
        via `file_name`, so it can be edited further with python-docx.

    Examples
    --------
    For more examples, see the [regression tables tutorial](/tutorials/regression-tables.qmd).

    ```{python}
    import pyfixest as pf

    # load data
    df = pf.get_data()
    fit1 = pf.feols("Y~X1 + X2 | f1", df)
    fit2 = pf.feols("Y~X1 + X2 | f1 + f2", df)

    pf.etable([fit1, fit2])
    ```

    ```{python}
    document = pf.etable(
        [fit1, fit2],
        type="docx",
        docx_style={"font_name": "Arial", "font_size_pt": 12},
    )
    # Save directly with file_name="results.docx", or edit the document first.
    document.add_paragraph("Additional discussion of the estimates.")
    ```

    See Also
    --------
    [summary()](/reference/report.summary.qmd) : Print model summaries.
    [coefplot()](/reference/report.coefplot.qmd) : Compare estimates graphically.
    [Regression tables tutorial](/tutorials/regression-tables.qmd) : Formatting and export examples.
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
        "md",
        "html",
        "gt",
        "tex",
        "typst",
        "docx",
    ], "type must be either 'df', 'md', 'html', 'gt', 'tex', 'typst' or 'docx'"

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
            assert isinstance(values, list), "custom_model_stats values must be lists"
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
    elif type == "docx":
        document = table.make(type="docx", docx_style=docx_style)
        if file_name is not None:
            document.save(file_name)
        return document
    elif type == "typst":
        result = table.make(type="typst")
        if file_name is not None:
            with open(file_name, "w") as f:
                f.write(result)
        return result

    return None


def summary(
    models: ModelInputType,
    digits: int = 3,
    inference_type: InferenceType = "regular",
) -> None:
    r"""
    Print a summary of estimation results for each estimated model.

    For each model, this method prints a header indicating the fixed-effects and the
    dependent variable, followed by a table of coefficient estimates with standard
    errors, t-values, and p-values.

    Parameters
    ----------
    models : Feols, Fepois, Feiv, Feglm, Quantreg, FixestMulti, or list
        A fitted model, a list of fitted models, or the `FixestMulti`
        result of multiple estimation. GLM and quantile results use the
        same interface through their fitted-model classes.
    digits : int, optional
        The number of decimal places to round the summary statistics to. Default is 3.
    inference_type : {"regular"}, optional
        Type of coefficient-wise inference to report, handled the same way as in
        `tidy()`. Only `"regular"` is currently available. Defaults to `"regular"`.

    Returns
    -------
    None
        Prints each model summary; use `fit.tidy()` to obtain a `DataFrame`.

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

    See Also
    --------
    [etable()](/reference/report.etable.qmd) : Return or export a formatted regression table.
    [fit.tidy()](/reference/estimation.models.feols_.Feols.tidy.qmd) : Return the coefficient table as a `DataFrame`.
    [Fitted results](/reference-guide/fitted-results.qmd) : Find the next operation on a fit.
    """
    models = _post_processing_input_checks(models)

    for fxst in list(models):
        depvar = fxst.model.depvar

        df = fxst.tidy(inference_type=inference_type).round(digits)

        estimation_method = _get_estimation_method_name(fxst)
        print("###")
        print("")
        print("Estimation: ", estimation_method)
        depvar_fixef = f"Dep. var.: {depvar}"
        if fxst.model.has_fixef:
            depvar_fixef += f", Fixed effects: {fxst.model.fixef}"
        print(depvar_fixef)
        sample_split = fxst.model.sample_split
        if sample_split is None:
            # An unsplit fit reports the full sample, with no split variable.
            print("sample: None = all")
        else:
            sample = "all" if sample_split.value is None else sample_split.value
            print(f"sample: {sample_split.var} = {sample}")
        print("Inference: ", fxst.variance_covariance.spec.vcov_type_detail)
        print("Observations: ", fxst.sample_info.n_obs)
        print("")
        print(df.to_markdown(floatfmt=f".{digits}f"))
        print("---")

        to_print = ""

        fitstat = fxst.fitstat
        if not np.isnan(fitstat.rmse):
            to_print += f"RMSE: {np.round(fitstat.rmse, digits)} "
        if not np.isnan(fitstat.r2):
            to_print += f"R2: {np.round(fitstat.r2, digits)} "
        if not np.isnan(fitstat.r2_within):
            to_print += f"R2 Within: {np.round(fitstat.r2_within, digits)} "
        if not np.isnan(fitstat.deviance):
            to_print += f"Deviance: {np.round(fitstat.deviance, digits)} "

        print(to_print)
