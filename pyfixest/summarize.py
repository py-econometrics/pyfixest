import warnings
from typing import Optional, Union

import pandas as pd

import pyfixest
from pyfixest.estimation import Feiv, Feols, Fepois
from pyfixest.utils._exceptions import find_stack_level


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

        'pyfixest.summarize.etable' is deprecated and will be removed in a future
        version. Please use 'pyfixest.etable' instead. You may refer the updated
        documentation at: https://s3alfisc.github.io/pyfixest/quickstart.html

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
    warnings.warn(
        "'pyfixest.summarize.etable' is deprecated and "
        "will be removed in a future version.\n"
        "Please use 'pyfixest.etable' instead. "
        "You may refer the updated documentation at: "
        "https://s3alfisc.github.io/pyfixest/quickstart.html",
        FutureWarning,
        stacklevel=find_stack_level(),
    )
    return pyfixest.report.etable(
        models=models,
        type=type,
        signif_code=signif_code,
        coef_fmt=coef_fmt,
        custom_stats=custom_stats,
        keep=keep,
        drop=drop,
        exact_match=exact_match,
        **kwargs,
    )


def summary(
    models: Union[Feols, Fepois, Feiv, list], digits: Optional[int] = 3
) -> None:
    """
    Print a summary of estimation results for each estimated model.

    For each model, this method prints a header indicating the fixed-effects and the
    dependent variable, followed by a table of coefficient estimates with standard
    errors, t-values, and p-values.

        'pyfixest.summarize.summary' is deprecated and will be removed in a future
        version. Please use 'pyfixest.summary' instead. You may refer the updated
        documentation at: https://s3alfisc.github.io/pyfixest/quickstart.html

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
    from pyfixest.utils import get_data
    from pyfixest.estimation import feols
    from pyfixest.summarize import summary

    # load data
    df = get_data()
    fit1 = feols("Y~X1 + X2 | f1", df)
    fit2 = feols("Y~X1 + X2 | f1 + f2", df)
    fit3 = feols("Y~X1 + X2 | f1 + f2 + f3", df)

    summary([fit1, fit2, fit3])
    ```
    """
    warnings.warn(
        "'pyfixest.summarize.summary' is deprecated and "
        "will be removed in a future version.\n"
        "Please use 'pyfixest.summary' instead. "
        "You may refer the updated documentation at: "
        "https://s3alfisc.github.io/pyfixest/quickstart.html",
        FutureWarning,
        stacklevel=find_stack_level(),
    )
    pyfixest.report.summary(models=models, digits=digits)
