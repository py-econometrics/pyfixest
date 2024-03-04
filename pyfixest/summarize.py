import warnings
from typing import Optional, Union

import pandas as pd

import pyfixest
from pyfixest.estimation import Feiv, Feols, Fepois
from pyfixest.utils._exceptions import find_stack_level
from pyfixest.utils.dev_utils import docstring_from


@docstring_from(pyfixest.report.etable)
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


@docstring_from(pyfixest.report.summary)
def summary(
    models: Union[Feols, Fepois, Feiv, list], digits: Optional[int] = 3
) -> None:
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
