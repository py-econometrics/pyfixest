import warnings
from typing import Union

import pandas as pd

import pyfixest
from pyfixest.estimation import Feols, Fepois
from pyfixest.utils._exceptions import find_stack_level
from pyfixest.utils.dev_utils import docstring_from


@docstring_from(pyfixest.bonferroni)
def bonferroni(models: Union[list[Feols, Fepois], Fepois], param: str) -> pd.DataFrame:
    warnings.warn(
        "'pyfixest.multcomp.bonferroni' is deprecated and "
        "will be removed in a future version.\n"
        "Please use 'pyfixest.bonferroni' instead. "
        "You may refer the updated documentation at: "
        "https://s3alfisc.github.io/pyfixest/quickstart.html",
        FutureWarning,
        stacklevel=find_stack_level(),
    )
    return pyfixest.bonferroni(
        models=models,
        param=param,
    )


@docstring_from(pyfixest.rwolf)
def rwolf(
    models: Union[list[Feols], Feols], param: str, B: int, seed: int
) -> pd.DataFrame:
    warnings.warn(
        "'pyfixest.multcomp.rwolf' is deprecated and "
        "will be removed in a future version.\n"
        "Please use 'pyfixest.rwolf' instead. "
        "You may refer the updated documentation at: "
        "https://s3alfisc.github.io/pyfixest/quickstart.html",
        FutureWarning,
        stacklevel=find_stack_level(),
    )
    return pyfixest.rwolf(
        models=models,
        param=param,
        B=B,
        seed=seed,
    )
