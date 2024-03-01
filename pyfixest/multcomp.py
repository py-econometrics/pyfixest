import warnings
from typing import Union

import pandas as pd

import pyfixest
from pyfixest.estimation import Feols, Fepois
from pyfixest.utils._exceptions import find_stack_level


def bonferroni(models: Union[list[Feols, Fepois], Fepois], param: str) -> pd.DataFrame:
    """
    Compute Bonferroni adjusted p-values for multiple hypothesis testing.

    For each model, it is assumed that tests to adjust are of the form
    "param = 0".

        'pyfixest.multcomp.bonferroni' is deprecated and will be removed in a future
        version. Please use 'pyfixest.bonferroni' instead. You may refer the updated
        documentation at: https://s3alfisc.github.io/pyfixest/quickstart.html

    Parameters
    ----------
    models : list[Feols, Fepois], Feols or Fepois
        A list of models for which the p-values should be adjusted, or a Feols or
        Fepois object.
    param : str
        The parameter for which the p-values should be adjusted.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing estimation statistics, including the Bonferroni
        adjusted p-values.

    Examples
    --------
    ```python
    from pyfixest.estimation import feols
    from pyfixest.utils import get_data
    from pyfixest.multcomp import bonferroni

    data = get_data().dropna()
    fit1 = feols("Y ~ X1", data=data)
    fit2 = feols("Y ~ X1 + X2", data=data)
    bonf_df = bonferroni([fit1, fit2], param="X1")
    bonf_df
    ```
    """
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


def rwolf(
    models: Union[list[Feols], Feols], param: str, B: int, seed: int
) -> pd.DataFrame:
    """
    Compute Romano-Wolf adjusted p-values for multiple hypothesis testing.

    For each model, it is assumed that tests to adjust are of the form
    "param = 0". This function uses the `wildboottest()` method for running the
    bootstrap, hence models of type `Feiv` or `Fepois` are not supported.

        'pyfixest.multcomp.rwolf' is deprecated and will be removed in a future
        version. Please use 'pyfixest.rwolf' instead. You may refer the updated
        documentation at: https://s3alfisc.github.io/pyfixest/quickstart.html

    Parameters
    ----------
    models : list[Feols] or FixestMulti
        A list of models for which the p-values should be computed, or a
        FixestMulti object.
        Models of type `Feiv` or `Fepois` are not supported.
    param : str
        The parameter for which the p-values should be computed.
    B : int
        The number of bootstrap replications.
    seed : int
        The seed for the random number generator.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing estimation statistics, including the Romano-Wolf
        adjusted p-values.

    Examples
    --------
    ```python
    from pyfixest.estimation import feols
    from pyfixest.utils import get_data
    from pyfixest.multcomp import rwolf

    data = get_data().dropna()
    fit = feols("Y ~ Y2 + X1 + X2", data=data)
    rwolf(fit.to_list(), "X1", B=9999, seed=123)

    fit1 = feols("Y ~ X1", data=data)
    fit2 = feols("Y ~ X1 + X2", data=data)
    rwolf_df = rwolf([fit1, fit2], "X1", B=9999, seed=123)
    rwolf_df
    ```
    """
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
