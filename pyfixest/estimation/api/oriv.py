"""ORIV (Obviously Related Instrumental Variables) estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyfixest.estimation.api.feols import feols
from pyfixest.estimation.internals.literals import VcovTypeOptions

if TYPE_CHECKING:
    from pyfixest.estimation.FixestMulti_ import FixestMulti
    from pyfixest.estimation.models.feols_ import Feols


def oriv(
    fml: str,
    proxies: list[str],
    data: pd.DataFrame,
    var_name: str = "mainVar",
    vcov: VcovTypeOptions | dict[str, str] | None = None,
) -> Feols | FixestMulti:
    """
    Estimate a regression correcting for measurement error using ORIV.

    Implements the Obviously Related Instrumental Variables method from
    Gillen, Snowberg, and Yariv (2019). When a latent variable is measured
    with error by two or more proxies, ORIV stacks the data and uses each
    proxy as an instrument for the others via 2SLS. This yields consistent
    estimates even when all proxies are noisy.

    Parameters
    ----------
    fml : str
        A pyfixest regression formula for the outcome equation, excluding the
        variable measured with error. For example, ``"sales ~ training"``
        specifies that ``sales`` is the dependent variable and ``training`` is
        an exogenous regressor.
    proxies : list[str]
        Column names of the proxy variables in ``data``. Must contain at least
        two proxies for the same latent variable.
    data : pd.DataFrame
        The input DataFrame. It is not modified in place.
    var_name : str, optional
        Name to use for the instrumented variable in the output. Defaults to
        ``"mainVar"``.
    vcov : dict or str or None, optional
        Variance-covariance estimator passed to ``feols``. If None (default),
        uses cluster-robust standard errors (CRV1) clustered on the original
        observation ID, which accounts for the within-observation correlation
        introduced by stacking.

    Returns
    -------
    Feols
        A fitted ``Feols`` object from the IV regression on the stacked data.

    Raises
    ------
    ValueError
        If fewer than two proxies are provided, or if any proxy name is not
        found in ``data``.

    Notes
    -----
    The method works by creating K copies of the dataset (where K is the
    number of proxies). In each copy k, proxy k serves as the endogenous
    variable and the remaining K-1 proxies serve as instruments. The stacked
    dataset is then estimated via 2SLS with cluster-robust standard errors
    at the original observation level.

    Copy-specific intercepts (dummies) are included to absorb any level
    differences across proxies.

    References
    ----------
    Gillen, B., Snowberg, E., & Yariv, L. (2019). Experimenting with
    Measurement Error: Techniques with Applications to the Caltech Cohort
    Study. Journal of Political Economy, 127(4), 1826-1863.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import pyfixest as pf
    >>> np.random.seed(42)
    >>> n = 1000
    >>> ability = np.random.normal(100, 15, n)
    >>> training = (ability + np.random.normal(0, 10, n) >= 100).astype(int)
    >>> sales = 40000 + training * 5000 + ability * 100 + np.random.normal(0, 4000, n)
    >>> test1 = ability + np.random.normal(0, 8, n)
    >>> test2 = ability + np.random.normal(0, 8, n)
    >>> df = pd.DataFrame(
    ...     {"sales": sales, "training": training, "test1": test1, "test2": test2}
    ... )
    >>> result = pf.oriv("sales ~ training", ["test1", "test2"], df, "ability")
    >>> result.coef()["training"]  # close to 5000
    """
    return _oriv(fml=fml, proxies=proxies, data=data, var_name=var_name, vcov=vcov)


def _oriv(
    fml: str,
    proxies: list[str],
    data: pd.DataFrame,
    var_name: str = "mainVar",
    vcov: VcovTypeOptions | dict[str, str] | None = None,
) -> Feols | FixestMulti:
    """Construct the stacked IV dataset and run 2SLS via feols."""
    # --- Input validation ---
    if len(proxies) < 2:
        raise ValueError(f"ORIV requires at least 2 proxies, got {len(proxies)}.")

    missing = [p for p in proxies if p not in data.columns]
    if missing:
        raise ValueError(f"Proxy columns not found in data: {missing}")

    # Avoid name collisions with internal columns
    _id_col = "_oriv_obs_id_"
    _copy_col = "_oriv_copy_"

    k = len(proxies)
    df_list = []

    for i in range(k):
        df_copy = data.copy()
        df_copy[_id_col] = np.arange(len(data))
        df_copy[_copy_col] = i

        # The endogenous variable for this copy is proxy i
        df_copy[var_name] = df_copy[proxies[i]]

        # Instruments: all other proxies
        others = [proxies[j] for j in range(k) if j != i]
        for j, other in enumerate(others):
            df_copy[f"_oriv_iv_{j}"] = df_copy[other]

        df_list.append(df_copy)

    df_stacked = pd.concat(df_list, ignore_index=True)

    # Create copy dummies (drop first to avoid collinearity with intercept)
    for i in range(1, k):
        df_stacked[f"_oriv_d_{i}"] = (df_stacked[_copy_col] == i).astype(float)

    # Build the formula
    # Exogenous part from user formula + copy dummies
    copy_dummies = " + ".join([f"_oriv_d_{i}" for i in range(1, k)])
    instruments = " + ".join([f"_oriv_iv_{j}" for j in range(k - 1)])

    # The user formula is like "y ~ x1 + x2"
    # We append copy dummies and the IV specification
    iv_fml = f"{fml} + {copy_dummies} | {var_name} ~ {instruments}"

    # Default vcov: cluster on original observation ID
    if vcov is None:
        vcov = {"CRV1": _id_col}

    result = feols(fml=iv_fml, data=df_stacked, vcov=vcov)

    return result
