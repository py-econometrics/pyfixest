from importlib import import_module
from typing import Optional

import numpy as np
import pandas as pd


def _get_ritest_coefs(
    data: pd.DataFrame,
    resampvar: str,
    fml: str,
    reps: int = 100,
    strata: Optional[str] = None,
    cluster: Optional[str] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    ri_coefs = np.zeros(reps)
    fml_update = fml.replace(resampvar, f"{resampvar}_resampled")

    fixest_module = import_module("pyfixest.estimation")
    fit_ = getattr(fixest_module, "feols")

    for i in range(reps):
        data_resampled = _resample(
            data=data, resampvar=resampvar, rng=rng, cluster=cluster
        )
        fit = fit_(fml_update, data=data_resampled, vcov="iid")
        ri_coefs[i] = fit.coef().xs(f"{resampvar}_resampled")

    return ri_coefs


def _get_ritest_pvalue(
    sample_coef: np.ndarray, ri_coefs: np.ndarray, method: str
) -> np.ndarray:
    if method == "rk":
        p_value = (ri_coefs <= sample_coef).mean()
    elif method == "rk_abs":
        p_value = (np.abs(ri_coefs) >= np.abs(sample_coef)).mean()
    elif method in ["right", "left"]:
        p_value_rk = (ri_coefs <= sample_coef).mean()
        M = len(ri_coefs)
        p_value = 1 - p_value_rk / M if method == "right" else p_value_rk / M
    else:
        raise ValueError(
            "The `method` argument must be one of 'rk', 'rk_abs', 'right', 'left'."
        )

    return p_value


def _get_ritest_confint(
    alpha: float, sample_coef: np.ndarray, ri_coefs: np.ndarray
) -> np.ndarray:
    ri_coefs_centered = sample_coef - ri_coefs
    lower = np.quantile(ri_coefs_centered, alpha / 2)
    upper = np.quantile(ri_coefs_centered, 1 - alpha / 2)

    return np.array([lower, upper])


def _plot_ritest_pvalue(
    sample_stat: np.ndarray, ri_stats: np.ndarray, method: str, ax=None, **kwargs
):
    pass


def _resample(
    data: pd.DataFrame, resampvar: str, rng, cluster: Optional[str] = None
) -> pd.DataFrame:
    if cluster is None:
        resampvar_values = data[resampvar].dropna().unique()
        N = data.shape[0]
        D_treat = rng.choice(resampvar_values, N, replace=True)
    else:
        # check that all observations in the same cluster have the same resampvar value
        G = data[cluster].dropna().unique()
        unique_counts = data.groupby(cluster)[resampvar].nunique()
        all_unique = np.all(unique_counts == 1)
        if not all_unique:
            raise ValueError(
                "The resampling variable must be unique within each cluster for clustered sampling."
            )

        resampvar_values = data[resampvar].dropna().unique()
        N = data.shape[0]
        D_treat = rng.choice(resampvar_values, G, replace=True)
        D_treat = np.repeat(D_treat, unique_counts)

    data[f"{resampvar}_resampled"] = D_treat

    return data
