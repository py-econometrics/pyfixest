from importlib import import_module
from typing import Optional, Union

import numpy as np
import pandas as pd
from lets_plot import (
    LetsPlot,
    aes,
    geom_density,
    geom_vline,
    ggplot,
    ggtitle,
    theme_bw,
    xlab,
    ylab,
)
from tqdm import tqdm

from pyfixest.estimation.demean_ import demean

LetsPlot.setup_html()


def _get_ritest_stats_slow(
    data: pd.DataFrame,
    resampvar: str,
    fml: str,
    type: str,
    reps: int,
    model: str,
    rng: np.random.Generator,
    vcov: Union[str, dict[str, str]],
) -> np.ndarray:
    fml_update = fml.replace(resampvar, f"{resampvar}_resampled")

    fixest_module = import_module("pyfixest.estimation")
    fit_ = getattr(fixest_module, model)

    ri_stats = np.zeros(reps)
    for i in tqdm(range(reps)):
        data_resampled = _resample(
            data=data, resampvar=resampvar, rng=rng, cluster=None
        )
        fit = fit_(fml_update, data=data_resampled, vcov=vcov)
        if type == "randomization-c":
            ri_stats[i] = fit.coef().xs(f"{resampvar}_resampled")
        else:
            ri_stats[i] = fit.tstat().xs(f"{resampvar}_resampled")

    return ri_stats


def _get_ritest_stats_fast(
    Y,
    X,
    D,
    coefnames,
    resampvar,
    reps,
    rng,
    has_fixef,
    fval_df,
    algo_iterations,
    weights,
):
    X_demean = X
    Y_demean = Y.flatten()
    coefnames = coefnames
    has_fixef = has_fixef

    fval = np.zeros_like(fval_df)
    for i, col in enumerate(fval_df.columns):
        fval[:, i] = pd.factorize(fval_df[col].to_numpy())[0]

    if fval.dtype != int:
        fval = fval.astype(int)

    idx = coefnames.index(resampvar)
    bool_idx = np.ones(len(coefnames), dtype=bool)
    bool_idx[idx] = False

    X_demean2 = X_demean[:, bool_idx]

    D = D.reshape(-1, 1) if D.ndim == 1 else D
    X_demean2 = X_demean2.reshape(-1, 1) if X_demean2.ndim == 1 else X_demean2
    fval = fval.reshape(-1, 1) if fval.ndim == 1 else fval

    # fwl step 1:
    unique_D = np.unique(D)
    N = D.shape[0]

    # FWL Stage 1 on "constant" data
    fwl_error_1 = (
        Y_demean - X_demean2 @ np.linalg.lstsq(X_demean2, Y_demean, rcond=None)[0]
    )

    iteration_length = reps // algo_iterations
    last_iteration = reps % algo_iterations

    def _run_ri(
        algo_iterations, iteration_length, last_iteration, unique_D, N, fwl_error_1, rng
    ):
        ri_coefs = np.zeros(reps)

        for i in tqdm(range(algo_iterations)):
            if i == (algo_iterations - 1) and last_iteration > 0:
                D2 = rng.choice(unique_D, N * last_iteration, True).reshape(
                    (N, last_iteration)
                )
            else:
                D2 = rng.choice(unique_D, N * iteration_length, True).reshape(
                    (N, iteration_length)
                )

            if has_fixef:
                D2_demean, _ = demean(D2, fval, weights)
            else:
                D2_demean = D2

            fwl_error_2 = (
                D2_demean
                - X_demean2 @ np.linalg.lstsq(X_demean2, D2_demean, rcond=None)[0]
            )

            ri_coefs[i * iteration_length : (i + 1) * iteration_length] = (
                np.linalg.lstsq(fwl_error_2, fwl_error_1, rcond=None)[0]
            )

        return ri_coefs

    return _run_ri(
        algo_iterations=algo_iterations,
        iteration_length=iteration_length,
        last_iteration=last_iteration,
        unique_D=unique_D,
        N=N,
        fwl_error_1=fwl_error_1,
        rng=rng,
    )


def _get_ritest_pvalue(
    sample_stat: np.ndarray, ri_stats: np.ndarray, method: str
) -> np.ndarray:
    if method == "rk":
        p_value = (ri_stats <= sample_stat).mean()
    elif method == "rk_abs":
        p_value = (np.abs(ri_stats) >= np.abs(sample_stat)).mean()
    elif method in ["right", "left"]:
        p_value_rk = (ri_stats <= sample_stat).mean()
        M = len(ri_stats)
        p_value = 1 - p_value_rk / M if method == "right" else p_value_rk / M
    else:
        raise ValueError(
            "The `method` argument must be one of 'rk', 'rk_abs', 'right', 'left'."
        )

    return p_value


def _get_ritest_confint(
    alpha: float, sample_stat: np.ndarray, ri_stats: np.ndarray
) -> np.ndarray:
    ri_stats_centered = sample_stat - ri_stats
    lower = np.quantile(ri_stats_centered, alpha / 2)
    upper = np.quantile(ri_stats_centered, 1 - alpha / 2)

    return np.array([lower, upper])


def _plot_ritest_pvalue(sample_stat: np.ndarray, ri_stats: np.ndarray):
    df = pd.DataFrame({"ri_stats": ri_stats})

    plot = (
        ggplot(df, aes(x="ri_stats"))
        + geom_density(fill="blue", alpha=0.5)
        + theme_bw()
        + geom_vline(xintercept=sample_stat, color="red")
        + ggtitle("Permutation distribution of the test statistic")
        + xlab("Test statistic")
        + ylab("Density")
    )

    return plot.show()


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
