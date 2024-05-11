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
    clustervar_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    data_resampled = data.copy()
    fml_update = fml.replace(resampvar, f"{resampvar}_resampled")

    fixest_module = import_module("pyfixest.estimation")
    fit_ = getattr(fixest_module, model)

    resampvar_arr = data[resampvar].dropna().to_numpy()

    ri_stats = np.zeros(reps)
    for i in tqdm(range(reps)):
        D_treat = _resample(
            resampvar_arr=resampvar_arr,
            clustervar_arr=clustervar_arr,
            rng=rng,
            iterations=1,
        )

        data_resampled[f"{resampvar}_resampled"] = D_treat
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
    clustervar_arr: Optional[np.ndarray] = None,
):
    X_demean = X
    Y_demean = Y.flatten()

    if has_fixef:
        fval = np.zeros_like(fval_df)
        for i, col in enumerate(fval_df.columns):
            fval[:, i] = pd.factorize(fval_df[col].to_numpy())[0]

        if fval.dtype != int:
            fval = fval.astype(int)

        fval = fval.reshape(-1, 1) if fval.ndim == 1 else fval

    idx = coefnames.index(resampvar)
    bool_idx = np.ones(len(coefnames), dtype=bool)
    bool_idx[idx] = False

    X_demean2 = X_demean[:, bool_idx]

    D = D.reshape(-1, 1) if D.ndim == 1 else D
    X_demean2 = X_demean2.reshape(-1, 1) if X_demean2.ndim == 1 else X_demean2

    # FWL Stage 1 on "constant" data
    fwl_error_1 = (
        Y_demean - X_demean2 @ np.linalg.lstsq(X_demean2, Y_demean, rcond=None)[0]
    )

    iteration_length = reps // algo_iterations
    last_iteration = reps % algo_iterations

    resampvar_arr = D

    def _run_ri(
        algo_iterations,
        iteration_length,
        last_iteration,
        resampvar_arr,
        clustervar_arr,
        fwl_error_1,
        rng,
    ):
        is_last_iteration = False
        ri_coefs = np.zeros(reps)

        for i in tqdm(range(algo_iterations)):
            if last_iteration > 0 and i == (algo_iterations - 1):
                is_last_iteration = True
                D2 = _resample(
                    resampvar_arr=resampvar_arr,
                    clustervar_arr=clustervar_arr,
                    rng=rng,
                    iterations=iteration_length + last_iteration,
                )
            else:
                D2 = _resample(
                    resampvar_arr=resampvar_arr,
                    clustervar_arr=clustervar_arr,
                    rng=rng,
                    iterations=iteration_length,
                )

            if has_fixef:
                D2_demean, _ = demean(D2, fval, weights)
            else:
                D2_demean = D2

            fwl_error_2 = (
                D2_demean
                - X_demean2 @ np.linalg.lstsq(X_demean2, D2_demean, rcond=None)[0]
            )

            if is_last_iteration:
                ri_coefs[(i * iteration_length) :] = np.linalg.lstsq(
                    fwl_error_2, fwl_error_1, rcond=None
                )[0]
            else:
                ri_coefs[i * iteration_length : (i + 1) * iteration_length] = (
                    np.linalg.lstsq(fwl_error_2, fwl_error_1, rcond=None)[0]
                )

        return ri_coefs

    return _run_ri(
        algo_iterations=algo_iterations,
        iteration_length=iteration_length,
        last_iteration=last_iteration,
        resampvar_arr=resampvar_arr,
        clustervar_arr=clustervar_arr,
        fwl_error_1=fwl_error_1,
        rng=rng,
    )


def _get_ritest_pvalue(
    sample_stat: np.ndarray, ri_stats: np.ndarray, method: str
) -> np.ndarray:
    if method == "two-sided":
        p_value = (np.abs(ri_stats) >= np.abs(sample_stat)).mean()
    elif method in ["right", "left"]:
        p_value_rk = (ri_stats <= sample_stat).mean()
        M = len(ri_stats)
        p_value = 1 - p_value_rk / M if method == "right" else p_value_rk / M
    else:
        raise ValueError(
            "The `method` argument must be one of 'two-sided', 'right', 'left'."
        )

    return p_value


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
    resampvar_arr: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 1,
    clustervar_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    N = resampvar_arr.shape[0]
    resampvar_values = np.unique(resampvar_arr)

    if clustervar_arr is not None:
        clustervar_values = np.unique(clustervar_arr)
        D_treat = np.zeros((N, iterations))

        for i, _ in enumerate(clustervar_values):
            idx = (clustervar_arr == clustervar_values[i]).flatten()
            D_treat[idx, :] = rng.choice(resampvar_values, 1 * iterations, replace=True)

    else:
        D_treat = rng.choice(resampvar_values, N * iterations, replace=True).reshape(
            (N, iterations)
        )

    return D_treat
