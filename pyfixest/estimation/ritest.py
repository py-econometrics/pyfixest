from importlib import import_module
from typing import Optional, Union

import numba as nb
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
    """
    Compute tests statistics using randomization inference (slow).

    Parameters
    ----------
    data : pd.DataFrame
        The input data set.
    resampvar : str
        The name of the treatment variable.
    fml : str
        The formula of the regression model.
    type : str
        The type of the test statistic. Must be one of 'randomization-c'
        or 'randomization-t'. If 'randomization-c', the statistic
        is the regression coefficient.
        If 'randomization-t', the statistic is the t-statistic.
    reps : int
        The number of repetitions.
    model : str
        The model to estimate. Must be one of 'feols' or 'fepois'.
    rng : np.random.Generator
        The random number generator.
    vcov : str or dict[str, str]
        The type of covarianc estimator. See `feols` or `fepois` for details.
    clustervar_arr : np.ndarray, optional
        Array containing the cluster variable. Defaults to None.

    Returns
    -------
    np.ndarray
        The test statistics. For this algorithm, regression coefficients are returned if
        `type` is 'randomization-c', otherwise t-statistics are returned.

    """
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
    Y: np.ndarray,
    X: np.ndarray,
    D: np.ndarray,
    coefnames: list,
    resampvar: str,
    reps: int,
    rng: np.random.Generator,
    algo_iterations: int,
    weights: np.ndarray,
    clustervar_arr: Optional[np.ndarray] = None,
    fval_df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Compute tests statistics using randomization inference (fast).

    Parameters
    ----------
    Y : np.ndarray
        The dependent variable. If the model has fixed effects, the dependent
        variable must be demeaned.
    X : np.ndarray
        The design matrix. If the model has fixed effects, the design matrix
        must be demeaned.
    D : np.ndarray
        The treatment variable.
    coefnames : list
        The names of the coefficients in the regression model.
    resampvar : str
        The name of the treatment variable.
    reps : int
        The number of repetitions.
    rng : np.random.Generator
        The random number generator.
    algo_iterations : int
        The number of iterations for the algorithm. If bigger than 1,
        core steps of the algorithm are vectorized, which may speed up
        the computation.
    weights : np.ndarray
        The sample weights.
    clustervar_arr : np.ndarray, optional
        Array containing the cluster variable. Defaults to None.
    fval_df : pd.DataFrame, optional
        The fixed effects. Defaults to None.

    Returns
    -------
    np.ndarray
        The test statistics. For this algorithm, regression coefficients are
        returned.
    """
    X_demean = X
    Y_demean = Y.flatten()

    if fval_df is not None:
        fval = np.zeros_like(fval_df)
        for i, col in enumerate(fval_df.columns):
            fval[:, i] = pd.factorize(fval_df[col])[0]

        if fval.dtype != int:
            fval = fval.astype(int)

        fval = fval.reshape(-1, 1) if fval.ndim == 1 else fval
    else:
        fval = None

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

    return _run_ri(
        reps=reps,
        algo_iterations=algo_iterations,
        iteration_length=iteration_length,
        last_iteration=last_iteration,
        resampvar_arr=resampvar_arr,
        clustervar_arr=clustervar_arr,
        fwl_error_1=fwl_error_1,
        X_demean2=X_demean2,
        fval=fval,
        weights=weights,
        rng=rng,
    )


def _get_ritest_pvalue(
    sample_stat: np.ndarray, ri_stats: np.ndarray, method: str
) -> np.ndarray:
    """Compute the p-value of the test statistic."""
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
    """Plot the permutation distribution of the test statistic."""
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


# @nb.njit
def _run_ri(
    reps: int,
    algo_iterations: int,
    iteration_length: int,
    last_iteration: int,
    resampvar_arr: np.ndarray,
    fwl_error_1: np.ndarray,
    rng: np.random.Generator,
    fval: np.ndarray,
    weights: np.ndarray,
    X_demean2: np.ndarray,
    clustervar_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Run the randomization inference.

    Parameters
    ----------
    reps : int
        The number of repetitions.
    algo_iterations : int
        The number of iterations for the algorithm. If bigger than 1,
        core steps of the algorithm are vectorized, which may speed up
        the computation.
    iteration_length : int
        The number of repetitions per iteration.
    last_iteration : int
        The number of repetitions for the last iteration.
    resampvar_arr : np.ndarray
        Array containing the treatment variable.
    fwl_error_1 : np.ndarray
        The FWL residuals from the first stage of the FWL algorithm.
    rng : np.random.Generator
        The random number generator.
    fval : np.ndarray
        The fixed effects decoded as integers.
    weights : np.ndarray
        The sample weights.
    X_demean2 : np.ndarray
        The demeaned design matrix.
    clustervar_arr : np.ndarray, optional
        Array containing the cluster variable. Defaults to None.

    Returns
    -------
    np.ndarray
        The coefficients of the randomization inference.
        For this algorithm, regression coefficients are
        returned.
    """
    is_last_iteration = False
    ri_coefs = np.zeros(reps)

    for i in nb.prange(algo_iterations):
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

        D2_demean = demean(D2, fval, weights)[0] if fval is not None else D2

        fwl_error_2 = D2_demean - X_demean2 @ np.linalg.lstsq(X_demean2, D2_demean)[0]

        if is_last_iteration:
            ri_coefs[(i * iteration_length) :] = np.linalg.lstsq(
                fwl_error_2, fwl_error_1
            )[0]
        else:
            ri_coefs[i * iteration_length : (i + 1) * iteration_length] = (
                np.linalg.lstsq(fwl_error_2, fwl_error_1)[0]
            )

    return ri_coefs


@nb.njit
def _resample(
    resampvar_arr: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 1,
    clustervar_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Random resampling of the treatment variable.

    Parameters
    ----------
    resampvar_arr : np.ndarray
        Array containing the treatment variable.
    rng : np.random.Generator
        The random number generator.
    iterations : int, optional
        The number of iterations. Defaults to 1. If bigger than 1,
        return `iterations` resampled treatment variables. Basically,
        this argument allows to vectorize the resampling process.
    clustervar_arr : np.ndarray, optional
        Array containing the cluster variable. Defaults to None.

    Returns
    -------
    np.ndarray
        The resampled treatment variable(s). If `iterations` is bigger
        than 1, the array has shape (N, iterations), where N is the
        number of observations. Otherwise, the array has shape (N,1).
    """
    N = resampvar_arr.shape[0]
    resampvar_values = np.unique(resampvar_arr)

    if clustervar_arr is not None:
        clustervar_values = np.unique(clustervar_arr)
        D_treat = np.zeros((N, iterations))

        for i, _ in enumerate(clustervar_values):
            idx = (clustervar_arr == clustervar_values[i]).flatten()
            D_treat[idx, :] = random_choice(resampvar_values, 1 * iterations, rng)

    else:
        D_treat = random_choice(resampvar_values, N * iterations, rng).reshape(
            (N, iterations)
        )

    return D_treat


@nb.njit
def random_choice(arr: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly sample from an array.

    Parameters
    ----------
    arr : np.ndarray
        The array from which to sample.
    size : int
        The number of samples.
    rng : np.random.Generator
        The random number generator.

    Returns
    -------
    np.ndarray
        The sampled array (with replacement) of size `size`.
    """
    n = len(arr)
    result = np.empty(size, dtype=arr.dtype)
    for i in range(size):
        idx = rng.integers(0, n)
        result[i] = arr[idx]
    return result
