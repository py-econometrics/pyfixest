from importlib import import_module
from typing import Optional, Union

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import seaborn as sns

# Make lets-plot an optional dependency
try:
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

    _HAS_LETS_PLOT = True
except ImportError:
    _HAS_LETS_PLOT = False

from scipy.stats import norm
from tqdm import tqdm

from pyfixest.estimation.internals.demean_ import demean

# Only setup lets-plot if it's available
if _HAS_LETS_PLOT:
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

    resampvar_arr = data_resampled[resampvar].to_numpy()

    ri_stats = np.zeros(reps)

    for i in tqdm(range(reps)):
        D_treat = _resample(
            resampvar_arr=resampvar_arr,
            clustervar_arr=clustervar_arr,
            rng=rng,
            iterations=1,
        ).flatten()

        data_resampled[f"{resampvar}_resampled"] = D_treat

        fixest_fit = fit_(fml_update, data=data_resampled, vcov=vcov)
        if type == "randomization-c":
            ri_stats[i] = fixest_fit.coef().xs(f"{resampvar}_resampled")
        else:
            ri_stats[i] = fixest_fit.tstat().xs(f"{resampvar}_resampled")

    return ri_stats


def _get_ritest_stats_fast(
    Y: np.ndarray,
    X: np.ndarray,
    D: np.ndarray,
    coefnames: list,
    resampvar: str,
    reps: int,
    rng: np.random.Generator,
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

    resampvar_arr = D

    return _run_ri(
        reps=reps,
        resampvar_arr=resampvar_arr,
        clustervar_arr=clustervar_arr,
        Y_demean=Y_demean,
        X_demean2=X_demean2,
        fval=fval,
        weights=weights,
        rng=rng,
    )


@nb.njit()
def _run_ri(
    reps: int,
    resampvar_arr: np.ndarray,
    rng: np.random.Generator,
    fval: np.ndarray,
    weights: np.ndarray,
    Y_demean: np.ndarray,
    X_demean2: np.ndarray,
    clustervar_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Run the randomization inference.

    Parameters
    ----------
    reps : int
        The number of repetitions.
    resampvar_arr : np.ndarray
        Array containing the treatment variable.
    rng : np.random.Generator
        The random number generator.
    fval : np.ndarray
        The fixed effects decoded as integers.
    weights : np.ndarray
        The sample weights.
    X_demean2 : np.ndarray
        The demeaned design matrix.
    Y_demean : np.ndarray
    clustervar_arr : np.ndarray, optional
        Array containing the cluster variable. Defaults to None.

    Returns
    -------
    np.ndarray
        The coefficients of the randomization inference.
        For this algorithm, regression coefficients are
        returned.
    """
    ri_coefs = np.zeros(reps)

    X_demean2 = np.ascontiguousarray(X_demean2)

    for i in range(reps):
        D2 = _resample(
            resampvar_arr=resampvar_arr,
            clustervar_arr=clustervar_arr,
            rng=rng,
            iterations=1,
        )

        D2_demean = demean(D2, fval, weights)[0] if fval is not None else D2

        ri_coefs[i] = lstsq_numba(
            np.concatenate((D2_demean, X_demean2), axis=1), Y_demean
        )[0]

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


@nb.njit()
def lstsq_numba(A, B):
    """Implement np.linalg.lstsq(A, B) using SVD decomposition."""
    U, s, VT = np.linalg.svd(A, full_matrices=False)
    c = np.dot(U.T, B)
    w = np.linalg.solve(np.diag(s), c)
    x = np.dot(VT.T, w)
    return x


def _get_ritest_pvalue(
    sample_stat: np.ndarray,
    ri_stats: np.ndarray,
    method: str,
    level: float,
    h0_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the p-value of the test statistic and
    standard error and CI of the p-value.
    """
    reps = len(ri_stats)
    ci_sides = [0, 1]

    if method == "two-sided":
        probs = np.abs(ri_stats) >= np.abs(sample_stat - h0_value)
    elif method == "greater":
        probs = ri_stats <= sample_stat - h0_value
    elif method == "lower":
        probs = ri_stats >= sample_stat - h0_value
    else:
        raise ValueError(
            "The `method` argument must be one of 'two-sided', 'right', 'left'."
        )

    p_value = probs.mean()
    se_pval = norm.ppf(level) * np.std(probs) / np.sqrt(reps)
    ci_margin = norm.ppf(level) * se_pval
    ci_pval = p_value + np.array([-ci_margin, ci_margin])[ci_sides]

    return p_value, se_pval, ci_pval


def _plot_ritest_pvalue(
    sample_stat: np.ndarray, ri_stats: np.ndarray, plot_backend: str
):
    """Plot the permutation distribution of the test statistic."""
    df = pd.DataFrame({"ri_stats": ri_stats})

    title = "Permutation distribution of the test statistic"
    x_lab = "Test statistic"
    y_lab = "Density"

    if plot_backend == "lets_plot":
        if not _HAS_LETS_PLOT:
            print("lets-plot is not installed. Falling back to matplotlib.")
            plot_backend = "matplotlib"
        else:
            plot = (
                ggplot(df, aes(x="ri_stats"))
                + geom_density(fill="blue", alpha=0.5)
                + theme_bw()
                + geom_vline(xintercept=sample_stat, color="red")
                + ggtitle(title)
                + xlab(x_lab)
                + ylab(y_lab)
            )

        return plot.show()

    elif plot_backend == "matplotlib":
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x="ri_stats", fill=True, color="blue", alpha=0.5)
        plt.axvline(x=sample_stat, color="red", linestyle="--")
        plt.title(title)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.show()

    else:
        raise ValueError(f"Unsupported plot backend: {plot_backend}")


def _decode_resampvar(resampvar: str) -> tuple[str, float, str, str]:
    """
    Decode the resampling variable.

    Parameters
    ----------
    resampvar : str
        The resampling variable. It can be of the form "var=value",
        "var>value", or "var<value".

    Returns
    -------
    str
        The name of the resampling variable, h0_value (float),
        transformed hypothesis,
        and test_type (two-sided, greater, lower).
    """
    if "=" in resampvar:
        resampvar_, h0_value_str = resampvar.split("=")
        test_type = "two-sided"
        hypothesis = f"{resampvar_}={h0_value_str}"
    elif ">" in resampvar:
        resampvar_, h0_value_str = resampvar.split(">")
        test_type = "greater"
        hypothesis = f"{resampvar_}>{h0_value_str}"
    elif "<" in resampvar:
        resampvar_, h0_value_str = resampvar.split("<")
        test_type = "lower"
        hypothesis = f"{resampvar_}<{h0_value_str}"
    else:
        resampvar_ = resampvar
        test_type = "two-sided"
        hypothesis = f"{resampvar_}=0"
        h0_value_str = "0"

    h0_value_float = float(h0_value_str)

    return resampvar_, h0_value_float, hypothesis, test_type
