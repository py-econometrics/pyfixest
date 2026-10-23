from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm
from tqdm import tqdm

from pyfixest.utils.dev_utils import _find_stack_level

# Numba is an optional dependency. The fast randomization-inference path uses it;
# the slow path does not. We import lazily so the module loads cleanly even when
# numba is not installed, and surface a clear error only when the fast path is
# actually requested.
try:
    import numba as nb

    from pyfixest.estimation.numba.demean_nb import demean

    _HAS_NUMBA = True
except ImportError:
    nb = None
    demean = None
    _HAS_NUMBA = False

_NUMBA_RITEST_ERROR = (
    "Fast randomization inference requires the optional `numba` extra. "
    "Install it with `pip install pyfixest[numba]`, or pass "
    "`choose_algorithm='slow'`."
)


def _get_ritest_stats_slow(
    data: pd.DataFrame,
    resampvar: str,
    type: str,
    reps: int,
    fit_fn: Callable[..., Any],
    rng: np.random.Generator,
    clustervar_arr: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute tests statistics using randomization inference (slow).

    Parameters
    ----------
    data : pd.DataFrame
        The input data set.
    resampvar : str
        The name of the treatment variable.
    type : str
        The type of the test statistic. Must be one of 'randomization-c'
        or 'randomization-t'. If 'randomization-c', the statistic
        is the regression coefficient.
        If 'randomization-t', the statistic is the t-statistic.
    reps : int
        The number of repetitions.
    fit_fn : Callable[..., Any]
        Refits the model on each resampled data set with the fitted model's
        estimation options and the covariance estimator of the test statistic,
        such as `refit` with the model and `vcov` bound. It is called with
        the keyword argument `data` only, reusing the fitted parsed formula.
    rng : np.random.Generator
        The random number generator.
    clustervar_arr : np.ndarray, optional
        Array containing the cluster variable. Defaults to None.

    Returns
    -------
    np.ndarray
        The test statistics. For this algorithm, regression coefficients are returned if
        `type` is 'randomization-c', otherwise t-statistics are returned.

    """
    # the resampled column goes into a shallow copy, so `data` keeps its
    # columns; `fit_fn` must not modify its input
    data_resampled = data.copy(deep=False)

    assignment, unit_codes = _assignment_units(
        resampvar_arr=data_resampled[resampvar].to_numpy(),
        clustervar_arr=clustervar_arr,
    )

    ri_stats = np.zeros(reps)

    for i in tqdm(range(reps)):
        data_resampled[resampvar] = _resample(
            assignment=assignment, unit_codes=unit_codes, rng=rng
        )

        fixest_fit = fit_fn(data=data_resampled)
        if type == "randomization-c":
            ri_stats[i] = fixest_fit.coef().xs(resampvar)
        else:
            ri_stats[i] = fixest_fit.tstat().xs(resampvar)

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
    clustervar_arr: np.ndarray | None = None,
    fval_df: pd.DataFrame | None = None,
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
    if nb is None or demean is None:
        raise ImportError(_NUMBA_RITEST_ERROR)

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

    X_demean2 = X_demean2.reshape(-1, 1) if X_demean2.ndim == 1 else X_demean2

    assignment, unit_codes = _assignment_units(
        resampvar_arr=D, clustervar_arr=clustervar_arr
    )

    return _run_ri(
        reps=reps,
        assignment=assignment,
        unit_codes=unit_codes,
        Y_demean=Y_demean,
        X_demean2=X_demean2,
        fval=fval,
        weights=weights,
        rng=rng,
        demean_fn=demean,
    )


def _run_ri(
    reps: int,
    assignment: np.ndarray,
    unit_codes: np.ndarray,
    rng: np.random.Generator,
    fval: np.ndarray | None,
    weights: np.ndarray,
    Y_demean: np.ndarray,
    X_demean2: np.ndarray,
    demean_fn: Callable[..., Any],
) -> np.ndarray:
    """
    Run the randomization inference.

    Parameters
    ----------
    reps : int
        The number of repetitions.
    assignment : np.ndarray
        The treatment of each unit of random assignment.
    unit_codes : np.ndarray
        The index into `assignment` of each observation.
    rng : np.random.Generator
        The random number generator.
    fval : np.ndarray
        The fixed effects decoded as integers.
    weights : np.ndarray
        The sample weights.
    X_demean2 : np.ndarray
        The demeaned design matrix.
    Y_demean : np.ndarray
    demean_fn : Callable
        The numba-backed `demean` function, validated as importable by the caller.

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
        D2 = _resample(assignment=assignment, unit_codes=unit_codes, rng=rng).reshape(
            (-1, 1)
        )

        D2_demean = demean_fn(D2, fval, weights)[0] if fval is not None else D2

        ri_coefs[i] = lstsq_numba(
            np.concatenate((D2_demean, X_demean2), axis=1), Y_demean
        )[0]

    return ri_coefs


def _assignment_units(
    resampvar_arr: np.ndarray, clustervar_arr: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collapse the treatment to its units of random assignment.

    Without clusters, every observation is a unit. With clusters, every
    cluster is a unit, and the caller must ensure that the treatment is
    constant within each cluster.

    Parameters
    ----------
    resampvar_arr : np.ndarray
        Array containing the treatment variable.
    clustervar_arr : np.ndarray, optional
        Array containing the cluster variable. Defaults to None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The treatment of each unit, and the index of each observation's unit.
    """
    D = np.ascontiguousarray(resampvar_arr, dtype=np.float64).ravel()

    if clustervar_arr is None:
        return D, np.arange(D.shape[0])

    unit_codes, units = pd.factorize(np.asarray(clustervar_arr).ravel())
    assignment = np.empty(len(units))
    assignment[unit_codes] = D

    return assignment, unit_codes


def _resample(
    assignment: np.ndarray, unit_codes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Draw a treatment assignment by permuting the observed one.

    A permutation keeps the number of treated units and, for a non-binary
    treatment, the distribution of treatment values fixed across draws.

    Parameters
    ----------
    assignment : np.ndarray
        The treatment of each unit of random assignment.
    unit_codes : np.ndarray
        The index into `assignment` of each observation.
    rng : np.random.Generator
        The random number generator.

    Returns
    -------
    np.ndarray
        The resampled treatment of each observation, of shape (N,).
    """
    return rng.permutation(assignment)[unit_codes]


def lstsq_numba(A, B):
    """Implement np.linalg.lstsq(A, B) using SVD decomposition."""
    U, s, VT = np.linalg.svd(A, full_matrices=False)
    c = np.dot(U.T, B)
    w = np.linalg.solve(np.diag(s), c)
    x = np.dot(VT.T, w)
    return x


if nb is not None:
    _resample = nb.njit(_resample)
    lstsq_numba = nb.njit()(lstsq_numba)
    _run_ri = nb.njit()(_run_ri)


def _get_ritest_pvalue(
    sample_stat: np.ndarray,
    ri_stats: np.ndarray,
    method: str,
    level: float,
    h0_value: float,
) -> tuple[np.float64, np.float64, np.ndarray]:
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
    sample_stat: float, ri_stats: NDArray[np.float64], plot_backend: str
):
    """Plot the permutation distribution of the test statistic."""
    df = pd.DataFrame({"ri_stats": ri_stats})

    title = "Permutation distribution of the test statistic"
    x_lab = "Test statistic"
    y_lab = "Density"

    if plot_backend == "lets_plot":
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
        except ImportError:
            warnings.warn(
                "lets-plot is not installed. Falling back to matplotlib.",
                UserWarning,
                stacklevel=_find_stack_level(),
            )
            plot_backend = "matplotlib"
        else:
            LetsPlot.setup_html()
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

    if plot_backend == "matplotlib":
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x="ri_stats", fill=True, color="blue", alpha=0.5)
        plt.axvline(x=sample_stat, color="red", linestyle="--")
        plt.title(title)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.show()

        return None

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
