from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.random import Generator
from scipy.stats import t

from pyfixest.estimation import feols


def _compute_CCV(
    fml: str,
    Y: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    rng: Generator,
    data: pd.DataFrame,
    treatment: str,
    cluster_vec: np.ndarray,
    pk: float,
    tau_full: float,
) -> float:
    """
    Compute the causal cluster variance estimator following Abadie et al (QJE 2023).

    Parameters
    ----------
    fml : str
        Formula of the regression model.
    Y : np.array
        Array with the dependent variable.
    X : np.array
        Array of the regression design matrix.
    W : np.array
        Array with the treatment variable.
    rng : np.random.default_rng
        Random number generator.
    data : pd.DataFrame
        Dataframe with the data.
    treatment : str
        Name of the treatment variable.
    cluster_vec : np.array
        Array with unique cluster identifiers.
    pk : float between 0 and 1.
        The proportion of clusters sampled.
        Default is 1, which means all clusters are sampled.
    tau_full : float
        The treatment effect estimate for the full sample.
    """
    unique_clusters = np.unique(cluster_vec)
    N = data.shape[0]
    G = len(unique_clusters)

    Z = rng.choice([False, True], size=N)
    # compute alpha, tau using Z == 0
    fit_split1 = feols(fml, data[Z])
    coefs_split = fit_split1.coef().to_numpy()
    tau = fit_split1.coef().xs(treatment)

    # estimate treatment effect for each cluster
    # for both the full sample and the subsample
    pk_term = 0.0
    tau_ms = np.zeros(G)
    N = 0
    for i, m in enumerate(unique_clusters):
        ind_m = cluster_vec == m
        Nm = np.sum(ind_m)
        N += Nm
        ind_m_and_split = ind_m & Z

        treatment_nested_in_cluster = data.loc[ind_m, treatment].nunique() == 1
        treatment_nested_in_cluster_split = (
            data.loc[ind_m_and_split, treatment].nunique() == 1
        )

        if treatment_nested_in_cluster:
            aux_tau_full = tau_full
        else:
            fit_m_full = feols(fml, data[ind_m])
            aux_tau_full = float(fit_m_full.coef().xs(treatment))  # type: ignore[arg-type]

        # treatment effect in cluster for subsample
        if treatment_nested_in_cluster_split:
            aux_tau = tau
        else:
            fit_m = feols(fml, data[ind_m_and_split])
            aux_tau = fit_m.coef().xs(treatment)
        tau_ms[i] = aux_tau

        # compute the pk term in Z0
        aux_pk = Nm * ((aux_tau_full - tau) ** 2)
        pk_term += aux_pk

    pk_term *= (1 - pk) / N
    uhat = Y - X @ coefs_split
    Wbar = np.mean(W[Z])
    Zavg = 1 - np.mean(Z)
    Zavg_squared = Zavg**2
    n_adj = N * (Wbar**2) * ((1 - Wbar) ** 2)

    vcov_ccv = 0
    for i, m in enumerate(unique_clusters):
        ind_m = cluster_vec == m

        res_term = (W[ind_m & ~Z] - Wbar) * uhat[ind_m & ~Z]
        tau_term = (tau_ms[i] - tau) * Wbar * (1.0 - Wbar)
        diff = res_term - tau_term
        sq_sum = np.sum(diff) ** 2
        sum_sq = np.sum(diff**2)
        vcov_ccv += (
            (1.0 / (Zavg**2)) * sq_sum
            - ((1.0 - Zavg) / (Zavg_squared)) * sum_sq
            + n_adj * pk_term
        )

    return vcov_ccv / n_adj


@dataclass(frozen=True, slots=True)
class CCVResult:
    """Causal cluster variance inference for one treatment coefficient.

    Attributes
    ----------
    estimate : float
        Treatment coefficient from the full-sample fit.
    se : float
        Causal cluster variance standard error.
    tstat : float
        `estimate / se`.
    pvalue : float
        Two-sided p-value against a t distribution with G - 1 degrees of freedom.
    conf_int : np.ndarray
        Lower and upper confidence bound, shape (2,).
    """

    estimate: float
    se: float
    tstat: float
    pvalue: float
    conf_int: np.ndarray


def compute_ccv(
    *,
    fml: str,
    Y: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    data: pd.DataFrame,
    treatment: str,
    cluster_vec: np.ndarray,
    tau_full: float,
    vcov_crv1: float,
    N: int,
    n_splits: int,
    pk: float,
    qk: float,
    rng: Generator,
    level: float = 0.95,
) -> CCVResult:
    """Average the causal cluster variance over splits and build its inference.

    Implements the estimator of Abadie, Athey, Imbens and Wooldridge (2023,
    QJE), https://doi.org/10.1093/qje/qjac038 . The per-split variance comes
    from `_compute_CCV`; this function averages it over `n_splits`
    cross-fitting draws, mixes it with the CRV1 variance by `qk`, and turns
    the result into a standard error, t-statistic, p-value and interval.

    Parameters
    ----------
    fml : str
        Formula of the regression model.
    Y : np.ndarray
        Dependent variable, shape (N,).
    X : np.ndarray
        Design matrix, shape (N, k).
    W : np.ndarray
        Binary treatment variable, shape (N,).
    data : pd.DataFrame
        Data used for the fit.
    treatment : str
        Name of the treatment variable.
    cluster_vec : np.ndarray
        Cluster identifier per observation, shape (N,).
    tau_full : float
        Treatment coefficient from the full-sample fit.
    vcov_crv1 : float
        CRV1 variance of the treatment coefficient.
    N : int
        Number of observations.
    n_splits : int
        Number of cross-fitting splits to average over.
    pk : float
        Share of sampled clusters.
    qk : float
        Share of sampled observations within each cluster.
    rng : Generator
        Random number generator.
    level : float, optional
        Confidence level for the interval. Defaults to 0.95.

    Returns
    -------
    CCVResult
        The point estimate and its CCV-based inference.
    """
    G = len(np.unique(cluster_vec))

    vcov_splits = 0.0
    for _ in range(n_splits):
        vcov_splits += _compute_CCV(
            fml=fml,
            Y=Y,
            X=X,
            W=W,
            rng=rng,
            data=data,
            treatment=treatment,
            cluster_vec=cluster_vec,
            pk=pk,
            tau_full=tau_full,
        )
    vcov_splits /= n_splits
    vcov_splits /= N

    vcov_ccv = qk * vcov_splits + (1 - qk) * vcov_crv1

    se = np.sqrt(vcov_ccv)
    tstat = tau_full / se
    df = G - 1
    pvalue = 2 * (1 - t.cdf(np.abs(tstat), df))
    z_se = np.abs(t.ppf((1 - level) / 2, df)) * se

    return CCVResult(
        estimate=tau_full,
        se=se,
        tstat=tstat,
        pvalue=pvalue,
        conf_int=np.array([tau_full - z_se, tau_full + z_se]),
    )
