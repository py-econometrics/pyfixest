from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
from numpy.random import Generator
from scipy.stats import t

from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation import feols
from pyfixest.estimation.internals.model_state import (
    EstimationSample,
    VarianceCovariance,
    WithinLinearData,
)


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
    demeaner: AnyDemeaner,
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
    demeaner : AnyDemeaner
        Demeaner configuration used by the original model.
    """
    unique_clusters = np.unique(cluster_vec)
    N = data.shape[0]
    G = len(unique_clusters)

    Z = rng.choice([False, True], size=N)
    # compute alpha, tau using Z == 0
    fit_split1 = feols(fml, data[Z], demeaner=demeaner)
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
            fit_m_full = feols(fml, data[ind_m], demeaner=demeaner)
            aux_tau_full = float(fit_m_full.coef().xs(treatment))  # type: ignore[arg-type]

        # treatment effect in cluster for subsample
        if treatment_nested_in_cluster_split:
            aux_tau = tau
        else:
            fit_m = feols(fml, data[ind_m_and_split], demeaner=demeaner)
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


def _run_ccv(
    *,
    fml: str,
    data: pd.DataFrame,
    W: np.ndarray,
    treatment: str,
    cluster_vec: np.ndarray,
    tau_full: np.ndarray,
    rng: Generator,
    n_splits: int,
    pk: float,
    qk: float,
    within_data: WithinLinearData,
    sample_info: EstimationSample,
    variance_covariance: VarianceCovariance,
    coefnames: list[str],
    demeaner: AnyDemeaner,
) -> pd.Series:
    """Compute causal cluster inference from retained typed model values."""
    Y = within_data.response.flatten()
    X = within_data.design
    N = sample_info.n_obs
    G = len(np.unique(cluster_vec))
    ccv_module = import_module("pyfixest.estimation.post_estimation.ccv")
    _compute_CCV = ccv_module._compute_CCV

    vcov_splits = 0.0
    for _ in range(n_splits):
        vcov_ccv = _compute_CCV(
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
            demeaner=demeaner,
        )
        vcov_splits += vcov_ccv

    vcov_splits /= n_splits
    vcov_splits /= N

    crv1_idx = coefnames.index(treatment)
    vcov_crv1 = variance_covariance.vcov[crv1_idx, crv1_idx]
    vcov_ccv = qk * vcov_splits + (1 - qk) * vcov_crv1

    se = np.sqrt(vcov_ccv)
    tstat = tau_full / se
    df = G - 1
    pvalue = 2 * (1 - t.cdf(np.abs(tstat), df))
    alpha = 0.95
    z = np.abs(t.ppf((1 - alpha) / 2, df))
    z_se = z * se
    conf_int = np.array([tau_full - z_se, tau_full + z_se])

    res_ccv_dict: dict[str, float | np.ndarray] = {
        "Estimate": tau_full,
        "Std. Error": se,
        "t value": tstat,
        "Pr(>|t|)": pvalue,
        "2.5%": conf_int[0],
        "97.5%": conf_int[1],
    }

    res_ccv = pd.Series(res_ccv_dict)

    res_ccv.name = "CCV"

    return res_ccv
