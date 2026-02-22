import numpy as np
import pandas as pd
from numpy.random import Generator

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
            aux_tau_full = float(fit_m_full.coef().xs(treatment))

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
