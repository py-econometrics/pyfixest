import warnings
from importlib import import_module
from typing import cast

import numpy as np
import pandas as pd
from numpy.random import Generator
from scipy.stats import t


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
    # lazy import to avoid a circular import at module load time
    feols = import_module("pyfixest.estimation").feols

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


def _ccv_impl(
    model,
    treatment,
    cluster: str | None = None,
    seed: int | None = None,
    n_splits: int = 8,
    pk: float = 1,
    qk: float = 1,
) -> pd.DataFrame:
    "Implementation of Feols.ccv; see the method docstring for details."
    assert model._supports_cluster_causal_variance, (
        "The model does not support the causal cluster variance estimator."
    )
    assert isinstance(treatment, str), "treatment must be a string."
    assert isinstance(cluster, str) or cluster is None, (
        "cluster must be a string or None."
    )
    assert isinstance(seed, int) or seed is None, "seed must be an integer or None."
    assert isinstance(n_splits, int), "n_splits must be an integer."
    assert isinstance(pk, (int, float)) and 0 <= pk <= 1
    assert isinstance(qk, (int, float)) and 0 <= qk <= 1

    if model._has_fixef:
        raise NotImplementedError(
            "The causal cluster variance estimator is currently not supported for models with fixed effects."
        )

    if treatment not in model._coefnames:
        raise ValueError(
            f"Variable {treatment} not found in the model's coefficients."
        )

    if cluster is None:
        if model._clustervar is None:
            raise ValueError("No cluster variable found in the model fit.")
        elif len(model._clustervar) > 1:
            raise ValueError(
                "Multiway clustering is currently not supported with the causal cluster variance estimator."
            )
        else:
            cluster = model._clustervar[0]

    # check that cluster is in data
    if cluster not in model._data.columns:
        raise ValueError(
            f"Cluster variable {cluster} not found in the data used for the model fit."
        )

    if not model._is_clustered:
        warnings.warn(
            "The initial model was not clustered. CRV1 inference is computed and stored in the model object."
        )
        model.vcov({"CRV1": cluster})

    if seed is None:
        seed = np.random.randint(1, 100_000_000)
    rng = np.random.default_rng(seed)

    fml = model._fml

    data = model._data
    Y = model._Y_wls.flatten()
    W = data[treatment].to_numpy()
    assert np.all(np.isin(W, [0, 1])), (
        "Treatment variable must be binary with values 0 and 1"
    )
    X = model._X_wls
    cluster_vec = data[cluster].to_numpy()
    unique_clusters = np.unique(cluster_vec)

    tau_full = np.array(model.coef().xs(treatment))

    N = model._N
    G = len(unique_clusters)

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
        )
        vcov_splits += vcov_ccv

    vcov_splits /= n_splits
    vcov_splits /= N

    crv1_idx = model._coefnames.index(treatment)
    vcov_crv1 = model._vcov[crv1_idx, crv1_idx]
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

    res_crv1 = cast(pd.Series, model.tidy().xs(treatment))
    res_crv1.name = "CRV1"

    return pd.concat([res_ccv, res_crv1], axis=1).T
