"""
Wild Cluster Bootstrap (WCB) inference for clustered regression.

Reference
---------
Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008).
Bootstrap-based improvements for inference with clustered errors.
Review of Economics and Statistics, 90(3), 414-427.

Algorithm (impose-null approach)
---------------------------------
Given OLS model  y = X @ beta + e  with G clusters:

1. Estimate unrestricted OLS -> beta_hat, CRV1 SE -> t_obs = beta_hat[j] / se[j].
2. Estimate restricted model (drop variable j) -> beta_r, resid_r.
3. For b = 1..B:
   a. Draw G iid cluster weights w_g (Rademacher: +/-1 w.p. 1/2;
      Mammen: two-point distribution matching first two moments of N(0,1)).
   b. Construct  y*_b = X_r @ beta_r + resid_r * w_g[cluster[i]]  for each i.
   c. Regress y*_b on X (full) -> beta*_b; compute CRV1 SE -> se*_b.
   d. t*_b = beta*_b[j] / se*_b[j].
4. p_value = mean(|t*_b| >= |t_obs|)   (two-sided).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Weight generators
# ---------------------------------------------------------------------------


def _rademacher_weights(B: int, G: int, rng: np.random.Generator) -> np.ndarray:
    """Draw B x G Rademacher weights (+/-1 w.p. 1/2 each)."""
    return rng.choice(np.array([-1.0, 1.0]), size=(B, G))


def _mammen_weights(B: int, G: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw B x G two-point Mammen weights.

    The Mammen (1993) distribution has:
        w = -(sqrt(5) - 1) / 2  with prob  (sqrt(5) + 1) / (2 * sqrt(5))
        w = +(sqrt(5) + 1) / 2  with prob  (sqrt(5) - 1) / (2 * sqrt(5))

    This ensures E[w] = 0, E[w^2] = 1, and skewness = 1.
    """
    sqrt5 = np.sqrt(5.0)
    low = -(sqrt5 - 1.0) / 2.0  # approx -0.618
    high = (sqrt5 + 1.0) / 2.0  # approx  1.618
    p_low = (sqrt5 + 1.0) / (2.0 * sqrt5)  # approx 0.724

    u = rng.uniform(size=(B, G))
    return np.where(u < p_low, low, high)


def _draw_weights(
    weights_type: str,
    B: int,
    G: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return B x G weight matrix for the requested distribution."""
    if weights_type == "rademacher":
        return _rademacher_weights(B, G, rng)
    elif weights_type == "mammen":
        return _mammen_weights(B, G, rng)
    else:
        raise ValueError(
            f"weights_type must be 'rademacher' or 'mammen', got {weights_type!r}"
        )


# ---------------------------------------------------------------------------
# Core standalone function (pure numpy)
# ---------------------------------------------------------------------------


def wildboottest_numpy(
    Y: np.ndarray,
    X: np.ndarray,
    cluster_ids: np.ndarray,
    param_idx: int,
    B: int = 999,
    weights_type: Literal["rademacher", "mammen"] = "rademacher",
    seed: int | None = None,
) -> dict:
    """
    Wild Cluster Bootstrap p-value for a single coefficient (pure numpy).

    Parameters
    ----------
    Y : np.ndarray, shape (N,) or (N, 1)
        Dependent variable.
    X : np.ndarray, shape (N, k)
        Full design matrix (including the variable being tested).
    cluster_ids : np.ndarray, shape (N,)
        Integer or string cluster identifiers for each observation.
    param_idx : int
        Column index in X of the coefficient under H0: beta[param_idx] = 0.
    B : int
        Number of bootstrap replications.
    weights_type : {'rademacher', 'mammen'}
        Distribution for the cluster-level weights.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        'param_idx'    : int
        't_stat'       : float  (unrestricted CRV1 t-statistic)
        'p_value'      : float
        'B'            : int    (actual number of bootstrap draws)
        'weights_type' : str
    """
    # --- shape normalisation ---
    Y = np.asarray(Y, dtype=float).ravel()  # (N,)
    X = np.asarray(X, dtype=float)  # (N, k)
    cluster_ids = np.asarray(cluster_ids).ravel()  # (N,)
    N, k = X.shape

    if not (0 <= param_idx < k):
        raise IndexError(
            f"param_idx={param_idx} is out of bounds for X with {k} columns."
        )

    # --- map cluster labels to consecutive integers 0..G-1 ---
    unique_clusters, cluster_col = np.unique(cluster_ids, return_inverse=True)
    G = len(unique_clusters)

    # --- unrestricted OLS ---
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)  # bread
    beta_hat = XtX_inv @ (X.T @ Y)  # (k,)
    resid_unr = Y - X @ beta_hat  # (N,)

    # --- CRV1 SE for unrestricted model ---
    def _crv1_vcov(
        X_: np.ndarray, resid_: np.ndarray, bread_: np.ndarray
    ) -> np.ndarray:
        """Compute CRV1 variance-covariance matrix."""
        scores = resid_[:, None] * X_  # (N, k)
        meat = np.zeros((k, k))
        for g in range(G):
            idx = np.where(cluster_col == g)[0]
            sg = scores[idx].sum(axis=0)  # (k,)
            meat += np.outer(sg, sg)
        return bread_ @ meat @ bread_  # (k, k)

    vcov_unr = _crv1_vcov(X, resid_unr, XtX_inv)
    se_unr = np.sqrt(np.diag(vcov_unr))
    t_obs = beta_hat[param_idx] / se_unr[param_idx]

    # --- restricted OLS (drop column param_idx) ---
    col_mask = np.ones(k, dtype=bool)
    col_mask[param_idx] = False
    X_r = X[:, col_mask]  # (N, k-1)
    if X_r.shape[1] == 0:
        # Edge case: only one predictor; restricted model is empty
        beta_r = np.array([])
        resid_r = Y.copy()
        fitted_r = np.zeros(N)
    else:
        XtX_r = X_r.T @ X_r
        XtX_r_inv = np.linalg.inv(XtX_r)
        beta_r = XtX_r_inv @ (X_r.T @ Y)
        fitted_r = X_r @ beta_r
        resid_r = Y - fitted_r  # (N,)

    # --- vectorised bootstrap ---
    rng = np.random.default_rng(seed)
    W = _draw_weights(weights_type, B, G, rng)  # (B, G)

    # Expand cluster weights to observation level: (B, N)
    W_obs = W[:, cluster_col]  # (B, N)

    # Bootstrap outcomes: (B, N)
    Y_boot = fitted_r[None, :] + resid_r[None, :] * W_obs  # (B, N)

    # OLS for all B at once: beta_b = (X'X)^-1 X' Y_boot.T -> (k, B)
    beta_b = XtX_inv @ (X.T @ Y_boot.T)  # (k, B)

    # Boot residuals: (N, B)
    resid_b = Y_boot.T - X @ beta_b  # (N, B)

    # Vectorised CRV1: scores (N, k, B), then cluster sums (G, k, B)
    # score[i, j, b] = resid_b[i, b] * X[i, j]
    scores_b = resid_b[:, np.newaxis, :] * X[:, :, np.newaxis]  # (N, k, B)

    cluster_scores = np.zeros((G, k, B))
    for g in range(G):
        idx = np.where(cluster_col == g)[0]
        cluster_scores[g] = scores_b[idx].sum(axis=0)  # (k, B)

    # meat_b[j, l, b] = sum_g cluster_scores[g, j, b] * cluster_scores[g, l, b]
    meat_b = np.einsum("gjb,glb->jlb", cluster_scores, cluster_scores)  # (k, k, B)

    # vcov_b[:, :, b] = bread @ meat_b[:, :, b] @ bread
    vcov_b = np.einsum("ij,jlb,lm->imb", XtX_inv, meat_b, XtX_inv)  # (k, k, B)

    se_b = np.sqrt(vcov_b[param_idx, param_idx, :])  # (B,)
    t_boot = beta_b[param_idx, :] / se_b  # (B,)

    p_value = float(np.mean(np.abs(t_boot) >= np.abs(t_obs)))

    return {
        "param_idx": param_idx,
        "t_stat": float(t_obs),
        "p_value": p_value,
        "B": B,
        "weights_type": weights_type,
    }


# ---------------------------------------------------------------------------
# High-level wrapper that accepts a pyfixest Feols model
# ---------------------------------------------------------------------------


def wildboottest(
    model,
    param: str,
    B: int = 999,
    weights_type: Literal["rademacher", "mammen"] = "rademacher",
    seed: int | None = None,
) -> dict:
    """
    Wild Cluster Bootstrap p-value for a single coefficient.

    Implements the impose-null wild cluster bootstrap of Cameron, Gelbach &
    Miller (2008, *Review of Economics and Statistics*) for clustered inference.

    Parameters
    ----------
    model : fitted pyfixest Feols model
        Must have been estimated with a ``vcov={'CRV1': varname}`` argument so
        that cluster information (``_clustervar``, ``_cluster_df``) is available.
        Fixed-effect models are supported: the demeaned design matrix stored in
        ``model._X`` is used directly (same convention as pyfixest's own CRV
        inference).
    param : str
        Name of the coefficient under H0: beta_param = 0.
        Must appear in ``model._coefnames``.
    B : int, default 999
        Number of bootstrap replications.
    weights_type : {'rademacher', 'mammen'}, default 'rademacher'
        Distribution for the cluster-level wild weights.

        * ``'rademacher'``: +/-1 with equal probability (preferred for G >= 10).
        * ``'mammen'``: two-point distribution that also matches the third moment
          of N(0, 1); slightly more reliable for very few clusters.
    seed : int or None, default None
        Random seed passed to ``numpy.random.default_rng`` for reproducibility.

    Returns
    -------
    dict with keys:

    * ``'param'``        – str, the tested parameter name
    * ``'t_stat'``       – float, observed CRV1 t-statistic
    * ``'p_value'``      – float in [0, 1], two-sided bootstrap p-value
    * ``'B'``            – int, number of bootstrap draws used
    * ``'weights_type'`` – str

    Raises
    ------
    ValueError
        If ``param`` is not found in the model's coefficient names, or if the
        model has no cluster information attached.
    NotImplementedError
        If the model has more than one clustering variable (multi-way clustering
        is not yet supported).

    Notes
    -----
    The bootstrap uses the *impose-null* approach:

    1. Unrestricted OLS gives :math:`\\hat{\\beta}` and :math:`t_{obs}`.
    2. A restricted model (dropping *param*) gives residuals
       :math:`\\tilde{e}`.
    3. Bootstrap outcomes :math:`y^*_b = \\hat{y}_{restricted} +
       \\tilde{e} \\cdot w_g[\\text{cluster}(i)]`.
    4. Each :math:`y^*_b` is regressed on the **full** X to get
       :math:`t^*_b`.
    5. :math:`p = \\text{mean}(|t^*_b| \\geq |t_{obs}|)`.

    The implementation is fully vectorised across bootstrap replications using
    NumPy's broadcasting and ``einsum``.

    Examples
    --------
    >>> import pyfixest as pf
    >>> import pandas as pd, numpy as np
    >>> from pyfixest.inference import wildboottest
    >>> rng = np.random.default_rng(0)
    >>> N, G = 200, 20
    >>> cluster = np.repeat(np.arange(G), N // G)
    >>> x = rng.standard_normal(N)
    >>> y = 2 * x + rng.standard_normal(N)
    >>> df = pd.DataFrame({"y": y, "x": x, "cluster": cluster})
    >>> fit = pf.feols("y ~ x", data=df, vcov={"CRV1": "cluster"})
    >>> result = wildboottest(fit, param="x", B=499, seed=42)
    >>> result["p_value"] < 0.05
    True
    """
    # ------------------------------------------------------------------ #
    # 1. Extract information from the pyfixest model                      #
    # ------------------------------------------------------------------ #
    coefnames: list = list(model._coefnames)

    if param not in coefnames:
        raise ValueError(
            f"Parameter {param!r} not found in model coefficients: {coefnames}"
        )
    param_idx = coefnames.index(param)

    # Cluster variable check
    clustervar = getattr(model, "_clustervar", None)
    cluster_df = getattr(model, "_cluster_df", None)

    if not clustervar or cluster_df is None:
        raise ValueError(
            "Model has no cluster information. "
            "Re-estimate with vcov={'CRV1': 'varname'} to enable wildboottest."
        )
    if len(clustervar) > 1:
        raise NotImplementedError(
            "wildboottest currently supports single-way clustering only. "
            f"Model has clustering variables: {clustervar}"
        )

    cluster_ids = cluster_df.iloc[:, 0].to_numpy()  # (N,)

    # Design matrix and dependent variable (already demeaned if FE model)
    X: np.ndarray = np.asarray(model._X, dtype=float)  # (N, k)
    Y: np.ndarray = np.asarray(model._Y, dtype=float).ravel()  # (N,)

    # ------------------------------------------------------------------ #
    # 2. Delegate to pure-numpy implementation                            #
    # ------------------------------------------------------------------ #
    result = wildboottest_numpy(
        Y=Y,
        X=X,
        cluster_ids=cluster_ids,
        param_idx=param_idx,
        B=B,
        weights_type=weights_type,
        seed=seed,
    )

    # Replace param_idx with param name
    result.pop("param_idx")
    result["param"] = param

    return result
