"""
Omnibus test for systematic treatment effect heterogeneity.

Implements the Ding, Feller, Miratrix (2019) decomposition test.
The null is that the CATE does not vary linearly with covariates X.

Reference
---------
Ding, P., A. Feller, and L. Miratrix (2019): "Decomposing Treatment Effect
Variation," Journal of the American Statistical Association, 114, 304-317.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def dfm_heterogeneity_test(
    y: np.ndarray,
    treatment: np.ndarray,
    X: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """
    Omnibus chi-squared test for systematic treatment effect heterogeneity.

    Fits separate OLS in each treatment arm, takes the coefficient
    difference (this is the linear projection of the CATE onto X),
    and jointly tests whether the slope differences are zero using a
    sandwich variance estimator.

    Parameters
    ----------
    y : np.ndarray
        Outcome vector, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator, shape (n,), values 0 or 1.
    X : np.ndarray
        Covariate matrix, shape (n, p). Do not include an intercept
        column here, one is added internally.

    Returns
    -------
    dict with keys:
        - "statistic": chi-squared test statistic
        - "pvalue": p-value from chi2(K-1) distribution
        - "df": degrees of freedom (number of covariates)
        - "beta_hat": coefficient difference vector (intercept + slopes)
        - "cov_beta": sandwich covariance matrix of beta_hat
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    treatment = np.asarray(treatment).ravel()
    X = np.asarray(X, dtype=np.float64)

    n = len(y)
    if X.ndim == 1:
        X = X.reshape(n, 1)

    if len(treatment) != n:
        raise ValueError("y and treatment must have the same length.")
    if X.shape[0] != n:
        raise ValueError("X must have the same number of rows as y.")

    unique_vals = np.unique(treatment)
    if not (len(unique_vals) == 2 and set(unique_vals) <= {0, 1}):
        raise ValueError("treatment must be binary with values 0 and 1.")

    # Add intercept column
    X_full = np.column_stack([np.ones(n), X])
    K = X_full.shape[1]

    if K < 2:
        raise ValueError("Need at least one covariate beyond the intercept.")

    # Split into treated and control
    mask1 = treatment == 1
    mask0 = treatment == 0
    X1, X0 = X_full[mask1], X_full[mask0]
    y1, y0 = y[mask1], y[mask0]
    n1, n0 = X1.shape[0], X0.shape[0]

    if n1 < K or n0 < K:
        raise ValueError(
            f"Not enough observations per arm. Need at least K={K}, "
            f"got n1={n1}, n0={n0}."
        )

    # OLS in each arm separately
    beta1, _, _, _ = np.linalg.lstsq(X1, y1, rcond=None)
    beta0, _, _, _ = np.linalg.lstsq(X0, y0, rcond=None)

    resid1 = y1 - X1 @ beta1
    resid0 = y0 - X0 @ beta0

    # Score matrices (residual * covariate row). These are the
    # building blocks for the HC0-style sandwich variance.
    E1 = resid1[:, None] * X1
    E0 = resid0[:, None] * X0

    # Sandwich variance of the coefficient difference.
    # Each arm contributes: (X'X/n)^{-1} * Cov(scores)/n * (X'X/n)^{-1}
    # np.cov uses ddof=1 by default, same as R's cov().
    Sxx1_inv = np.linalg.inv(X1.T @ X1 / n1)
    Sxx0_inv = np.linalg.inv(X0.T @ X0 / n0)

    cov_beta = (
        Sxx1_inv @ (np.cov(E1, rowvar=False) / n1) @ Sxx1_inv
        + Sxx0_inv @ (np.cov(E0, rowvar=False) / n0) @ Sxx0_inv
    )

    # The coefficient difference is the projection of CATE onto X
    beta_hat = beta1 - beta0

    # We only test the slope coefficients (drop the intercept).
    # The intercept difference is just the average treatment effect,
    # which we don't want to test here.
    beta_slopes = beta_hat[1:]
    cov_slopes = cov_beta[1:, 1:]

    # Wald-type chi-squared: beta' * V^{-1} * beta
    stat = float(beta_slopes @ np.linalg.solve(cov_slopes, beta_slopes))
    df = K - 1
    pvalue = float(chi2.sf(stat, df=df))

    return {
        "statistic": stat,
        "pvalue": pvalue,
        "df": df,
        "beta_hat": beta_hat,
        "cov_beta": cov_beta,
    }
