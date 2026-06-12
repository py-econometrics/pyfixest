"""Wald tests for linear hypotheses R @ beta = q."""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, f


def _wald_test_impl(model, R=None, q=None, distribution="F") -> pd.Series:
    "Implementation of Feols.wald_test; see the method docstring for details."
    k_fe = np.sum(model._k_fe.values) if model._has_fixef else 0

    # If R is None, default to the identity matrix
    if R is None:
        R = np.eye(model._k)

    # Ensure R is two-dimensional
    if R.ndim == 1:
        R = R.reshape((1, len(R)))

    if R.shape[1] != model._k:
        raise ValueError(
            "The number of columns of R must be equal to the number of coefficients."
        )

    # If q is None, default to a vector of zeros
    if q is None:
        q = np.zeros(R.shape[0])
    else:
        if not isinstance(q, (int, float, np.ndarray)):
            raise ValueError("q must be a numeric scalar or array.")
        if isinstance(q, np.ndarray):
            if q.ndim != 1:
                raise ValueError("q must be a one-dimensional array or a scalar.")
            if q.shape[0] != R.shape[0]:
                raise ValueError("q must have the same number of rows as R.")

    n_restriction = R.shape[0]
    model._dfn = n_restriction

    if model._is_clustered:
        model._dfd = np.min(np.array(model._G)) - 1
    else:
        model._dfd = model._N - model._k - k_fe

    bread = R @ model._beta_hat - q
    meat = np.linalg.pinv(R @ model._vcov @ R.T)
    W = bread.T @ meat @ bread
    model._wald_statistic = W

    # Check if distribution is "F" and R is not identity matrix
    # or q is not zero vector
    if distribution == "F" and (
        not np.array_equal(R, np.eye(model._k)) or not np.all(q == 0)
    ):
        warnings.warn(
            "Distribution changed to chi2, as R is not an identity matrix and q is not a zero vector."
        )
        distribution = "chi2"

    if distribution == "F":
        model._f_statistic = W / model._dfn
        model._p_value = 1 - f.cdf(model._f_statistic, dfn=model._dfn, dfd=model._dfd)
        res = pd.Series({"statistic": model._f_statistic, "pvalue": model._p_value})
    elif distribution == "chi2":
        model._f_statistic = W / model._dfn
        model._p_value = chi2.sf(model._wald_statistic, model._dfn)
        res = pd.Series({"statistic": model._wald_statistic, "pvalue": model._p_value})
    else:
        raise ValueError("Distribution must be F or chi2")

    return res
