"""Coefficient-level inference from a covariance estimate."""

from __future__ import annotations

import numpy as np

from pyfixest.estimation.internals.families import InferenceDist
from pyfixest.estimation.internals.model_state import CoefficientTable


def coefficient_table(
    *,
    beta_hat: np.ndarray,
    vcov: np.ndarray,
    df_t: int | float,
    dist: InferenceDist,
    alpha: float,
) -> CoefficientTable:
    """Build the coefficient table from estimates and their covariance.

    Follows fixest's ``fixest_CI_factor``: the bounds are
    ``beta +- q(1 - alpha / 2) * se`` with the quantile of ``dist`` at
    ``df_t`` degrees of freedom.
    """
    se = np.sqrt(np.diagonal(vcov))
    tstat = beta_hat / se
    pvalue = dist.pvalue(tstat, df_t)
    z_se = dist.crit_val(alpha, df_t) * se
    return CoefficientTable(
        estimate=beta_hat,
        se=se,
        tstat=tstat,
        pvalue=pvalue,
        conf_int=np.array([beta_hat - z_se, beta_hat + z_se]),
        alpha=alpha,
    )
