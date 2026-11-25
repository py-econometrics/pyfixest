"""
Omnibus test for systematic treatment effect heterogeneity.

Implements the test of Ding, Feller, and Miratrix (2019), "Decomposing
Treatment Effect Variation," Journal of the American Statistical Association,
114, 304-317. The null hypothesis is that the conditional average treatment
effect does not vary linearly with the regressors.

This is a NumPy port of `dfmTest` in Netflix-Skunkworks/causaltransportr
(`R/dfmTest.R`, MIT license, Copyright 2022 Netflix, Inc.). The R source is
vendored under `tests/vendored/causaltransportr/`, and
`tests/test_dfm_test_vs_r.py` compares this port against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2


@dataclass(frozen=True, slots=True, kw_only=True)
class EffectHeterogeneityTest:
    """Omnibus test of systematic treatment effect heterogeneity.

    `Feols.dfm_heterogeneity_test()` returns one value. The test fits a
    separate regression in each treatment arm and jointly tests whether the
    differences between the arms' slopes are zero.

    Parameters
    ----------
    stat : float
        Chi-squared test statistic.
    pvalue : float
        P-value of `stat` under the chi-squared distribution with `df`
        degrees of freedom.
    df : int
        Degrees of freedom, the number of regressors tested for effect
        modification.

    Examples
    --------
    ```{python}
    import numpy as np
    import pandas as pd
    import pyfixest as pf

    rng = np.random.default_rng(42)
    n = 500
    X1 = rng.standard_normal(n)
    D = rng.binomial(1, 0.5, n)
    Y = 1.0 + 0.5 * X1 + D * (1.0 + 2.0 * X1) + rng.standard_normal(n)
    data = pd.DataFrame({"Y": Y, "D": D, "X1": X1})

    pf.feols("Y ~ D + X1", data=data).dfm_heterogeneity_test(treatment="D")
    ```
    """

    stat: float
    pvalue: float
    df: int


def _dfm_heterogeneity_test(
    *,
    y: NDArray[np.float64],
    treated: NDArray[np.bool_],
    X: NDArray[np.float64],
) -> EffectHeterogeneityTest:
    """Test whether the treatment effect varies linearly with `X`.

    Fits OLS in each treatment arm and takes the difference of the arms'
    coefficients, the linear projection of the conditional average treatment
    effect on `X`. The Wald test on the slope differences uses a sandwich
    variance (Ding, Feller, and Miratrix 2019).

    Parameters
    ----------
    y : NDArray[np.float64]
        Dependent variable, shape (n,).
    treated : NDArray[np.bool_]
        Treatment indicator, shape (n,).
    X : NDArray[np.float64]
        Regressors tested for effect modification, shape (n, p), without an
        intercept column; one is added here. Each arm needs at least p + 1
        observations.
    """
    X_full = np.column_stack([np.ones(X.shape[0]), X])
    K = X_full.shape[1]

    X1, X0 = X_full[treated], X_full[~treated]
    y1, y0 = y[treated], y[~treated]
    n1, n0 = X1.shape[0], X0.shape[0]

    beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
    beta0 = np.linalg.lstsq(X0, y0, rcond=None)[0]
    scores1 = (y1 - X1 @ beta1)[:, None] * X1
    scores0 = (y0 - X0 @ beta0)[:, None] * X0

    # Each arm contributes (X'X/n)^{-1} cov(scores)/n (X'X/n)^{-1}, as in the
    # reference `dfmTest`. np.cov uses ddof=1, like R's cov(), so the meat is
    # the HC0 meat scaled by n/(n-1) per arm. OLS scores with an intercept have
    # mean zero, so the centering in np.cov changes nothing.
    bread1 = np.linalg.inv(X1.T @ X1 / n1)
    bread0 = np.linalg.inv(X0.T @ X0 / n0)
    vcov_diff = (
        bread1 @ (np.cov(scores1, rowvar=False) / n1) @ bread1
        + bread0 @ (np.cov(scores0, rowvar=False) / n0) @ bread0
    )

    # The intercept difference is the average treatment effect, which is not
    # part of the null; test only the slope differences.
    slope_diff = (beta1 - beta0)[1:]
    stat = float(slope_diff @ np.linalg.solve(vcov_diff[1:, 1:], slope_diff))
    df = K - 1
    return EffectHeterogeneityTest(stat=stat, pvalue=float(chi2.sf(stat, df)), df=df)
