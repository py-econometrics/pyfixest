from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

import pyfixest as pf
from pyfixest.estimation.post_estimation.wald import wald_test


def test_standalone_wald_test_matches_model_method():
    data = pf.get_data()
    method_fit = pf.feols("Y ~ X1 + X2 | f1", data)
    restriction = np.array([[1.0, -1.0]])

    expected = method_fit.wald_test(R=restriction, q=0.5, distribution="chi2")
    actual = wald_test(
        beta_hat=method_fit.coef().to_numpy(),
        vcov=method_fit._vcov,
        df_denom=method_fit._dfd,
        R=restriction,
        q=0.5,
        distribution="chi2",
    )

    pd.testing.assert_series_equal(
        pd.Series({"statistic": actual.statistic, "pvalue": actual.pvalue}), expected
    )
    assert actual.wald_statistic == method_fit._wald_statistic
    assert actual.f_statistic == method_fit._f_statistic
    assert actual.dfn == method_fit._dfn == 1


@pytest.mark.parametrize(
    "fml, options",
    [
        ("Y ~ X1 + X2", {}),
        ("Y ~ X1 + X2 | f1", {"weights": "weights", "store_data": False}),
        (
            "Y ~ X1 + X2 | f1",
            {"weights": "fweights", "weights_type": "fweights", "lean": True},
        ),
        ("Y ~ X1 + X2 | f1", {"vcov": {"CRV1": "f1+f2"}}),
    ],
)
def test_standalone_wald_test_preserves_default_f_statistics(fml, options):
    data = pf.get_data()
    data["fweights"] = np.arange(len(data)) % 3 + 1
    fit = pf.feols(fml, data, **options)
    k_fe = np.sum(fit._k_fe.values) if fit._has_fixef else 0
    df_denom = min(fit._G) - 1 if fit._is_clustered else fit._N - fit._k - k_fe

    result = wald_test(
        beta_hat=fit.coef().to_numpy(), vcov=fit._vcov, df_denom=df_denom
    )

    assert result.statistic == fit._f_statistic
    assert result.pvalue == fit._p_value
    assert result.dfn == fit._dfn
    assert df_denom == fit._dfd
    assert fit._wald_statistic == fit._f_statistic * fit._dfn
    pd.testing.assert_series_equal(
        fit.wald_test(),
        pd.Series({"statistic": result.statistic, "pvalue": result.pvalue}),
    )


def test_standalone_wald_test_with_readonly_numerical_inputs():
    beta_hat = np.array([1.0, 2.0])
    vcov = np.diag([2.0, 4.0])
    restriction = np.array([[1.0, -1.0]])
    null = np.array([0.5])
    for array in (beta_hat, vcov, restriction, null):
        array.flags.writeable = False

    with pytest.warns(UserWarning, match="Distribution changed to chi2"):
        result = wald_test(
            beta_hat=beta_hat, vcov=vcov, df_denom=20, R=restriction, q=null
        )

    # The restriction has estimate -1, null 0.5, and variance 2 + 4.
    expected_statistic = 1.5**2 / 6
    assert result.statistic == result.wald_statistic == expected_statistic
    assert result.f_statistic == expected_statistic
    assert result.dfn == 1
    assert result.pvalue == chi2.sf(expected_statistic, 1)
