import numpy as np
import pandas as pd

import pyfixest as pf
from pyfixest.estimation.post_estimation.wald import wald_test


def test_standalone_wald_test_matches_model_method():
    data = pf.get_data()
    method_fit = pf.feols("Y ~ X1 + X2 | f1", data)
    standalone_fit = pf.feols("Y ~ X1 + X2 | f1", data)
    restriction = np.array([[1.0, -1.0]])

    expected = method_fit.wald_test(R=restriction, q=0.5, distribution="chi2")
    actual = wald_test(
        standalone_fit,
        R=restriction,
        q=0.5,
        distribution="chi2",
    )

    pd.testing.assert_series_equal(actual, expected)
    assert standalone_fit._wald_statistic == expected["statistic"]


def test_standalone_wald_test_preserves_default_f_statistics():
    fit = pf.feols("Y ~ X1 + X2", pf.get_data())

    result = wald_test(fit)

    assert result["statistic"] == fit._f_statistic
    assert result["pvalue"] == fit._p_value
    assert fit._wald_statistic == fit._f_statistic * fit._dfn
