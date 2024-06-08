import numpy as np
import pandas as pd

from pyfixest.estimation.estimation import feols
from pyfixest.utils.utils import get_data
from tests.py_test_comparisons import _get_data_doubleml_test


def test_confint():
    """Test the confint method of the feols class."""
    data = get_data()
    fit = feols("Y ~ X1 + X2 + C(f1)", data=data)
    confint = fit.confint()

    np.testing.assert_allclose(confint, fit.confint(alpha=0.05))
    assert np.all(confint.loc[:, "0.025%"] == fit.confint(alpha=0.05).loc[:, "0.025%"])
    assert np.all(confint.loc[:, "0.975%"] == fit.confint(alpha=0.05).loc[:, "0.975%"])
    assert np.all(confint.loc[:, "0.025%"] < fit.confint(alpha=0.10).loc[:, "0.05%"])
    assert np.all(confint.loc[:, "0.975%"] > fit.confint(alpha=0.10).loc[:, "0.95%"])

    # test keep, drop, and exact_match
    assert fit.confint(keep="X1", exact_match=True).shape[0] == 1
    assert (
        fit.confint(drop=["X2"], exact_match=True).shape[0] == len(fit._coefnames) - 1
    )
    assert fit.confint(keep="X").shape[0] == 2

    # simultaneous CIs: simultaneous CIs always wider
    for _ in range(5):
        assert np.all(
            confint.loc[:, "0.025%"]
            > fit.confint(alpha=0.05, joint=True).loc[:, "0.025%"]
        )
        assert np.all(
            confint.loc[:, "0.975%"]
            < fit.confint(alpha=0.05, joint=True).loc[:, "0.975%"]
        )

    # test seeds
    confint1 = fit.confint(joint=True, seed=1)
    confint2 = fit.confint(joint=True, seed=1)
    confint3 = fit.confint(joint=True, seed=2)

    assert np.all(confint1 == confint2)
    assert np.all(confint1 != confint3)


def test_against_doubleml():
    """Test joint CIs against DoubleML."""
    _, _, df, n_vars = _get_data_doubleml_test()
    dml_res = pd.read_csv("tests/data/dml_res.csv")

    m = feols(
        f"y ~ -1 + {'+'.join(['X_'+str(x) for x in range(n_vars)])}", df, vcov="hetero"
    )
    pyfixest_res = m.confint(keep="X_.$", nboot=10_000, joint=True)

    assert np.all(np.abs(dml_res.iloc[:, 1:].values - pyfixest_res.values) < 1e-2)
