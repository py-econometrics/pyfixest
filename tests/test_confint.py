import sys

import numpy as np
import pandas as pd
import pytest

from pyfixest.estimation import feols
from pyfixest.utils.utils import get_data


def test_confint():
    """Test the confint method of the feols class."""
    data = get_data()
    fit = feols("Y ~ X1 + X2 + C(f1)", data=data)
    confint = fit.confint()

    np.testing.assert_allclose(confint, fit.confint(alpha=0.05))
    assert np.all(confint.loc[:, "2.5%"] == fit.confint(alpha=0.05).loc[:, "2.5%"])
    assert np.all(confint.loc[:, "97.5%"] == fit.confint(alpha=0.05).loc[:, "97.5%"])
    assert np.all(confint.loc[:, "2.5%"] < fit.confint(alpha=0.10).loc[:, "5.0%"])
    assert np.all(confint.loc[:, "97.5%"] > fit.confint(alpha=0.10).loc[:, "95.0%"])

    # test keep, drop, and exact_match
    assert fit.confint(keep="X1", exact_match=True).shape[0] == 1
    assert (
        fit.confint(drop=["X2"], exact_match=True).shape[0] == len(fit._coefnames) - 1
    )
    assert fit.confint(keep="X").shape[0] == 2

    # simultaneous CIs: simultaneous CIs always wider
    for _ in range(5):
        assert np.all(
            confint.loc[:, "2.5%"] > fit.confint(alpha=0.05, joint=True).loc[:, "2.5%"]
        )
        assert np.all(
            confint.loc[:, "97.5%"]
            < fit.confint(alpha=0.05, joint=True).loc[:, "97.5%"]
        )

    # test seeds
    confint1 = fit.confint(joint=True, seed=1)
    confint2 = fit.confint(joint=True, seed=1)
    confint3 = fit.confint(joint=True, seed=2)

    assert np.all(confint1 == confint2)
    assert np.all(confint1 != confint3)


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="requires python3.11 or lower.")
def test_against_doubleml():
    """Test joint CIs against DoubleML."""
    import doubleml as dml
    from sklearn.base import clone
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(2002)
    n_obs = 5_000
    n_vars = 100
    X = rng.normal(size=(n_obs, n_vars))
    theta = np.array([3.0, 3.0, 3.0])
    y = np.dot(X[:, :3], theta) + rng.standard_normal(size=(n_obs,))

    dml_data = dml.DoubleMLData.from_arrays(X[:, 10:], y, X[:, :10])
    learner = LinearRegression()
    ml_l = clone(learner)
    ml_m = clone(learner)
    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m)
    dml_res = dml_plr.fit().bootstrap(n_rep_boot=10_000).confint(joint=True)

    df = pd.DataFrame(
        np.c_[y, X], columns=["y"] + ["X_" + str(x) for x in range(n_vars)]
    )
    m = feols(
        f"y ~ -1 + {'+'.join(['X_' + str(x) for x in range(n_vars)])}",
        df,
        vcov="hetero",
    )
    pyfixest_res = m.confint(keep="X_.$", reps=10_000, joint=True)

    assert np.all(np.abs(dml_res.values - pyfixest_res.values) < 1e-2)
