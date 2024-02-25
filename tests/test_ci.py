import doubleml as dml
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from pyfixest.estimation import feols
from pyfixest.utils import get_data


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
    m = feols(f"y ~ -1 + {'+'.join(['X_'+str(x) for x in range(n_vars)])}", df)
    pyfixest_res = m.confint(keep="X_.$", nboot=10_000, joint=True)

    assert np.all(np.abs(dml_res.values - pyfixest_res.values) < 1e-2)
