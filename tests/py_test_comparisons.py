import doubleml as dml
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression


def _get_data_doubleml_test():
    rng = np.random.default_rng(2002)
    n_obs = 5_000
    n_vars = 100
    X = rng.normal(size=(n_obs, n_vars))
    theta = np.array([3.0, 3.0, 3.0])
    y = np.dot(X[:, :3], theta) + rng.standard_normal(size=(n_obs,))

    df = pd.DataFrame(
        np.c_[y, X], columns=["y"] + ["X_" + str(x) for x in range(n_vars)]
    )

    return y, X, df, n_vars


def test_against_doubleml():
    """Test joint CIs against DoubleML."""
    y, X, _, _ = _get_data_doubleml_test()

    dml_data = dml.DoubleMLData.from_arrays(X[:, 10:], y, X[:, :10])
    learner = LinearRegression()
    ml_l = clone(learner)
    ml_m = clone(learner)
    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m)
    dml_res = dml_plr.fit().bootstrap(n_rep_boot=10_000).confint(joint=True)
    dml_res.to_csv("tests/data/dml_res.csv")


def main():
    test_against_doubleml()


if __name__ == "__main__":
    main()
