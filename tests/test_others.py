import numpy as np
import pandas as pd
import polars as pl

from pyfixest.estimation.estimation import feols, fepois
from pyfixest.utils.utils import get_data, ssc


def test_multicol_overdetermined_iv():
    data = get_data()
    fit = feols("Y ~ X2 +  f1| f1 | X1 ~ Z1 + Z2", data=data, ssc=ssc(adj=False))

    assert fit._collin_vars == ["f1"]
    assert fit._collin_vars_z == ["f1"]

    np.testing.assert_allclose(
        fit._beta_hat, np.array([-0.993607, -0.174227], dtype=float), rtol=1e-5
    )
    np.testing.assert_allclose(fit._se, np.array([0.104009, 0.018416]), rtol=1e-5)


def test_polars_input():
    data = get_data()
    data_pl = pl.from_pandas(data)
    fit = feols("Y ~ X1", data=data)
    fit.predict(newdata=data_pl)

    data = get_data(model="Fepois")
    data_pl = pl.from_pandas(data)
    fit = fepois("Y ~ X1", data=data_pl)


def test_integer_XY():
    # Create a random number generator
    rng = np.random.default_rng()

    N = 1000
    X = rng.normal(0, 1, N)
    f = rng.choice([0, 1], N)
    Y = 2 * X + rng.normal(0, 1, N) + f * 2
    Y = np.round(Y).astype(np.int64)
    X = np.round(X).astype(np.int64)

    df = pd.DataFrame({"Y": Y, "X": X, "f": f})

    fit1 = feols("Y ~ X | f", data=df, vcov="iid")
    fit2 = feols("Y ~ X + C(f)", data=df)

    np.testing.assert_allclose(fit1.coef().xs("X"), fit2.coef().xs("X"))


def test_coef_update():
    data = get_data()
    data_subsample = data.sample(frac=0.5)
    m = feols("Y ~ X1 + X2", data=data_subsample)
    new_points_id = np.random.choice(
        list(set(data.index) - set(data_subsample.index)), 5
    )
    X_new, y_new = (
        np.c_[
            np.ones(len(new_points_id)), data.loc[new_points_id][["X1", "X2"]].values
        ],
        data.loc[new_points_id]["Y"].values,
    )
    updated_coefs = m.update(X_new, y_new)
    full_coefs = (
        feols(
            "Y ~ X1 + X2",
            data=data.loc[data_subsample.index.append(pd.Index(new_points_id))],
        )
        .coef()
        .values
    )

    np.testing.assert_allclose(updated_coefs, full_coefs)

def test_coef_update_inplace():
    data = get_data()
    data_subsample = data.sample(frac=0.3)
    m = feols("Y ~ X1 + X2", data=data_subsample)
    new_points_id = np.random.choice(
        list(set(data.index) - set(data_subsample.index)), 5
    )
    X_new, y_new = (
        np.c_[
            data.loc[new_points_id][["X1", "X2"]].values # only pass columns; let `update` add the intercept
        ],
        data.loc[new_points_id]["Y"].values,
    )
    m.update(X_new, y_new, inplace=True)
    full_coefs = (
        feols(
            "Y ~ X1 + X2",
            data=data.loc[data_subsample.index.append(pd.Index(new_points_id))],
        )
        .coef()
        .values
    )
    np.testing.assert_allclose(m.coef().values, full_coefs)
