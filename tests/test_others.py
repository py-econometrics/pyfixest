from pyfixest.estimation import feols, fepois
from pyfixest.utils import get_data, ssc
import polars as pl
import pandas as pd
import numpy as np


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
