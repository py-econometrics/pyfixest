from pyfixest.estimation import feols, fepois
from pyfixest.utils import get_data, ssc
import polars as pl
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
