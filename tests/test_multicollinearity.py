import numpy as np
import pandas as pd

from pyfixest.estimation import feols


def test_multicollinearity_error():
    rng = np.random.default_rng(4)

    N = 10000
    X1 = rng.normal(0, 1, N)
    Y = rng.normal(0, 1, N)
    f1 = rng.choice([0, 1, 2, 3, 4], N, True)
    f2 = f1.copy()
    f3 = f1.copy()
    f3 = np.where(f3 == 1, 0, f3)

    data = pd.DataFrame(
        {
            "Y": Y,
            "X1": X1,
            "X2": X1 + rng.normal(0, 0.00000000001, N),
            "f1": f1,
            "f2": f2,
            "f3": f3,
        }
    )

    fit = feols("Y ~ X1 + X2", data=data)
    assert fit._coefnames == ["Intercept", "X1"]

    fit = feols("Y ~ X1 + f1 + X2 + f2", data=data)
    assert fit._coefnames == ["Intercept", "X1", "f1"]

    fit = feols("Y ~ X1 + f1 + X2 | f2", data=data)
    assert fit._coefnames == ["X1"]
