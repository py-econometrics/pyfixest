import pytest
import numpy as np
import pandas as pd
from pyfixest.estimation import feols
from pyfixest.utils import ssc

# rpy2 imports
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

pandas2ri.activate()
clubSandwich = importr("clubSandwich")
stats = importr("stats")
base = importr("base")


@pytest.mark.parametrize(
    "R",
    [
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([-1, 2]),
        np.eye(2),
        # np.eye(2) * 2,
    ],
)
@pytest.mark.skip("Wald tests will be released with pyfixest 0.14.0.")
def test_wald_test(R):
    data = pd.read_csv("pyfixest/did/data/df_het.csv")
    data = data.iloc[1:3000]
    fml = "dep_var ~ treat"
    fit = feols(fml, data, vcov={"CRV1": "year"}, ssc=ssc(adj=False))
    _k = fit._k

    # Wald test
    fit.wald_test(R=R)
    f_stat = fit._f_statistic

    # Compare with R
    r_fit = stats.lm(fml, data=data)
    r_wald = clubSandwich.Wald_test(
        r_fit, constraints=base.matrix(R, 1, 2), vcov="CR1", cluster=data["year"]
    )
    r_fstat = pd.DataFrame(r_wald).T[1]

    np.testing.assert_allclose(f_stat, r_fstat[0], rtol=1e-02, atol=1e-02)
