import numpy as np
import pandas as pd
from pyfixest.estimation import feols
from pyfixest.utils import get_data

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()
fixest = importr("fixest")

def test_multicol_overdetermined_iv():

    df_het = pd.read_csv("pyfixest/experimental/data/df_het.csv")

    fit = feols("Y ~ X2 +  f1| f1 | X1 ~ Z1 + Z2 ", data=data)
    fit_r = fixest.feols(
        formula="Y ~ X2 +  f1| f1 | X1 ~ Z1 + Z2 ",
        data=df_het,
        vcov="CRV1",
        ssc=False,
    )

    np.testing.assert_equal(fit._collin_vars, ["f1"])
    np.testing.assert_equal(fit._collin_vars_z, ["f1"])
    np.testing.assert_equal(fit._beta_hat, fit_r.rx2("coefficients"))
    np.testing.assert_equal(fit._se, fit_r.rx2("std.err"))


