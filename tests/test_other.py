import numpy as np
from pyfixest.estimation import feols
from pyfixest.utils import get_data

def test_multicol_overdetermined_iv():

    data = get_data(seed = 33102)
    fit = feols("Y ~ X2 +  f1| f1 | X1 ~ Z1 + Z2 ", data=data)

    np.testing.assert_equal(fit._collin_vars, ["f1"])
    np.testing.assert_equal(fit._collin_vars_z, ["f1"])
    np.testing.assert_equal(fit._beta_hat, np.array([0.568049, 0.358857]))
    np.testing.assert_equal(fit._se, np.array([ 0.091935, 0.012501]))


