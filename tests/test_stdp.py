import numpy as np
import statsmodels.api as sm

from pyfixest.estimation.estimation import feols
from pyfixest.utils.utils import get_data


def test_stdp():
    """Compare the standard error of the prediction to statsmodels.get_prediction()."""
    data = get_data().dropna()

    fit = feols("Y ~ X1 + X2", data=data)
    stdp = fit.predict(compute_stdp=True).stdp

    X = data[["X1","X2"]]
    Y = data["Y"]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    predictions = model.get_prediction()
    stdp_sm = predictions.var_pred_mean

    np.testing.assert_allclose(stdp, stdp_sm)
