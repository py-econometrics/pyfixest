import pytest
import numpy as np
import pandas as pd
import pyfixest as pf
from pyfixest.utils import get_data

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

fixest = importr("fixest")
stats = importr('stats')

@pytest.fixture
def data():
    data = get_data(seed = 6574)
    data = data.dropna()
    data["X3"] = pd.Categorical(data.X3.astype(str))
    data["X4"] = pd.Categorical(data.X4.astype(str))

    return data


def test_internally(data):

    '''
    Test predict() method internally.
    Currently only for OLS.
    '''

    pyfixest = pf.Fixest(data = data).feols(fml = "Y~csw(X1, X2) | X3", vcov = 'iid')

    mod = pyfixest.fetch_model("0")
    mod.fixef()
    original_prediction = mod.predict()
    updated_prediction = mod.predict(data = mod.data)
    np.allclose(original_prediction, updated_prediction)

    # now expect error with updated predicted being a subset of data
    with pytest.raises(ValueError):
        updated_prediction = mod.predict(data = data.iloc[0:100, :])
        np.allclose(original_prediction, updated_prediction)


@pytest.mark.parametrize("fml", [

    "Y~ X1 | X3",
    "Y~ X1 | X3 + X4",
    #"Y~ X1 | X3^X4",
])

def test_vs_fixest(data, fml):

    '''
    Test predict and resid methods against fixest.
    '''

    pyfixest = pf.Fixest(data = data).feols(fml = fml, vcov = 'iid')

    mod = pyfixest.fetch_model("0")
    mod.fixef()

    # fixest estimation
    r_fixest = fixest.feols(
        ro.Formula(fml),
        data=data,
        ssc = fixest.ssc(True, "none", True, "min", "min", False)
    )

    # only if has fixef
    np.allclose(
        mod.sumFE,
        r_fixest.rx2("sumFE")
    )

    np.allclose(
        mod.predict().values,
        r_fixest.rx2("fitted.values")
    )

    np.allclose(
        mod.resid(),
        r_fixest.rx2("residuals")
    )



