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
    return get_data(seed = 6574)


def test_internally(data):

    data = data.dropna()

    data =  get_data()
    data = data.dropna()
    data["X3"] = pd.Categorical(data.X3)
    data["X4"] = pd.Categorical(data.X4)


    pyfixest = pf.Fixest(data = data).feols(fml = "Y~csw(X1, X2) | X3", vcov = 'iid')

    mod = pyfixest.fetch_model("0")
    mod.fixef()
    original_prediction = mod.predict()
    updated_prediction = mod.predict(data = mod.data)

    np.allclose(original_prediction, updated_prediction)

@pytest.mark.skip(reason="There is a bug in the test. Need to fix later. Error likely with rpy2 / how I use it.")
def test_vs_fixest(data):

    '''
    Test predict and resid methods against fixest.
    '''

    data = data.dropna()
    data["X3"] = pd.Categorical(data.X3)
    data["X4"] = pd.Categorical(data.X4)

    fml = "Y~ X1 | X3"

    pyfixest = pf.Fixest(data = data).feols(fml = fml, vcov = 'iid')

    mod = pyfixest.fetch_model("0")

    # fixest estimation
    r_fixest = fixest.feols(
        ro.Formula(fml),
        data=data,
        ssc = fixest.ssc(True, "none", True, "min", "min", False)
    )

    np.allclose(
        mod.predict().values,
        r_fixest.rx2("fitted_values")
    )
    np.allclose(
        mod.resid(),
        r_fixest.rx2("residuals")
    )



