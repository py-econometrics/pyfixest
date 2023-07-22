import numpy as np
import pandas as pd
import pyfixest as pf
from pyfixest.utils import get_data

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

fixest = importr("fixest")
stats = importr('stats')



def test_internally():

    data = get_data()
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


def test_vs_fixest():

    data = get_data(seed = 76)
    data = data.dropna()
    data["X3"] = pd.Categorical(data.X3)
    data["X4"] = pd.Categorical(data.X4)

    pyfixest = pf.Fixest(data = data).feols(fml = "Y~ X1 | X3", vcov = 'iid')

    mod = pyfixest.fetch_model("0")
    alpha_py = mod.fixef()

    # fixest estimation
    r_fixest = fixest.feols(
        ro.Formula("Y~ X1 | X3"),
        data=data,
        ssc = fixest.ssc(True, "none", True, "min", "min", False)
    )

    alpha_r = np.array(fixest.predict(r_fixest))




