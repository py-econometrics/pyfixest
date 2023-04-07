import pytest
import numpy as np
import pyfixest as pf
from pyfixest.ssc_utils import ssc
from pyfixest.utils import get_data


@pytest.fixture
def data():
    data = get_data()
    data["id"] = list(range(data.shape[0]))
    return data


def test_HC1_vs_CRV1(data):

    fixest = pf.Fixest(data = data)
    fixest.feols('Y ~ X1', vcov = "HC1")
    res_hc1 = fixest.tidy()
    #fixest.tidy()["t value"]

    fixest.feols('Y ~ X1', vcov = {'CRV1':'id'}, ssc = ssc(cluster_adj = False))
    res_crv1 = fixest.tidy()

    if not np.allclose(res_hc1["Std. Error"], res_crv1["Std. Error"], rtol = 1e-12):
        raise ValueError("HC1 and CRV1 ses are not the same.")

    if not np.allclose(res_hc1["t value"], res_crv1["t value"], rtol = 1e-12):
        raise ValueError("HC1 and CRV1 t values are not the same.")

    if not np.allclose(res_hc1["Pr(>|t|)"], res_crv1["Pr(>|t|)"], rtol = 1e-12):
        raise ValueError("HC1 and CRV1 p values are not the same.")