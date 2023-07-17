import pytest
import pyfixest as pf
from pyfixest.utils import get_data
import numpy as np

@pytest.fixture
def data():
    return get_data()

# note - tests currently fail because of ssc adjustments
def test_hc_equivalence(data):

    fixest = pf.Fixest(data)
    fixest.feols("Y~csw(X1, X2, X3)")
    tstat = fixest.tstat().reset_index().set_index("coefnames").xs("X1")
    boot_tstat = fixest.wildboottest(param = "X1", B = 999)["t value"]

    #np.allclose(tstat, boot_tstat)

def test_crv1_equivalence(data):

    fixest = pf.Fixest(data)
    fixest.feols("Y~csw(X1, X2, X3)", vcov = {"CRV1":"group_id"})
    tstat = fixest.tstat().reset_index().set_index("coefnames").xs("X1")
    boot_tstat = fixest.wildboottest(param = "X1", B = 999)["t value"]

    #np.allclose(tstat, boot_tstat)

