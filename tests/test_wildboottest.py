import pytest
from pyfixest.estimation import feols
from pyfixest.utils import get_data


@pytest.fixture
def data():
    return get_data()


# note - tests currently fail because of ssc adjustments
def test_hc_equivalence(data):
    fixest = feols(fml="Y~csw(X1, X2, f1)", data=data)
    tstat = fixest.tstat().reset_index().set_index("Coefficient").xs("X1")
    boot_tstat = fixest.wildboottest(param="X1", B=999)["statistic"]

    # np.allclose(tstat, boot_tstat)


def test_crv1_equivalence(data):
    fixest = feols(fml="Y~csw(X1, X2, f1)", data=data, vcov={"CRV1": "group_id"})
    tstat = fixest.tstat().reset_index().set_index("Coefficient").xs("X1")
    boot_tstat = fixest.wildboottest(param="X1", B=999)["statistic"]

    # np.allclose(tstat, boot_tstat)
