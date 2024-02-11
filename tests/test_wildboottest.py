import numpy as np
import pytest

from pyfixest.estimation import feols
from pyfixest.utils import get_data, ssc


@pytest.fixture
def data():
    return get_data()


# note - tests currently fail because of ssc adjustments
def test_hc_equivalence(data):
    fixest = feols(fml="Y~X2", data=data, ssc=ssc(adj=False, cluster_adj=False))
    tstat = fixest.tstat()
    boot_tstat = fixest.wildboottest(param="X2", B=999, adj=False, cluster_adj=False)[
        "t value"
    ]

    np.allclose(tstat, boot_tstat)

    fixest = feols(fml="Y ~ X1 + X2 | f1 + f2", data=data, vcov="hetero")
    tstat = fixest.tstat()
    boot_tstat = fixest.wildboottest(param="X1", B=999, adj=False, cluster_adj=False)[
        "t value"
    ]

    np.allclose(tstat, boot_tstat)


def test_crv1_equivalence(data):
    fixest = feols(fml="Y~X1", data=data, vcov={"CRV1": "group_id"})
    tstat = fixest.tstat()
    boot_tstat = fixest.wildboottest(param="X1", B=999)["t value"]

    np.allclose(tstat, boot_tstat)

    fixest = feols(fml="Y ~ X1 + X2 | f1 + f2", data=data)
    tstat = fixest.tstat()
    boot_tstat = fixest.wildboottest(param="X1", B=999, adj=False, cluster_adj=False)[
        "t value"
    ]

    np.allclose(tstat, boot_tstat)

    fixest = feols(fml="Y ~ X1 | f1", vcov="hetero", data=data)
    boot_tstat = fixest.wildboottest(
        param="X1", cluster="f1", B=999, adj=False, cluster_adj=False
    )["t value"]
    tstat = fixest.vcov("hetero").tstat()

    np.allclose(tstat, boot_tstat)
