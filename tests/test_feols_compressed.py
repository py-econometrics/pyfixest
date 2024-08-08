import numpy as np
import pytest

import pyfixest as pf

ATOL = 1e-6


@pytest.fixture
def data():
    return pf.get_data()


@pytest.mark.parametrize("fml", ["Y ~ f1", "Y ~ f1 + f2", "Y2 ~ f1 + f2 + f3"])
@pytest.mark.parametrize("vcov", ["hetero"])
# @pytest.mark.parametrize("ssc", [pf.ssc(adj=True, cluster_adj=True), pf.ssc(adj=False, cluster_adj=False)])
@pytest.mark.parametrize("dropna", [True])
def test_feols_compressed(data, fml, vcov, dropna):
    fit = pf.feols(
        fml=fml,
        data=data.dropna() if dropna else data,
        vcov=vcov,
    )

    fit_c = pf.feols_c(
        fml=fml,
        data=data.dropna() if dropna else data,
        vcov=vcov,
    )

    assert np.all(fit.coef().xs("f1") - fit_c.coef().xs("f1") < ATOL)
    assert np.all(fit.se().xs("f1") - fit_c.se().xs("f1") < ATOL)
    assert np.all(fit.pvalue().xs("f1") - fit_c.pvalue().xs("f1") < ATOL)
