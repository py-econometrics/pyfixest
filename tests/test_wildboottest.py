import numpy as np
import pytest

import pyfixest as pf
from pyfixest.utils.utils import get_data, ssc


@pytest.fixture
def data():
    return get_data(N=2_000, seed=9)


# note - tests currently fail because of ssc adjustments
@pytest.mark.parametrize("fml", ["Y~X1", "Y~X1|f1", "Y~X1|f1+f2"])
def test_hc_equivalence(data, fml):
    ssc = pf.ssc(k_adj=False, G_adj=False)
    # note: cannot turn of ssc for wildboottest HC
    fixest = pf.feols(fml=fml, data=data, ssc=ssc, vcov="hetero")
    tstat = fixest.tstat().xs("X1")
    boot = fixest.wildboottest(param="X1", reps=999)
    boot_tstat = boot["t value"]
    ssc = boot["ssc"]

    # cannot test for for equality because of ssc adjustments
    np.testing.assert_allclose(tstat / boot_tstat, np.sqrt(ssc))


@pytest.mark.parametrize("fml", ["Y~X1", "Y~X1|f1", "Y~X1|f1+f2"])
def test_crv1_equivalence(data, fml):
    fixest = pf.feols(
        fml, data=data, vcov={"CRV1": "group_id"}, ssc=ssc(k_adj=False, G_adj=False)
    )
    tstat = fixest.tstat().xs("X1")
    boot_tstat = fixest.wildboottest(param="X1", reps=999, k_adj=False, G_adj=False)[
        "t value"
    ]

    np.testing.assert_allclose(tstat, boot_tstat)


def test_collinear_covariate_is_dropped():
    """wildboottest() uses the fitted columns, not the collinear covariate."""
    data = get_data().dropna().reset_index(drop=True)
    data["Xc"] = data.groupby("f1")["X2"].transform("mean")
    fit = pf.feols("Y ~ X1 + Xc | f1", data=data)
    ref = pf.feols("Y ~ X1 | f1", data=data)

    _, X, xnames = fit._model_matrix_one_hot()
    assert "Xc" not in xnames
    assert np.linalg.matrix_rank(X) == X.shape[1] == len(xnames)

    boot = fit.wildboottest(param="X1", reps=999, seed=3)
    boot_ref = ref.wildboottest(param="X1", reps=999, seed=3)
    np.testing.assert_allclose(boot["t value"], boot_ref["t value"])
    np.testing.assert_allclose(boot["Pr(>|t|)"], boot_ref["Pr(>|t|)"])
