import numpy as np
import pandas as pd
import pytest

import pyfixest as pf


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(12345)
    n = 1200
    df = pd.DataFrame(
        {
            "Y": rng.normal(0, 1, n),
            "X1": rng.normal(0, 1, n),
            "X2": rng.normal(0, 1, n),
            "f1": rng.choice(["a", "b", "c", "d"], size=n),
            "f2": rng.choice(["u", "v"], size=n),
        }
    )
    return df


@pytest.mark.parametrize("k_adj", [True, False])
@pytest.mark.parametrize("k_fixef", ["none", "full", "nonnested"])
@pytest.mark.parametrize("G_adj", [True, False])
@pytest.mark.parametrize("G_df", ["min", "conventional"])
def test_ssc_backward_compatibility_feols(data, k_adj, k_fixef, G_adj, G_df):
    ssc_new = pf.ssc(k_adj=k_adj, k_fixef=k_fixef, G_adj=G_adj, G_df=G_df)

    with pytest.warns(DeprecationWarning) as rec:
        ssc_old = pf.ssc(adj=k_adj, fixef_k=k_fixef, cluster_adj=G_adj, cluster_df=G_df)

    assert len(rec) == 4
    assert any("adj" in str(w.message) for w in rec)
    assert any("fixef_k" in str(w.message) for w in rec)
    assert any("cluster_adj" in str(w.message) for w in rec)
    assert any("cluster_df" in str(w.message) for w in rec)

    fit_new = pf.feols("Y ~ X1", data=data, vcov={"CRV1": "f1"}, ssc=ssc_new)
    fit_old = pf.feols("Y ~ X1", data=data, vcov={"CRV1": "f1"}, ssc=ssc_old)

    np.testing.assert_allclose(
        fit_new.coef(), fit_old.coef(), rtol=0, atol=0, err_msg="coef differ"
    )
    np.testing.assert_allclose(
        fit_new.se(), fit_old.se(), rtol=0, atol=0, err_msg="se differ"
    )
    np.testing.assert_allclose(
        fit_new.pvalue(), fit_old.pvalue(), rtol=0, atol=0, err_msg="pvalues differ"
    )
    np.testing.assert_allclose(
        fit_new.tstat(), fit_old.tstat(), rtol=0, atol=0, err_msg="tstats differ"
    )

    ci_new = fit_new.confint().values
    ci_old = fit_old.confint().values
    np.testing.assert_allclose(ci_new, ci_old, rtol=0, atol=0, err_msg="confint differ")

    assert fit_new._df_t == fit_old._df_t
    assert fit_new._df_k == fit_old._df_k
    np.testing.assert_allclose(
        fit_new._vcov, fit_old._vcov, rtol=0, atol=0, err_msg="vcov differ"
    )

    assert fit_new._ssc == fit_old._ssc
