import pytest
import numpy as np
import pyfixest as pf
from pyfixest.ssc_utils import ssc
from pyfixest.utils import get_data
import logging

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def data(seed = 3212, N = 1000, G = 25, beta_type = "1", error_type = "1"):
    df = get_data()
    df = df.dropna()
    df["id"] = list(range(df.shape[0]))
    return df

@pytest.mark.parametrize("seed", [3212, 3213, 3214])
@pytest.mark.parametrize("N", [100, 400])
@pytest.mark.parametrize("G", [25])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])

def test_HC1_vs_CRV1(N, G, seed, beta_type, error_type):

    data = get_data(N = N, G = G, seed = seed, beta_type = beta_type, error_type = error_type)
    data["id"] = list(range(data.shape[0]))

    fixest = pf.Fixest(data = data)
    fixest.feols('Y~X1', vcov = "HC1", ssc = ssc(adj = False, cluster_adj = False))
    res_hc1 = fixest.tidy()

    fixest.feols('Y~X1', vcov = {'CRV1':'id'}, ssc = ssc(adj = False, cluster_adj = False))
    res_crv1 = fixest.tidy()

    _, B = next(iter(fixest.all_fitted_models.items()))
    N = B.N

    # adj: default adjustments are different for HC3 and CRV3
    adj_correction = np.sqrt((N-1) / N)

    if not np.allclose(res_hc1["t value"] / adj_correction, res_crv1["t value"]):
        raise ValueError("HC1 and CRV1 t values are not the same.")

    #if not np.allclose(res_hc1["Pr(>|t|)"], res_crv1["Pr(>|t|)"]):
    #    raise ValueError("HC1 and CRV1 p values are not the same.")

def test_HC3_vs_CRV3(N, G, seed, beta_type, error_type):

    data = get_data(N = N, G = G, seed = seed, beta_type = beta_type, error_type = error_type)
    data["id"] = list(range(data.shape[0]))

    fixest = pf.Fixest(data = data)
    fixest.feols('Y~X1', vcov = "HC3", ssc = ssc(adj = False, cluster_adj = False))
    res_hc3 = fixest.tidy()

    fixest.vcov({'CRV3':'id'})
    res_crv3 = fixest.tidy()

    _, B = next(iter(fixest.all_fitted_models.items()))
    N = B.N

    # adj: default adjustments are different for HC3 and CRV3
    adj_correction = np.sqrt((N-1) / N)

    #if not np.allclose(np.sort(res_hc3["Std. Error"]) * adj_correction , np.sort(res_crv3["Std. Error"])):
    #    raise ValueError("HC3 and CRV3 ses are not the same.")
    if not np.allclose(res_hc3["t value"] / adj_correction, res_crv3["t value"]):
        raise ValueError("HC3 and CRV3 t values are not the same.")
    #if not np.allclose(res_hc3["Pr(>|t|)"], res_crv3["Pr(>|t|)"]):
    #    raise ValueError("HC3 and CRV3 p values are not the same.")

@pytest.mark.skip("HC3 not implemented for regressions with fixed effects.")
def test_HC3_vs_CRV3_fixef(N, G, seed, beta_type, error_type):

    data = get_data(N = N, G = G, seed = seed, beta_type = beta_type, error_type = error_type)
    data["id"] = list(range(data.shape[0]))


    fixest = pf.Fixest(data = data)
    fixest.feols('Y~X1 | X2', vcov = "HC3", ssc = ssc(adj = False, cluster_adj = False))
    res_hc3 = fixest.tidy()

    fixest.vcov({'CRV3':'id'})
    res_crv3 = fixest.tidy()

    _, B = next(iter(fixest.all_fitted_models.items()))
    N = B.N

    #if not np.allclose(res_hc3["Std. Error"] * adj_correction , res_crv3["Std. Error"]):
    #    raise ValueError("HC3 and CRV3 ses are not the same.")
    if not np.allclose(res_hc3["t value"] , res_crv3["t value"]):
        raise ValueError("HC3 and CRV3 t values are not the same.")
    #if not np.allclose(res_hc3["Pr(>|t|)"], res_crv3["Pr(>|t|)"]):
    #    raise ValueError("HC3 and CRV3 p values are not the same.")


def test_CRV3_fixef(data):

    fixest = pf.Fixest(data = data)
    fixest.feols('Y~X1 + C(X2)', vcov = {'CRV3':'id'}, ssc = ssc(adj = False, cluster_adj = False))
    res_crv3a = fixest.tidy().reset_index().set_index("coefnames").xs("X1")

    fixest2 = pf.Fixest(data = data)
    fixest2.feols('Y~X1 | X2', vcov = {'CRV3':'id'}, ssc = ssc(adj = False, cluster_adj = False))
    res_crv3b = fixest2.tidy()

    if not np.allclose(res_crv3a["Std. Error"] , res_crv3b["Std. Error"]):
        raise ValueError("HC3 and CRV3 ses are not the same.")
    if not np.allclose(res_crv3a["t value"], res_crv3b["t value"]):
        raise ValueError("HC3 and CRV3 t values are not the same.")
