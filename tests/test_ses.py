import numpy as np
import pytest

from pyfixest.estimation.estimation import feols, fepois
from pyfixest.utils.utils import get_data, ssc


@pytest.mark.parametrize("seed", [3212, 3213, 3214])
@pytest.mark.parametrize("N", [100, 400])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
def test_HC1_vs_CRV1(N, seed, beta_type, error_type):
    data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type).dropna()
    data["id"] = list(range(data.shape[0]))

    fit1 = feols(
        fml="Y~X1", data=data, vcov="HC1", ssc=ssc(adj=False, cluster_adj=False)
    )
    res_hc1 = fit1.tidy()

    fit2 = feols(
        fml="Y~X1",
        data=data,
        vcov={"CRV1": "id"},
        ssc=ssc(adj=False, cluster_adj=False),
    )
    res_crv1 = fit2.tidy()

    N = fit1._N

    # adj: default adjustments are different for HC3 and CRV3
    adj_correction = 1

    if not np.allclose(res_hc1["t value"] / adj_correction, res_crv1["t value"]):
        raise ValueError("HC1 and CRV1 t values are not the same.")

    # if not np.allclose(res_hc1["Pr(>|t|)"], res_crv1["Pr(>|t|)"]):
    #    raise ValueError("HC1 and CRV1 p values are not the same.")


@pytest.mark.parametrize("seed", [3212, 3213, 3214])
@pytest.mark.parametrize("N", [100, 400])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
def test_HC3_vs_CRV3(N, seed, beta_type, error_type):
    data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type).dropna()
    data["id"] = list(range(data.shape[0]))

    fit1 = feols(
        fml="Y~X1", data=data, vcov="HC3", ssc=ssc(adj=False, cluster_adj=False)
    )
    res_hc3 = fit1.tidy()

    fit1.vcov({"CRV3": "id"})
    res_crv3 = fit1.tidy()

    N = fit1._N

    # adj: default adjustments are different for HC3 and CRV3
    adj_correction = 1

    # if not np.allclose(np.sort(res_hc3["Std. Error"]) * adj_correction , np.sort(res_crv3["Std. Error"])):  # noqa: W505
    #    raise ValueError("HC3 and CRV3 ses are not the same.")
    if not np.allclose(res_hc3["t value"] / adj_correction, res_crv3["t value"]):
        raise ValueError("HC3 and CRV3 t values are not the same.")
    # if not np.allclose(res_hc3["Pr(>|t|)"], res_crv3["Pr(>|t|)"]):
    #    raise ValueError("HC3 and CRV3 p values are not the same.")


@pytest.mark.parametrize("seed", [3212, 3213, 3214])
@pytest.mark.parametrize("N", [100, 400])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
def test_CRV3_fixef(N, seed, beta_type, error_type):
    data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type).dropna()

    fit1 = feols(
        fml="Y~X1 + C(f2)",
        data=data,
        vcov={"CRV3": "f1"},
        ssc=ssc(adj=False, cluster_adj=False),
    )
    res_crv3a = fit1.tidy().reset_index().set_index("Coefficient").xs("X1")

    fit2 = feols(
        fml="Y~X1 | f2",
        data=data,
        vcov={"CRV3": "f1"},
        ssc=ssc(adj=False, cluster_adj=False),
    )
    res_crv3b = fit2.tidy()

    if not np.allclose(res_crv3a["Std. Error"], res_crv3b["Std. Error"]):
        raise ValueError("HC3 and CRV3 ses are not the same.")
    if not np.allclose(res_crv3a["t value"], res_crv3b["t value"]):
        raise ValueError("HC3 and CRV3 t values are not the same.")

    # with weights:
    fit3 = feols(
        fml="Y~X1 + C(f2)",
        data=data,
        vcov={"CRV3": "f1"},
        ssc=ssc(adj=False, cluster_adj=False),
        weights="weights",
        weights_type="aweights",
    )

    fit4 = feols(
        fml="Y~X1 |f2",
        data=data,
        vcov={"CRV3": "f1"},
        ssc=ssc(adj=False, cluster_adj=False),
        weights="weights",
        weights_type="aweights",
    )

    res_crv3c = fit3.tidy().reset_index().set_index("Coefficient").xs("X1")
    res_crv3d = fit4.tidy()

    if not np.allclose(res_crv3c["Std. Error"], res_crv3d["Std. Error"]):
        raise ValueError("HC3 and CRV3 ses with aweights and weights are not the same.")
    if not np.allclose(res_crv3c["t value"], res_crv3d["t value"]):
        raise ValueError(
            "HC3 and CRV3 t values with aweights and weights are not the same."
        )

    # fweights
    data2_w = (
        data[["Y", "X1", "f1"]]
        .groupby(["Y", "X1", "f1"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )
    fit5 = feols(
        fml="Y~X1 + C(f1)",
        data=data2_w,
        vcov={"CRV3": "f1"},
        ssc=ssc(adj=False, cluster_adj=False),
        weights="count",
        weights_type="fweights",
    )
    fit6 = feols(
        fml="Y~X1 |f1",
        data=data2_w,
        vcov={"CRV3": "f1"},
        ssc=ssc(adj=False, cluster_adj=False),
        weights="count",
        weights_type="fweights",
    )

    res_crv3e = fit5.tidy().reset_index().set_index("Coefficient").xs("X1")
    res_crv3f = fit6.tidy()

    if not np.allclose(res_crv3e["Std. Error"], res_crv3f["Std. Error"]):
        raise ValueError("HC3 and CRV3 ses with fweights are not the same.")
    if not np.allclose(res_crv3e["t value"], res_crv3f["t value"]):
        raise ValueError("HC3 and CRV3 t values with fweights are not the same.")


def run_crv3_poisson():
    data = get_data(N=1000, seed=1234, beta_type="1", error_type="1", model="Fepois")
    fit = fepois(  # noqa: F841
        fml="Y~X1 + C(f2)",
        data=data,
        vcov={"CRV3": "f1"},
        ssc=ssc(adj=False, cluster_adj=False),
    )

    fit = fepois(  # noqa: F841
        fml="Y~X1 |f1 + f2",
        data=data,
        vcov={"CRV3": "f1"},
        ssc=ssc(adj=False, cluster_adj=False),
    )
