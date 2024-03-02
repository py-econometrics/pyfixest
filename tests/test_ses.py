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
    adj_correction = np.sqrt((N - 1) / N)

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
    adj_correction = np.sqrt((N - 1) / N)

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
