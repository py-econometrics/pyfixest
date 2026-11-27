import numpy as np
import pandas as pd
import pytest

from pyfixest.estimation import feols, fepois
from pyfixest.utils.utils import get_data, ssc


@pytest.mark.parametrize("seed", [3212, 3213, 3214])
@pytest.mark.parametrize("N", [100, 400])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
@pytest.mark.parametrize("weights", [None, "weights"])
def test_HC1_vs_CRV1(N, seed, beta_type, error_type, weights):
    data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type).dropna()
    data["id"] = list(range(data.shape[0]))

    fit1 = feols(
        fml="Y~X1",
        data=data,
        vcov="HC1",
        ssc=ssc(k_adj=False, G_adj=False),
        weights=weights,
    )
    res_hc1 = fit1.tidy()

    fit2 = feols(
        fml="Y~X1",
        data=data,
        vcov={"CRV1": "id"},
        ssc=ssc(k_adj=False, G_adj=False),
        weights=weights,
    )
    res_crv1 = fit2.tidy()

    _N = fit1.sample_info.n_obs
    _k = fit1._k

    k_adj = False
    G_adj = False

    adj1 = _N / (_N - 1)
    adj2 = (_N - 1) / (_N - _k)
    adj3 = _N / (_N - _k)
    if k_adj and G_adj:
        adj_factor = adj3
    elif k_adj and not G_adj:
        adj_factor = adj2
    elif not k_adj and G_adj:
        adj_factor = adj1
    elif not k_adj and not G_adj:
        adj_factor = 1

    if not np.allclose(res_hc1["t value"] * np.sqrt(adj_factor), res_crv1["t value"]):
        raise ValueError("HC1 and CRV1 t values are not the same.")

    if not np.allclose(
        fit1.variance_covariance.vcov / adj_factor, fit2.variance_covariance.vcov
    ):
        raise ValueError("HC1 and CRV1 vcov are not the same.")


@pytest.mark.parametrize("seed", [3212, 3213, 3214])
@pytest.mark.parametrize("N", [100, 400])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
@pytest.mark.parametrize("weights", [None, "weights"])
def test_HC3_vs_CRV3(N, seed, beta_type, error_type, weights):
    data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type).dropna()
    data["id"] = list(range(data.shape[0]))

    fit1 = feols(
        fml="Y~X1",
        data=data,
        vcov="HC3",
        ssc=ssc(k_adj=False, G_adj=False),
        weights=weights,
    )
    res_hc3 = fit1.tidy()

    fit2 = feols(
        fml="Y~X1",
        data=data,
        vcov={"CRV3": "id"},
        ssc=ssc(k_adj=False, G_adj=False),
        weights=weights,
    )

    res_crv3 = fit1.tidy()

    _N = fit1.sample_info.n_obs
    _k = fit1._k

    k_adj = False
    G_adj = False

    adj1 = _N / (_N - 1)
    adj2 = (_N - 1) / (_N - _k)
    adj3 = _N / (_N - _k)
    if k_adj and G_adj:
        adj_factor = adj3
    elif k_adj and not G_adj:
        adj_factor = adj2
    elif not k_adj and G_adj:
        adj_factor = adj1
    elif not k_adj and not G_adj:
        adj_factor = 1

    if not np.allclose(res_hc3["t value"] * np.sqrt(adj_factor), res_crv3["t value"]):
        raise ValueError("HC3 and CRV3 t values are not the same.")

    if not np.allclose(
        fit1.variance_covariance.vcov / adj_factor, fit2.variance_covariance.vcov
    ):
        raise ValueError("HC1 and CRV1 vcov are not the same.")


@pytest.mark.extended
@pytest.mark.parametrize("seed", [3212])
@pytest.mark.parametrize("N", [100, 400])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
def test_CRV3_fixef(N, seed, beta_type, error_type):
    data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type).dropna()

    fit1 = feols(
        fml="Y~X1 + C(f2)",
        data=data,
        vcov={"CRV3": "f1"},
        ssc=ssc(k_adj=False, G_adj=False),
    )
    res_crv3a = fit1.tidy().reset_index().set_index("Coefficient").xs("X1")

    fit2 = feols(
        fml="Y~X1 | f2",
        data=data,
        vcov={"CRV3": "f1"},
        ssc=ssc(k_adj=False, G_adj=False),
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
        ssc=ssc(k_adj=False, G_adj=False),
        weights="weights",
        weights_type="aweights",
    )

    fit4 = feols(
        fml="Y~X1 |f2",
        data=data,
        vcov={"CRV3": "f1"},
        ssc=ssc(k_adj=False, G_adj=False),
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
        ssc=ssc(k_adj=False, G_adj=False),
        weights="count",
        weights_type="fweights",
    )
    fit6 = feols(
        fml="Y~X1 |f1",
        data=data2_w,
        vcov={"CRV3": "f1"},
        ssc=ssc(k_adj=False, G_adj=False),
        weights="count",
        weights_type="fweights",
    )

    res_crv3e = fit5.tidy().reset_index().set_index("Coefficient").xs("X1")
    res_crv3f = fit6.tidy()

    if not np.allclose(res_crv3e["Std. Error"], res_crv3f["Std. Error"]):
        raise ValueError("HC3 and CRV3 ses with fweights are not the same.")
    if not np.allclose(res_crv3e["t value"], res_crv3f["t value"]):
        raise ValueError("HC3 and CRV3 t values with fweights are not the same.")


@pytest.mark.extended
def run_crv3_poisson():
    data = get_data(N=1000, seed=1234, beta_type="1", error_type="1", model="Fepois")
    fit = fepois(
        fml="Y~X1 + C(f2)",
        data=data,
        vcov={"CRV3": "f1"},
        ssc=ssc(k_adj=False, G_adj=False),
    )

    fit = fepois(  # noqa: F841
        fml="Y~X1 |f1 + f2",
        data=data,
        vcov={"CRV3": "f1"},
        ssc=ssc(k_adj=False, G_adj=False),
    )


@pytest.mark.parametrize("vcov", ["NW", "DK"])
def test_hac_rejects_fweights(vcov):
    """HAC does not support frequency weights."""
    data = get_data().dropna().reset_index(drop=True)
    data["t"] = np.arange(len(data))
    data["unit"] = np.repeat(np.arange(len(data) // 10 + 1), 10)[: len(data)]
    kwargs = {"time_id": "t", "lag": 2}
    if vcov == "DK":
        kwargs["panel_id"] = "unit"

    with pytest.raises(NotImplementedError, match="fweights"):
        feols(
            "Y ~ X1",
            data=data,
            vcov=vcov,
            vcov_kwargs=kwargs,
            weights="weights",
            weights_type="fweights",
        )

    # aweights remain supported
    feols(
        "Y ~ X1",
        data=data,
        vcov=vcov,
        vcov_kwargs=kwargs,
        weights="weights",
        weights_type="aweights",
    )


@pytest.mark.parametrize("vcov", ["NW", "DK"])
@pytest.mark.parametrize("panel_dtype", ["object", "string", "category"])
def test_panel_hac_accepts_string_panel_id(vcov, panel_dtype):
    """Panel NW and DK accept string/object/categorical panel_id and match integer codes."""
    rng = np.random.default_rng(42)
    n_units, n_times = 12, 8
    unit_codes = np.repeat(np.arange(n_units), n_times)
    years = np.tile(np.arange(n_times), n_units)
    x = rng.normal(size=n_units * n_times)
    y = 1.0 + 2.0 * x + rng.normal(size=n_units * n_times)
    data = pd.DataFrame(
        {
            "Y": y,
            "X1": x,
            "year": years,
            "unit_int": unit_codes,
            "unit_label": pd.Series(
                [f"unit_{u}" for u in unit_codes], dtype=panel_dtype
            ),
        }
    )

    fit_int = feols(
        "Y ~ X1",
        data=data,
        vcov=vcov,
        vcov_kwargs={"time_id": "year", "panel_id": "unit_int", "lag": 2},
    )
    fit_str = feols(
        "Y ~ X1",
        data=data,
        vcov=vcov,
        vcov_kwargs={"time_id": "year", "panel_id": "unit_label", "lag": 2},
    )

    np.testing.assert_allclose(
        fit_str.se().to_numpy(), fit_int.se().to_numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(fit_str._vcov, fit_int._vcov, rtol=1e-12)

    # Post-estimation vcov() update with string panel_id also matches.
    fit_post = feols("Y ~ X1", data=data).vcov(
        vcov, vcov_kwargs={"time_id": "year", "panel_id": "unit_label", "lag": 2}
    )
    np.testing.assert_allclose(
        fit_post.se().to_numpy(), fit_int.se().to_numpy(), rtol=1e-12
    )

    # Duplicate (panel_id, time_id) pairs with string labels still raise ValueError.
    dup = data.copy()
    dup.loc[0, "year"] = dup.loc[1, "year"]
    with pytest.raises(
        ValueError,
        match=r"duplicate pairs for panel_id='unit_label' and time_id='year'",
    ):
        feols(
            "Y ~ X1",
            data=dup,
            vcov=vcov,
            vcov_kwargs={"time_id": "year", "panel_id": "unit_label", "lag": 2},
        )
