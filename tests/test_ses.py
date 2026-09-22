import numpy as np
import pandas as pd
import pytest

from pyfixest.errors import MissingModelDataError
from pyfixest.estimation import feols, fepois
from pyfixest.utils.utils import get_data, ssc


@pytest.mark.parametrize(
    "model,fml", [(feols, "y ~ x"), (feols, "y ~ x | fe"), (fepois, "y ~ x | fe")]
)
@pytest.mark.parametrize("vcov_type", ["CRV1", "CRV3"])
def test_multiway_cluster_intersections(model, fml, vcov_type):
    """Exercise the CRV loop with colliding string labels and refit paths."""
    rng = np.random.default_rng(221)
    groups = np.tile(np.indices((2, 3, 4)).reshape(3, -1), 8)
    data = pd.DataFrame(
        {
            "a": np.array(["a-b", "a"])[groups[0]],
            "b": np.array(["c", "b-c", "d"])[groups[1]],
            "c": groups[2],
            "fe": rng.integers(2, size=groups.shape[1]),
            "x": rng.normal(size=groups.shape[1]),
            "y": rng.poisson(4, size=groups.shape[1]),
        }
    )
    original = data.copy(deep=True)
    options = ssc(k_adj=False, G_adj=False)
    fit = model(fml, data, vcov={vcov_type: "a+b+c"}, ssc=options)
    expected = np.zeros_like(fit.variance_covariance.vcov)
    # Explicit seven-term identity, independently encoded with tuple labels.
    for columns, sign in [
        ("a", 1),
        ("b", 1),
        ("c", 1),
        ("ab", -1),
        ("ac", -1),
        ("bc", -1),
        ("abc", 1),
    ]:
        reference_data = data.assign(
            group=list(zip(*(data[c] for c in columns), strict=True))
        )
        reference = model(fml, reference_data, vcov={vcov_type: "group"}, ssc=options)
        expected += sign * reference.variance_covariance.vcov
    np.testing.assert_allclose(
        fit.variance_covariance.vcov,
        expected,
        rtol=1e-10,
        atol=1e-12,
        err_msg="multiway inclusion-exclusion covariance",
    )
    if vcov_type == "CRV1":
        bread = fit.sandwich.bread
        np.testing.assert_allclose(
            expected,
            bread @ fit.variance_covariance.meat @ bread,
            rtol=1e-10,
            atol=1e-12,
            err_msg="multiway sandwich meat",
        )
    fit.vcov({vcov_type: "c+b+a"})
    np.testing.assert_allclose(
        fit.variance_covariance.vcov,
        expected,
        rtol=1e-10,
        atol=1e-12,
        err_msg="cluster ordering",
    )
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("storage", [{}, {"lean": True}, {"store_data": False}])
def test_multiway_cluster_multiple_estimation_storage(storage):
    data = get_data(N=500, seed=9289).dropna()
    vcov = {"CRV1": "f1+f2+group_id"}
    fits = feols("Y ~ sw(X1, X2) | f1", data, vcov=vcov, **storage).to_list()
    for fit, regressor in zip(fits, ["X1", "X2"], strict=True):
        reference = feols(f"Y ~ {regressor} | f1", data).vcov(vcov)
        np.testing.assert_allclose(
            fit.variance_covariance.vcov,
            reference.variance_covariance.vcov,
            rtol=1e-10,
            atol=1e-12,
            err_msg="multiway multiple estimation covariance",
        )
        np.testing.assert_allclose(
            fit.pvalue(),
            reference.pvalue(),
            rtol=1e-10,
            atol=1e-12,
            err_msg="multiway retained inference",
        )
        if storage:
            with pytest.raises(MissingModelDataError, match="vcov"):
                fit.vcov(vcov)


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
