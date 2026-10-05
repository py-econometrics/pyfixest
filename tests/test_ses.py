from __future__ import annotations

import numpy as np
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


@pytest.mark.parametrize("vcov_type", ["CRV1", "CRV3"])
@pytest.mark.parametrize("scale", [1.0, 1e-4])
def test_vcov_fix_updates_inference(indefinite_cluster_data, vcov_type, scale):
    """Repair after combination, including material and sub-threshold changes."""
    import warnings

    data = indefinite_cluster_data.assign(y=indefinite_cluster_data.y * scale)
    vcov = {vcov_type: "c1+c2+c3+c4"}
    fit = feols("y ~ x + z", data, vcov=vcov)
    raw = fit.variance_covariance
    assert np.linalg.eigvalsh(raw.vcov).min() < 0
    coefficients = fit.coef().copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert fit.vcov(vcov, vcov_fix=True) is fit
    fixed = fit.variance_covariance
    repair_warnings = [w for w in caught if "not positive definite" in str(w.message)]
    assert len(repair_warnings) == (scale == 1.0)
    assert all(w.category is UserWarning for w in repair_warnings)
    assert fixed.spec.vcov_fix
    assert np.linalg.eigvalsh(fixed.vcov).min() >= -1e-15
    assert np.isfinite(fit.tidy().to_numpy()).all()
    np.testing.assert_array_equal(
        fit.coef(), coefficients, err_msg="unchanged coefficients"
    )
    if raw.meat is not None:
        np.testing.assert_array_equal(fixed.meat, raw.meat, err_msg="raw sandwich meat")
    assert (fixed.df_k, fixed.df_t, fixed.G) == (raw.df_k, raw.df_t, raw.G)
    # A later call with the default False must restore the uncorrected result.
    fit.vcov(vcov)
    np.testing.assert_array_equal(
        fit.variance_covariance.vcov, raw.vcov, err_msg="opt-in repair"
    )


@pytest.mark.parametrize("retention", [{}, {"lean": True}, {"store_data": False}])
@pytest.mark.parametrize("demeaner", ["numba", "within"])
def test_vcov_fix_multiple_estimation(indefinite_cluster_data, retention, demeaner):
    """Each result is repaired before cleanup; both FE backends reach the seam."""
    import pyfixest as pf
    from pyfixest.errors import MissingModelDataError

    backend = (
        pf.demeaners.MapDemeaner()
        if demeaner == "numba"
        else pf.demeaners.LsmrDemeaner()
    )
    vcov = {"CRV1": "c1+c2+c3"}
    with pytest.warns(UserWarning, match="not positive definite.*fixed"):
        fits = feols(
            "y ~ x + sw(z, d) | c1",
            indefinite_cluster_data,
            vcov=vcov,
            vcov_fix=True,
            demeaner=backend,
            **retention,
        )
    for fit in fits.to_list():
        assert fit.variance_covariance.spec.vcov_fix
        assert np.isfinite(fit.tidy().to_numpy()).all()
    if retention:
        with pytest.raises(MissingModelDataError, match="vcov"):
            fits.vcov(vcov, vcov_fix=True)
    else:
        fits.vcov(vcov)
        with pytest.warns(UserWarning, match="not positive definite.*fixed"):
            assert fits.vcov(vcov, vcov_fix=True) is fits


@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "c1"}, {"CRV1": "c1+c2"}])
def test_vcov_fix_leaves_valid_covariance_unchanged(indefinite_cluster_data, vcov):
    """PD multiway estimates and other covariance types keep their exact values."""
    import warnings

    fit = feols("y ~ x + z", indefinite_cluster_data, vcov=vcov)
    expected = fit.variance_covariance.vcov.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit.vcov(vcov, vcov_fix=True)
    assert not caught
    np.testing.assert_array_equal(
        fit.variance_covariance.vcov, expected, err_msg="unchanged covariance"
    )


@pytest.mark.parametrize(
    "eigenvalue,expect_warning",
    [
        (1.0, False),
        (0.0, False),
        (-1e-9, False),
        (-(1e-8 - 1e-16), False),
        (-1e-8, True),
        (-2e-8, True),
    ],
)
def test_vcov_fix_eigenvalue_floor_and_warning(eigenvalue, expect_warning):
    """Diagonal matrices isolate the absolute threshold and fixest's floor."""
    import warnings

    from pyfixest.estimation.internals.vcov_utils import repair_cluster_vcov

    vcov = np.diag([1.0, eigenvalue])
    before = vcov.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fixed = repair_cluster_vcov(vcov=vcov)
    expected = np.diag([1.0, max(eigenvalue, 1e-16)])
    np.testing.assert_array_equal(fixed, expected, err_msg="fixest eigenvalue floor")
    np.testing.assert_array_equal(vcov, before, err_msg="input covariance unchanged")
    assert len(caught) == int(expect_warning)
    if caught:
        assert caught[0].category is UserWarning
        assert "not positive definite" in str(caught[0].message)
