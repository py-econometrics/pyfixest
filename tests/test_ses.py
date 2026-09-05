from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pyfixest.estimation as estimation
from pyfixest.demeaners import LsmrDemeaner, MapDemeaner
from pyfixest.errors import MatrixNotFullRankError
from pyfixest.estimation import feols, fepois
from pyfixest.utils.utils import get_data, ssc
from tests.test_estimator_state_lifecycle import (
    _assert_inference_unchanged,
    _snapshot_inference,
)


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

    _N = fit1._N
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

    if not np.allclose(fit1._vcov / adj_factor, fit2._vcov):
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

    _N = fit1._N
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

    if not np.allclose(fit1._vcov / adj_factor, fit2._vcov):
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


def test_fepois_crv3_replays_estimation_contract(monkeypatch):
    """Poisson jackknife refits must preserve every coefficient-affecting option."""
    rng = np.random.default_rng(20260901)
    n_clusters = 6
    rows_per_cluster = 12
    n = n_clusters * rows_per_cluster
    cluster = np.repeat(np.arange(n_clusters), rows_per_cluster)
    fixed_effect = np.tile(np.arange(6), n // 6)
    x = rng.normal(size=n)
    exposure = rng.uniform(0.5, 3.0, size=n)
    weight = rng.uniform(0.75, 1.5, size=n)
    fixed_effect_value = np.linspace(-0.25, 0.25, 6)[fixed_effect]

    def shift(values):
        return values + 0.125

    eta = 0.35 * shift(x) + fixed_effect_value + np.log(exposure)
    data = pd.DataFrame(
        {
            "y": rng.poisson(np.exp(eta)),
            "x": x,
            "exposure": exposure,
            "weight": weight,
            "fixed_effect": fixed_effect,
            "cluster": cluster,
        }
    )

    demeaner = MapDemeaner(fixef_tol=1e-8, fixef_maxiter=20_000)
    ssc_config = ssc(k_adj=False, G_adj=False)
    real_fepois = estimation.fepois
    refit_calls = []
    beta_jack = []

    def recording_fepois(*, data, **kwargs):
        refit_calls.append((data, kwargs))
        refit = real_fepois(data=data, **kwargs)
        beta_jack.append(refit.coef().to_numpy())
        return refit

    monkeypatch.setattr(estimation, "fepois", recording_fepois)
    fit = real_fepois(
        "y ~ shift(x) | fixed_effect",
        data=data,
        vcov={"CRV3": "cluster"},
        weights="weight",
        weights_type="aweights",
        offset="log(exposure)",
        ssc=ssc_config,
        fixef_rm="none",
        iwls_tol=1e-10,
        iwls_maxiter=100,
        collin_tol=1e-8,
        separation_check=[],
        solver="np.linalg.lstsq",
        demeaner=demeaner,
        drop_intercept=True,
        context={"shift": shift},
    )

    assert len(refit_calls) == n_clusters
    for refit_data, kwargs in refit_calls:
        assert refit_data["cluster"].nunique() == n_clusters - 1
        assert kwargs["fml"] == "y ~ shift(x) | fixed_effect"
        assert kwargs["vcov"] == "iid"
        assert kwargs["weights"] == "weight"
        assert kwargs["weights_type"] == "aweights"
        assert kwargs["ssc"] == ssc_config
        assert kwargs["fixef_rm"] == "none"
        assert kwargs["iwls_tol"] == 1e-10
        assert kwargs["iwls_maxiter"] == 100
        assert kwargs["collin_tol"] == 1e-8
        assert kwargs["separation_check"] == []
        assert kwargs["solver"] == "np.linalg.lstsq"
        assert kwargs["demeaner"] is demeaner
        assert kwargs["drop_intercept"] is True
        assert kwargs["offset"] == "log(exposure)"
        assert kwargs["context"]["shift"] is shift

    expected_vcov = np.zeros_like(fit._vcov)
    for beta in beta_jack:
        centered = beta - fit.coef().to_numpy()
        expected_vcov += np.outer(centered, centered)
    np.testing.assert_allclose(fit._vcov, expected_vcov, rtol=1e-12, atol=1e-12)


def test_crv3_rebuilds_preconditioner_for_each_changed_design():
    """Leave-cluster-out refits must not reuse a full-sample factorization."""
    rng = np.random.default_rng(20260902)
    n_groups = 6
    rows_per_group = 6
    n = n_groups * rows_per_group
    data = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "group": np.repeat(np.arange(n_groups), rows_per_group),
            "time": np.tile(np.arange(rows_per_group), n_groups),
        }
    )
    base = feols(
        "y ~ x | group + time",
        data=data,
        vcov="iid",
        demeaner=LsmrDemeaner(backend="within"),
    )
    assert base.preconditioner is not None
    reused = LsmrDemeaner(
        backend="within",
        preconditioner=base.preconditioner,
    )

    fit = feols(
        "y ~ x | group + time",
        data=data,
        vcov={"CRV3": "group"},
        demeaner=reused,
    )

    assert np.isfinite(fit._vcov).all()


def _crv3_rank_loss_data() -> pd.DataFrame:
    """Build a sample where omitting cluster zero makes z collinear with x."""
    rng = np.random.default_rng(1499)
    n = 120
    cluster = np.repeat(np.arange(6), 20)
    x = rng.normal(size=n)
    z = x + 1e-5 * rng.normal(size=n)
    z[cluster == 0] += rng.normal(size=20)
    return pd.DataFrame(
        {
            "y": 0.5 * x + 2 * z + rng.normal(scale=0.01, size=n),
            "x": x,
            "z": z,
            "fixed_effect": np.tile(np.arange(4), 30),
            "cluster": cluster,
        }
    )


def test_crv3_rejects_leave_cluster_out_coefficient_mismatch():
    """CRV3 must not broadcast a rank-deficient refit's coefficients."""
    data = _crv3_rank_loss_data()
    error = (
        r"omitting cluster .*0.*missing terms: \['z'\]; "
        r"unexpected terms: \[\]"
    )

    with pytest.raises(MatrixNotFullRankError, match=error):
        feols(
            "y ~ x + z | fixed_effect",
            data=data,
            vcov={"CRV3": "cluster"},
            collin_tol=1e-4,
        )

    fit = feols(
        "y ~ x + z | fixed_effect",
        data=data,
        vcov="iid",
        collin_tol=1e-4,
    )
    before = _snapshot_inference(fit)
    with pytest.raises(MatrixNotFullRankError, match=error):
        fit.vcov({"CRV3": "cluster"})
    _assert_inference_unchanged(fit, before)


def test_crv3_aligns_refit_coefficient_names(monkeypatch):
    """Slow CRV3 aligns names and rejects unexpected or duplicate terms."""
    data = _crv3_rank_loss_data()
    fit = feols(
        "y ~ x + z | fixed_effect",
        data=data,
        vcov="iid",
        collin_tol=1e-4,
    )
    refit_coefs = []

    def reordered_refit(*, data):
        i = len(refit_coefs)
        coef = pd.Series([20.0 + i, 10.0 + i], index=["z", "x"])
        refit_coefs.append(coef)
        return SimpleNamespace(coef=lambda: coef)

    monkeypatch.setattr(fit, "_crv3_refit", reordered_refit)
    fit.vcov({"CRV3": "cluster"})
    expected = np.zeros_like(fit._vcov)
    for coef in refit_coefs:
        centered = coef.reindex(fit._coefnames).to_numpy() - fit._beta_hat
        expected += np.outer(centered, centered)
    np.testing.assert_allclose(
        fit._vcov,
        fit._ssc * expected,
        err_msg="CRV3 covariance differs after coefficient-name alignment",
    )

    invalid_refits = [
        (pd.Series([1.0, 2.0, 3.0], index=["x", "z", "w"]), "unexpected.*w"),
        (pd.Series([1.0, 2.0, 3.0], index=["x", "x", "z"]), "duplicate.*x"),
    ]
    for coef, error in invalid_refits:
        monkeypatch.setattr(
            fit,
            "_crv3_refit",
            lambda *, data, coef=coef: SimpleNamespace(coef=lambda: coef),
        )
        before = _snapshot_inference(fit)
        with pytest.raises(MatrixNotFullRankError, match=error):
            fit.vcov({"CRV3": "cluster"})
        _assert_inference_unchanged(fit, before)


@pytest.mark.parametrize("vcov", ["NW", "DK"])
def test_hac_rejects_fweights(vcov):
    """HAC does not support frequency weights."""
    data = get_data().dropna().reset_index(drop=True)
    data["t"] = np.arange(len(data))
    data["unit"] = np.repeat(np.arange(len(data) // 10 + 1), 10)[: len(data)]
    kwargs = {"time_id": "t", "lag": 2}
    if vcov == "DK":
        kwargs["panel_id"] = "unit"

    with pytest.raises(
        NotImplementedError,
        match=r"HAC inference \(NW, DK\) is not supported for models with frequency weights",
    ):
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
