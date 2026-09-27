from dataclasses import fields, replace
from inspect import signature

import numpy as np
import pytest

from pyfixest.demeaners import LsmrDemeaner, MapDemeaner
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


def _log1p_abs(x):
    return np.log1p(np.abs(x))


@pytest.fixture
def data_offset():
    data = get_data(model="Fepois").dropna().reset_index(drop=True)
    rng = np.random.default_rng(8123)
    data["off"] = np.log(rng.uniform(0.5, 2.0, size=len(data)))
    return data


@pytest.mark.parametrize(
    ("estimator", "fml", "options"),
    [
        (fepois, "Y ~ X1 + X2", {"offset": "off", "iwls_tol": 1e-12}),
        (fepois, "Y ~ X1 + X2", {"offset": "off", "drop_intercept": True}),
        (
            feols,
            "Y ~ log1p_abs(X1) + X2 | f3",
            {"context": {"log1p_abs": _log1p_abs}, "fixef_rm": "none"},
        ),
    ],
)
def test_crv3_refits_replay_estimation_options(data_offset, estimator, fml, options):
    "The CRV3 jackknife refits with the options of the original fit."
    fit = estimator(
        fml,
        data=data_offset,
        vcov={"CRV3": "f1"},
        ssc=ssc(k_adj=False, G_adj=False),
        **options,
    )
    beta_hat = fit.coef().to_numpy()

    # reference: leave-one-cluster-out refits through the public API
    vcov_jack = np.zeros((len(beta_hat), len(beta_hat)))
    for g in np.unique(data_offset["f1"]):
        beta_g = estimator(fml, data=data_offset[data_offset["f1"] != g], **options)
        deviation = beta_g.coef().to_numpy() - beta_hat
        vcov_jack += np.outer(deviation, deviation)

    # identical refits; only the accumulation order of the sum differs
    np.testing.assert_allclose(
        fit.variance_covariance.vcov, vcov_jack, rtol=1e-10, atol=0, err_msg="CRV3"
    )


@pytest.mark.parametrize("estimator", [feols, fepois])
def test_refit_replays_options(data_offset, estimator):
    "Leave-out and resampled refits reuse the fit's options without copying the data."
    fml = "Y ~ log1p_abs(X1) + X2 | f3"
    options = {
        "weights": "weights",
        "ssc": ssc(k_adj=False, k_fixef="full"),
        "fixef_rm": "none",
        "collin_tol": 1e-7,
        "solver": "np.linalg.solve",
        "demeaner": MapDemeaner(fixef_tol=1e-9),
        "context": {"log1p_abs": _log1p_abs},
        "lean": True,
    }
    if estimator is fepois:
        options |= {
            "offset": "off",
            "iwls_tol": 1e-10,
            "iwls_maxiter": 40,
            "separation_check": ["fe"],
            "accelerate": False,
        }
    fit = estimator(fml, data=data_offset, **options)

    refit = fit._refit(fml=fml, data=data_offset, vcov="iid")

    assert refit.options == replace(fit.options, copy_data=False)
    np.testing.assert_allclose(
        refit.coef().to_numpy(), fit.coef().to_numpy(), rtol=1e-12, err_msg="coef"
    )


# estimation arguments that are not options of a single fit: the caller of a
# refit chooses the formula, data and covariance, the refit data is already one
# model's sample, and the compression arguments are deprecated
_ARGUMENTS_OUTSIDE_OPTIONS = {
    "fml",
    "data",
    "vcov",
    "vcov_kwargs",
    "split",
    "fsplit",
    "use_compression",
    "reps",
    "seed",
}
_OPTION_OF_ARGUMENT = {
    "fixef_rm": "drop_singletons",
    "iwls_tol": "tol",
    "iwls_maxiter": "maxiter",
}


@pytest.mark.parametrize("estimator", [feols, fepois])
def test_estimation_arguments_reach_options(data_offset, estimator):
    "Refits replay `fit.options`, so every other estimation argument must land there."
    fit = estimator("Y ~ X1", data=data_offset)
    arguments = set(signature(estimator).parameters) - _ARGUMENTS_OUTSIDE_OPTIONS

    option_names = {option.name for option in fields(fit.options)}
    assert {_OPTION_OF_ARGUMENT.get(a, a) for a in arguments} <= option_names


@pytest.mark.parametrize("variant", ["additive", "diagonal"])
def test_refits_rebuild_prebuilt_preconditioner(variant):
    "A preconditioner built on the full sample is not reused on refit samples."
    data = get_data().dropna()
    data["Y"] = np.abs(data["Y"]).round()
    fml = "Y ~ X1 | f1 + f2"
    by_name = LsmrDemeaner(preconditioner=variant)
    prebuilt = LsmrDemeaner(
        preconditioner=fepois(fml, data, demeaner=by_name).preconditioner
    )

    fit = fepois(fml, data, demeaner=prebuilt, vcov={"CRV3": "f1"})
    expected = fepois(fml, data, demeaner=by_name, vcov={"CRV3": "f1"})
    # same preconditioner variant; the LSMR solves stop at their 1e-8 tolerance
    np.testing.assert_allclose(
        fit.variance_covariance.vcov,
        expected.variance_covariance.vcov,
        rtol=1e-6,
        err_msg="CRV3",
    )

    ritest_kwargs = {"resampvar": "X1", "reps": 3, "store_ritest_statistics": True}
    fit.ritest(rng=np.random.default_rng(5), **ritest_kwargs)
    expected.ritest(rng=np.random.default_rng(5), **ritest_kwargs)
    np.testing.assert_allclose(
        fit.ritest_statistics.statistics,
        expected.ritest_statistics.statistics,
        rtol=1e-6,
        err_msg="ri stats",
    )
