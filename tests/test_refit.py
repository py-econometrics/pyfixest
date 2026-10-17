"""Leave-out and resampled refits replay the estimation options of a fit.

`refit()` reruns a fitted model's estimator on other data. `vcov("CRV3")`
uses it for the leave-one-cluster-out fits, `ritest()` for the resampled fits,
`ccv()` for the split and cluster fits, and IV fits for their first stage, so
all must see every non-default argument of the original fit: the weights of a
`feols` fit, the offset of a `fepois` fit, and so on.
"""

from dataclasses import fields, replace
from functools import partial
from inspect import signature

import numpy as np
import pandas as pd
import pytest

from pyfixest.demeaners import LsmrDemeaner, MapDemeaner
from pyfixest.estimation import feols, fepois
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.internals.model_state import VcovSpec
from pyfixest.estimation.post_estimation.ccv import _compute_CCV
from pyfixest.estimation.post_estimation.ritest import _get_ritest_stats_slow
from pyfixest.estimation.refit import refit
from pyfixest.utils.utils import get_data, ssc


def _log1p_abs(x):
    return np.log1p(np.abs(x))


# One fit per estimator with non-default values for as many estimation
# arguments as possible. `ssc` disables the small-sample corrections so the
# CRV3 covariance is the plain sum of the leave-one-cluster-out deviations.
_COMMON_OPTIONS = {
    "ssc": ssc(k_adj=False, G_adj=False),
    "fixef_rm": "none",
    "collin_tol": 1e-7,
    "solver": "np.linalg.solve",
    "demeaner": MapDemeaner(fixef_tol=1e-9),
    "context": {"log1p_abs": _log1p_abs},
}
CASES = {
    "feols-weights": (feols, {**_COMMON_OPTIONS, "weights": "weights"}),
    "fepois-offset": (
        fepois,
        {
            **_COMMON_OPTIONS,
            "offset": "off",
            "iwls_tol": 1e-10,
            "iwls_maxiter": 40,
            "separation_check": ["fe"],
            "accelerate": False,
        },
    ),
}
FML = "Y ~ log1p_abs(X1) + X2 | f3"
CLUSTER = "f1"
IID = VcovSpec(vcov_type="iid", vcov_type_detail="iid")


def _without_weights(options):
    "`ritest` rejects weighted fits."
    return {name: value for name, value in options.items() if name != "weights"}


@pytest.fixture
def data():
    "Poisson data with a non-default index, weights and an offset column."
    data = get_data(model="Fepois").dropna()
    # a non-default index would expose an in-place index reset in a refit
    data.index = data.index * 2 + 5
    rng = np.random.default_rng(8123)
    data["off"] = np.log(rng.uniform(0.5, 2.0, size=len(data)))
    return data


@pytest.fixture(params=CASES, ids=CASES)
def case(request):
    return CASES[request.param]


@pytest.mark.parametrize(
    "fml", [FML, "Y ~ log1p_abs(X1) + X2 - 1", "Y ~ {X1 * X2} + X2"]
)
def test_refit_replays_options(data, case, fml, monkeypatch):
    "A refit on a subsample equals a public fit with the same options on it."
    estimator, options = case
    fit = estimator(fml, data=data, **options)
    subsample = data[data[CLUSTER] != data[CLUSTER].iloc[0]]

    expected_data = subsample.assign(product=subsample["X1"] * subsample["X2"])
    expected_fml = fml.replace("{X1 * X2}", "product")
    expected = estimator(expected_fml, data=expected_data, **options)

    def fail_on_reparse(*args, **kwargs):
        raise AssertionError("A refit must reuse the parsed formula.")

    monkeypatch.setattr(Formula, "parse", fail_on_reparse)
    monkeypatch.setattr(
        Formula, "render", lambda self: "descriptive text, not a formula"
    )

    refitted = refit(fit, data=subsample, vcov=IID)

    assert refitted.options == replace(fit.options, copy_data=False)
    np.testing.assert_allclose(
        refitted.coef().to_numpy(),
        expected.coef().to_numpy(),
        rtol=1e-12,
        err_msg="coef",
    )


def test_refit_on_the_fit_sample_keeps_every_row(data, case):
    "A same-sample refit drops no singletons and raises if it loses rows."
    estimator, options = case
    fit = estimator(FML, data=data, **{**options, "fixef_rm": "singleton"})

    refitted = refit(fit, data=fit._data, vcov=IID, same_sample=True)

    assert not refitted.options.drop_singletons
    assert refitted.sample_info.n_obs == fit.sample_info.n_obs
    np.testing.assert_allclose(
        refitted.coef().to_numpy(), fit.coef().to_numpy(), rtol=1e-12, err_msg="coef"
    )
    # a missing regressor value drops a row the fit kept
    with_missing = fit._data.copy()
    with_missing.loc[with_missing.index[0], "X2"] = np.nan
    with pytest.raises(ValueError, match="observations"):
        refit(fit, data=with_missing, vcov=IID, same_sample=True)
    # without the flag, the same data is new data and the row is dropped
    assert (
        refit(fit, data=with_missing, vcov=IID).sample_info.n_obs
        == fit.sample_info.n_obs - 1
    )
    with pytest.raises(ValueError, match="same index"):
        refit(fit, data=fit._data.iloc[1:], vcov=IID, same_sample=True)


@pytest.mark.parametrize("fml", [FML, "Y ~ log1p_abs(X1) + X2 - 1"])
def test_crv3_is_the_leave_one_cluster_out_jackknife(data, case, fml):
    "CRV3 sums the outer products of the leave-one-cluster-out deviations."
    estimator, options = case
    fit = estimator(fml, data=data, vcov={"CRV3": CLUSTER}, **options)
    beta_hat = fit.coef().to_numpy()

    vcov_jack = np.zeros((len(beta_hat), len(beta_hat)))
    for g in data[CLUSTER].unique():
        beta_g = estimator(fml, data=data[data[CLUSTER] != g], **options).coef()
        deviation = beta_g.to_numpy() - beta_hat
        vcov_jack += np.outer(deviation, deviation)

    # identical refits; only the accumulation order of the sum differs
    np.testing.assert_allclose(
        fit.variance_covariance.vcov, vcov_jack, rtol=1e-10, atol=0, err_msg="CRV3"
    )


def test_ritest_refits_replay_options(data, case):
    "The resampled refits of the slow ritest algorithm keep the fit's options."
    estimator, options = case
    options = _without_weights(options)
    fit = estimator(FML, data=data, **options)
    ritest_kwargs = {"resampvar": "X2", "reps": 5, "type": "randomization-c"}

    fit.ritest(
        **ritest_kwargs,
        choose_algorithm="slow",
        rng=np.random.default_rng(3),
        store_ritest_statistics=True,
    )

    expected = _get_ritest_stats_slow(
        **ritest_kwargs,
        data=fit._data,
        fit_fn=partial(estimator, fml=FML, vcov="iid", **options),
        rng=np.random.default_rng(3),
    )
    np.testing.assert_allclose(
        fit.ritest_statistics.statistics, expected, rtol=1e-12, err_msg="ri stats"
    )


@pytest.mark.parametrize("drop_intercept", [False, True])
def test_ccv_refits_replay_options(data, drop_intercept):
    "The split and cluster refits of `ccv()` keep the fit's options."
    # ccv rejects fixed effects and weights; without an intercept, split
    # coefficients that ignored `drop_intercept` would not match the design
    options = {**_COMMON_OPTIONS, "drop_intercept": drop_intercept}
    fml = "Y ~ D + log1p_abs(X1) + X2"
    if not drop_intercept:
        fml += " - 1"
    rng = np.random.default_rng(41)
    data["D"] = rng.integers(0, 2, size=len(data))
    # few large clusters, so every cluster's split subsample has full rank
    data["cluster"] = rng.integers(0, 5, size=len(data))
    fit = feols(fml, data=data, vcov={"CRV1": "cluster"}, **options)

    ccv = fit.ccv(treatment="D", seed=7, n_splits=1, pk=0.5)

    expected = _compute_CCV(
        fit_fn=partial(feols, fml, vcov="iid", **options),
        Y=fit.within_data.response.flatten(),
        X=fit.within_data.design,
        W=fit._data["D"].to_numpy(),
        rng=np.random.default_rng(7),
        data=fit._data,
        treatment="D",
        cluster_vec=fit._data["cluster"].to_numpy(),
        pk=0.5,
        tau_full=fit.coef()["D"],
    )
    np.testing.assert_allclose(
        ccv.loc["CCV", "Std. Error"],
        np.sqrt(expected / fit.sample_info.n_obs),
        rtol=1e-12,
        err_msg="ccv se",
    )


def test_refits_leave_data_untouched(data, case):
    "Refits modify neither the fit's sample nor the user's frame."
    estimator, options = case
    options = _without_weights(options)
    # copy_data=False lets the original fit reset the index; refits must not
    fit = estimator(FML, data=data, copy_data=False, **options)
    data_before = data.copy()
    sample_before = fit._data.copy()

    fit.vcov({"CRV3": CLUSTER})
    fit.ritest("X2", reps=2, choose_algorithm="slow", rng=np.random.default_rng(1))
    refit(fit, data=fit._data, vcov=IID, same_sample=True)

    pd.testing.assert_frame_equal(fit._data, sample_before)
    pd.testing.assert_frame_equal(data, data_before)


@pytest.mark.parametrize("variant", ["additive", "diagonal"])
def test_refit_rebuilds_prebuilt_preconditioner(variant):
    "A preconditioner built on the full sample is not reused on a refit sample."
    data = get_data().dropna()
    data["Y"] = np.abs(data["Y"]).round()
    fml = "Y ~ X1 | f1 + f2"
    by_name = LsmrDemeaner(preconditioner=variant)
    prebuilt = LsmrDemeaner(
        preconditioner=fepois(fml, data, demeaner=by_name).preconditioner
    )
    fit = fepois(fml, data, demeaner=prebuilt)
    subsample = data[data["f1"] != data["f1"].iloc[0]]

    refitted = refit(fit, data=subsample, vcov=IID)

    assert refitted.options.demeaner == by_name
    expected = fepois(fml, data=subsample, demeaner=by_name)
    # same preconditioner variant; the LSMR solves stop at their 1e-8 tolerance
    np.testing.assert_allclose(
        refitted.coef().to_numpy(),
        expected.coef().to_numpy(),
        rtol=1e-6,
        err_msg="coef",
    )


@pytest.mark.parametrize("variant", ["additive", "diagonal"])
def test_refit_reuses_preconditioner_on_the_fit_sample(variant):
    "A same-sample refit reuses the preconditioner the fit built."
    data = get_data().dropna()
    data["Y"] = np.abs(data["Y"]).round()
    fml = "Y ~ X1 | f1 + f2"
    fit = fepois(fml, data, demeaner=LsmrDemeaner(preconditioner=variant))

    refitted = refit(fit, data=fit._data, vcov=IID, same_sample=True)

    assert fit.preconditioner is not None
    assert refitted.options.demeaner.preconditioner is fit.preconditioner


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
def test_estimation_arguments_reach_options(data, estimator):
    "Refits replay `fit.options`, so every other estimation argument must land there."
    fit = estimator("Y ~ X1", data=data)
    arguments = set(signature(estimator).parameters) - _ARGUMENTS_OUTSIDE_OPTIONS

    option_names = {option.name for option in fields(fit.options)}
    assert {_OPTION_OF_ARGUMENT.get(a, a) for a in arguments} <= option_names
