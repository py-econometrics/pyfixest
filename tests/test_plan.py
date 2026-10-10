"""Unit tests for the estimation planner (`pyfixest.estimation.plan_`)."""

from __future__ import annotations

import numpy as np
import pytest

import pyfixest as pf
from pyfixest.demeaners import MapDemeaner
from pyfixest.estimation.config import EstimationConfig, QuantileProcess
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.internals.literals import EstimationMethod
from pyfixest.estimation.internals.model_state import (
    EstimationOptions,
    GlmEstimationOptions,
    QuantregEstimationOptions,
    SampleSplit,
    VcovSpec,
)
from pyfixest.estimation.models.fegaussian_ import Fegaussian
from pyfixest.estimation.models.feiv_ import Feiv
from pyfixest.estimation.models.felogit_ import Felogit
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.estimation.plan_ import (
    MODEL_REGISTRY,
    ModelSpec,
    _resolve_model_class,
    build_all_splits,
    expand_specs,
    fit_one,
)
from pyfixest.estimation.quantreg.quantreg_ import Quantreg
from pyfixest.estimation.quantreg.QuantregMulti import QuantregMulti
from pyfixest.utils.utils import Ssc

_SHARED_OPTIONS = dict(
    ssc=Ssc(),
    drop_singletons=True,
    drop_intercept=False,
    weights=None,
    weights_type="aweights",
    offset=None,
    collin_tol=1e-9,
    solver="scipy.linalg.solve",
    demeaner=MapDemeaner(),
    store_data=True,
    copy_data=True,
    lean=False,
    context={},
)


def _config(method: EstimationMethod, fml: str, data, **overrides) -> EstimationConfig:
    """Minimal config builder for planner tests, with feols options."""
    base = dict(
        method=method,
        data=data,
        fml=fml,
        options=EstimationOptions(**_SHARED_OPTIONS),
        vcov=VcovSpec.from_user_input("iid"),
    )
    base.update(overrides)
    return EstimationConfig(**base)


def _parse(fml: str):
    return Formula.parse_to_dict(fml)


def _is_iv(formula_dict) -> bool:
    return any(f.first_stage is not None for fs in formula_dict.values() for f in fs)


# ---------------------------------------------------------------------------
# Model registry / dispatch
# ---------------------------------------------------------------------------


def test_registry_covers_every_supported_method():
    expected = {
        "feols",
        "fepois",
        "feglm-logit",
        "feglm-probit",
        "feglm-gaussian",
        "quantreg",
    }
    assert set(MODEL_REGISTRY.keys()) == expected


@pytest.mark.parametrize(
    "method,is_iv,fits_quantile_process,expected_cls",
    [
        ("feols", False, False, Feols),
        ("feols", True, False, Feiv),
        ("fepois", False, False, Fepois),
        ("feglm-logit", False, False, Felogit),
        ("feglm-gaussian", False, False, Fegaussian),
        ("quantreg", False, False, Quantreg),
        ("quantreg", False, True, QuantregMulti),
    ],
)
def test_resolve_model_class(method, is_iv, fits_quantile_process, expected_cls):
    resolved = _resolve_model_class(
        method, is_iv=is_iv, fits_quantile_process=fits_quantile_process
    )
    assert resolved is expected_cls


def test_iv_only_promotes_feols():
    """is_iv=True for any non-feols method falls back to the registry entry."""
    resolved = _resolve_model_class("fepois", is_iv=True, fits_quantile_process=False)
    assert resolved is Fepois


def test_quantile_process_needs_a_method_that_fits_one():
    """A method without a quantile-process model class rejects a process."""
    with pytest.raises(TypeError, match="cannot fit a quantile process"):
        _resolve_model_class("feols", is_iv=False, fits_quantile_process=True)


# ---------------------------------------------------------------------------
# Split enumeration
# ---------------------------------------------------------------------------


def test_build_all_splits_full_only():
    data = pf.get_data()
    splits = build_all_splits(run_full=True, run_split=False, splitvar=None, data=data)
    assert splits == [None]


def test_build_all_splits_split_only():
    data = pf.get_data()
    splits = build_all_splits(run_full=False, run_split=True, splitvar="f1", data=data)
    expected = sorted(data["f1"].dropna().unique().tolist())
    assert splits == [SampleSplit(var="f1", value=value) for value in expected]


def test_build_all_splits_full_plus_split_puts_full_first():
    data = pf.get_data()
    splits = build_all_splits(run_full=True, run_split=True, splitvar="f1", data=data)
    assert splits[0] == SampleSplit(var="f1", value=None)
    assert [split.value for split in splits[1:]] == sorted(
        data["f1"].dropna().unique().tolist()
    )


# ---------------------------------------------------------------------------
# expand_specs: spec count & ordering
# ---------------------------------------------------------------------------


def test_single_formula_emits_one_spec():
    data = pf.get_data()
    cfg = _config("feols", "Y ~ X1 + X2 | f1", data)
    fd = _parse(cfg.fml)
    specs = expand_specs(
        config=cfg,
        formula_dict=fd,
        data=data,
        splits=[None],
        is_iv=False,
    )
    assert len(specs) == 1
    assert specs[0].method == "feols"
    assert specs[0].model_cls is Feols
    assert specs[0].cache_key == (None, "f1")


def test_csw_emits_one_spec_per_fixef_step():
    data = pf.get_data()
    cfg = _config("feols", "Y ~ X1 | csw(f1, f2)", data)
    fd = _parse(cfg.fml)
    specs = expand_specs(
        config=cfg,
        formula_dict=fd,
        data=data,
        splits=[None],
        is_iv=False,
    )
    # csw(f1, f2) → two fixef keys: "f1" then "f1+f2"
    assert len(specs) == 2
    assert specs[0].fixef_key != specs[1].fixef_key


def test_cache_keys_are_contiguous_blocks():
    """Cache blocks form contiguous runs in the spec list.

    This is the invariant the runner relies on to drop the demean /
    preconditioner cache without re-allocating per spec.
    """
    data = pf.get_data()
    cfg = _config("feols", "Y + Y2 ~ X1 | csw(f1, f2)", data)
    fd = _parse(cfg.fml)
    specs = expand_specs(
        config=cfg,
        formula_dict=fd,
        data=data,
        splits=[None],
        is_iv=False,
    )
    seen: list = []
    for spec in specs:
        if not seen or spec.cache_key != seen[-1]:
            seen.append(spec.cache_key)
    # Each cache_key should appear in `seen` exactly once if blocks
    # are contiguous — i.e. once the runner has left a block it
    # never comes back.
    assert len(seen) == len(set(seen))


def test_split_expansion_walks_full_then_each_split_value():
    data = pf.get_data()
    cfg = _config(
        "feols",
        "Y ~ X1 | f1",
        data,
        fsplit="f2",
    )
    fd = _parse(cfg.fml)
    splits = build_all_splits(run_full=True, run_split=True, splitvar="f2", data=data)
    specs = expand_specs(
        config=cfg,
        formula_dict=fd,
        data=data,
        splits=splits,
        is_iv=False,
    )
    assert len(specs) == len(splits)
    assert [s.sample_split for s in specs] == splits


def test_iv_formula_resolves_each_spec_to_feiv():
    data = pf.get_data()
    cfg = _config("feols", "Y ~ X2 | f1 | X1 ~ Z1", data)
    fd = _parse(cfg.fml)
    is_iv = _is_iv(fd)
    assert is_iv
    specs = expand_specs(
        config=cfg,
        formula_dict=fd,
        data=data,
        splits=[None],
        is_iv=is_iv,
    )
    assert all(s.model_cls is Feiv for s in specs)


# ---------------------------------------------------------------------------
# Method-specific estimation options
# ---------------------------------------------------------------------------

_GLM_KWARGS = {
    "demeaner": MapDemeaner(fixef_tol=1e-3),
    "iwls_tol": 1e-7,
    "iwls_maxiter": 13,
    "separation_check": ["fe"],
    "accelerate": False,
}
_GLM_OPTIONS = {
    "demeaner": MapDemeaner(fixef_tol=1e-3),
    "tol": 1e-7,
    "maxiter": 13,
    "separation_check": ["fe"],
    "accelerate": False,
}


@pytest.mark.parametrize(
    "estimate,options_cls,expected",
    [
        # feols: the shared options only, with the configured demeaner
        (
            lambda data: pf.feols(
                "Y ~ X1 | f1",
                data,
                demeaner=MapDemeaner(fixef_tol=1e-3),
                fixef_rm="none",
                weights="weights",
                collin_tol=1e-8,
            ),
            EstimationOptions,
            {
                "demeaner": MapDemeaner(fixef_tol=1e-3),
                "drop_singletons": False,
                "weights": "weights",
                "collin_tol": 1e-8,
                "offset": None,
            },
        ),
        # fepois: the IRLS options, the offset, and the user's `accelerate`
        (
            lambda data: pf.fepois(
                "Y ~ X1 | f1", data.assign(Y=data.Y.abs()), offset="X2", **_GLM_KWARGS
            ),
            GlmEstimationOptions,
            {**_GLM_OPTIONS, "offset": "X2"},
        ),
        # feglm-logit: the IRLS options; only Poisson takes an offset
        (
            lambda data: pf.feglm(
                "Y ~ X1 | f1",
                data.assign(Y=(data.Y > 0).astype(int)),
                family="logit",
                **_GLM_KWARGS,
            ),
            GlmEstimationOptions,
            {**_GLM_OPTIONS, "offset": None},
        ),
        # quantreg: the solver options of the quantile fit
        (
            lambda data: pf.quantreg(
                "Y ~ X1",
                data,
                quantile=0.5,
                method="pfn",
                tol=1e-5,
                maxiter=7,
                seed=42,
            ),
            QuantregEstimationOptions,
            {
                "quantile": 0.5,
                "method": "pfn",
                "quantile_tol": 1e-5,
                "quantile_maxiter": 7,
                "seed": 42,
            },
        ),
    ],
)
def test_estimation_functions_build_the_options_of_the_method(
    estimate, options_cls, expected
):
    """Each estimation function builds the options value its model class takes."""
    data = pf.get_data().dropna()
    options = estimate(data).options
    assert type(options) is options_cls
    for name, value in expected.items():
        assert getattr(options, name) == value, name


def test_fepois_and_feglm_poisson_honor_accelerate():
    """`accelerate=False` must reach the fitted model, not just `fepois()`'s default.

    Regression test: the `fepois` and `feglm-poisson` calls share the
    `"fepois"` planner entry, which used to hard-code `accelerate=True` for
    that entry regardless of what the caller asked for — silently ignoring
    an explicit `accelerate=False` passed through either `fepois()` or
    `feglm(family="poisson")`.
    """
    data = pf.get_data(model="Fepois")

    fit_default = pf.fepois("Y ~ X1 + X2 | f1", data)
    assert fit_default.options.accelerate is True

    fit_fepois = pf.fepois("Y ~ X1 + X2 | f1", data, accelerate=False)
    assert fit_fepois.options.accelerate is False

    fit_feglm = pf.feglm("Y ~ X1 + X2 | f1", data, family="poisson", accelerate=False)
    assert fit_feglm.options.accelerate is False


def test_options_object_is_shared_across_multiple_estimation():
    """`expand_specs` builds one options object and shares it across every spec.

    Guards the options-at-boundary refactor's core invariant: a
    multi-estimation call (here, multiple LHS combined with `csw()`) must
    not rebuild or mutate the options per spec, since that could let option
    values silently drift between the estimations it fans out.
    """
    data = pf.get_data()
    cfg = _config("feols", "Y + Y2 ~ csw(X1, X2) | f1", data)
    fd = _parse(cfg.fml)
    specs = expand_specs(
        config=cfg,
        formula_dict=fd,
        data=data,
        splits=[None],
        is_iv=False,
    )
    assert len(specs) > 1
    assert all(spec.options is cfg.options for spec in specs)


def test_quantile_process_is_handed_to_every_spec():
    """`QuantregMulti` gets the fan-out on top of the shared options."""
    data = pf.get_data()
    process = QuantileProcess(quantiles=[0.25, 0.75], multi_method="cfm1")
    options = QuantregEstimationOptions(
        **_SHARED_OPTIONS,
        quantile=0.25,
        method="fn",
        quantile_tol=1e-6,
        quantile_maxiter=None,
        seed=None,
    )
    cfg = _config("quantreg", "Y ~ X1", data, options=options, quantile_process=process)
    specs = expand_specs(
        config=cfg,
        formula_dict=_parse(cfg.fml),
        data=data,
        splits=[None],
        is_iv=False,
    )
    assert [spec.quantile_process for spec in specs] == [process]
    assert specs[0].model_cls is QuantregMulti


def test_quantile_process_children_get_distinct_quantiles_shared_options():
    """Each child `Quantreg` in `QuantregMulti` carries its own quantile but
    shares every other option, notably `seed`, so bootstrap-style draws stay
    reproducible across the quantiles fit within one process.
    """
    fits = pf.quantreg(
        "Y ~ X1", pf.get_data(), quantile=[0.25, 0.5, 0.75], method="pfn", seed=7
    )
    children = fits.all_fitted_models.values()
    assert sorted(child.options.quantile for child in children) == [0.25, 0.5, 0.75]
    for child in children:
        assert child.options.seed == 7
        assert child.options.method == "pfn"


def test_options_must_match_the_model_class():
    """A config whose options do not fit the method's model class is rejected."""
    data = pf.get_data()
    cfg = _config("fepois", "Y ~ X1", data)
    with pytest.raises(TypeError, match="GlmEstimationOptions"):
        expand_specs(
            config=cfg,
            formula_dict=_parse(cfg.fml),
            data=data,
            splits=[None],
            is_iv=False,
        )


# ---------------------------------------------------------------------------
# End-to-end smoke: planner output matches the public API
# ---------------------------------------------------------------------------


def test_public_feols_matches_legacy_behavior():
    """Sanity check: the planner doesn't change end-to-end results."""
    data = pf.get_data()
    fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data)
    # If the planner regressed anything, coefficients would shift.
    assert abs(fit.coef().iloc[0] - (-0.9240461507764969)) < 1e-10


def test_fit_one_uses_the_structural_lifecycle_contract():
    """The generic pipeline delegates estimator-specific work through hooks."""
    events: list[str] = []

    class StubModel:
        _X_is_empty = False

        def __init__(self, **kwargs):
            self.events = events

        def prepare_model_matrix(self):
            self.events.append("prepare")

        def _validate_response(self):
            self.events.append("validate")

        def get_fit(self):
            self.events.append("fit")

        def _publish_fit_statistics(self):
            self.events.append("fit statistics")

        def _check_vcov_support(self, spec):
            assert spec == iid
            self.events.append("check vcov")

        def _vcov_from_spec(self, spec):
            assert spec == iid
            self.events.append("vcov")

        def get_inference(self):
            self.events.append("inference")

        def _finalize_fit(self):
            self.events.append("finalize")

        def _clear_attributes(self):
            self.events.append("clear")

        def _iter_fitted_models(self):
            return ()

    iid = VcovSpec.from_user_input("iid")
    formula = Formula.parse_to_dict("Y ~ X1")[None][0]
    spec = ModelSpec(
        method="quantreg",
        model_cls=StubModel,
        formula=formula,
        fixef_key=None,
        data=pf.get_data(),
        options=EstimationOptions(**_SHARED_OPTIONS),
        sample_split=None,
    )

    fit_one(
        spec,
        lookup_demeaned_data={},
        lookup_preconditioner={},
        vcov=iid,
    )

    assert events == [
        "prepare",
        "validate",
        "check vcov",
        "fit",
        "fit statistics",
        "vcov",
        "inference",
        "finalize",
    ], "fit_one returns complete models; the estimation functions apply retention"


def test_quantreg_multi_prepares_children_in_lifecycle_hook():
    """Multi-quantile construction is deferred to the preparation phase."""
    events: list[str] = []

    class StubQuantreg:
        def prepare_model_matrix(self):
            events.append("prepare")

        def to_array(self):
            events.append("to_array")

        def drop_multicol_vars(self):
            events.append("drop_multicol_vars")

    fit = QuantregMulti.__new__(QuantregMulti)
    fit.all_quantregs = {0.25: StubQuantreg(), 0.75: StubQuantreg()}

    fit.prepare_model_matrix()

    assert events == [
        "prepare",
        "to_array",
        "drop_multicol_vars",
        "prepare",
        "to_array",
        "drop_multicol_vars",
    ]
    assert fit._X_is_empty is False


@pytest.mark.xfail(
    strict=True,
    reason="Pending structural formula identity; regression from PR stack #1862",
)
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize(
    "fml, references",
    [
        ("Y ~ X1 | sw({f1 + f2}, f1 + f2)", ["Y ~ X1 | {f1 + f2}", "Y ~ X1 | f1 + f2"]),
        (
            "Y ~ X1 | sw(`f1 + f2`, {f1 + f2})",
            ["Y ~ X1 | `f1 + f2`", "Y ~ X1 | {f1 + f2}"],
        ),
        ("Y ~ sw({X1 + X2}, X1 + X2) | f1", ["Y ~ {X1 + X2} | f1", "Y ~ X1 + X2 | f1"]),
    ],
)
def test_distinct_structures_with_same_display_survive(fml, references, weights):
    """#1776: model identity and demean-cache scopes use parsed semantics."""
    data = pf.get_data(N=400, seed=123).dropna()
    data["f1 + f2"] = data.f2
    multi = pf.feols(fml, data=data, weights=weights, fixef_rm="none")
    fits = multi.to_list()
    assert list(multi.all_fitted_models) == [fit.model.model_name for fit in fits]
    assert all(multi.all_fitted_models[fit.model.model_name] is fit for fit in fits)
    assert len(fits) == len(references)
    for fit, reference_formula in zip(fits, references, strict=True):
        reference = pf.feols(
            reference_formula, data=data, weights=weights, fixef_rm="none"
        )
        assert list(fit.coef().index) == list(reference.coef().index)
        np.testing.assert_allclose(
            fit.coef(), reference.coef(), rtol=1e-10, err_msg="stepwise coefficients"
        )
        np.testing.assert_allclose(
            fit.se(), reference.se(), rtol=1e-10, err_msg="stepwise standard errors"
        )


@pytest.mark.xfail(
    strict=True,
    reason="Pending structural formula identity; regression from PR stack #1862",
)
def test_full_sample_and_group_named_all_have_distinct_identity():
    data = pf.get_data(N=300).dropna()
    data["group"] = np.resize(["all", "'all'", "rest"], len(data))
    multi = pf.feols("Y ~ X1 | f1", data=data, fsplit="group")
    fits = multi.to_list()
    assert [fit.model.sample_split.value for fit in fits] == [
        None,
        "'all'",
        "all",
        "rest",
    ]
    assert all(multi.all_fitted_models[fit.model.model_name] is fit for fit in fits)


@pytest.mark.parametrize("fml", ["Y ~ sw(X1, X2)", "Y ~ sw(X1, X2) | f1"])
@pytest.mark.parametrize("split", [False, True])
def test_multiple_models_support_name_lookup(fml, split):
    data = pf.get_data(N=200, seed=17).dropna()
    data["group"] = data.f1 % 2
    multi = pf.feols(fml, data=data, fsplit="group" if split else None)
    models = multi.to_list()
    assert list(multi.all_fitted_models) == [fit.model.model_name for fit in models]
    for fit in models:
        assert multi.all_fitted_models[fit.model.model_name] is fit
        reference = pf.feols(fit.model.formula, data=data)
        suffix = ""
        if fit.model.sample_split is not None:
            value = fit.model.sample_split.value
            suffix = f" (Sample: group = {'all' if value is None else value})"
        assert fit.model.model_name == reference.model.model_name + suffix


def test_quantile_models_support_name_lookup():
    multi = pf.quantreg(
        "Y ~ X1",
        pf.get_data(N=100, seed=17).dropna(),
        quantile=[0.25, 0.75],
        method="pfn",
        seed=7,
    )
    models = multi.to_list()
    assert len(models) == 2
    assert list(multi.all_fitted_models) == [fit.model.model_name for fit in models]
    for fit in models:
        assert multi.all_fitted_models[fit.model.model_name] is fit
        assert fit.model.model_name.endswith(f"(q = {fit.options.quantile})")


@pytest.mark.xfail(strict=True, reason="Pending unambiguous formula rendering")
@pytest.mark.parametrize(
    "fml, names",
    [
        ("Y ~ X1 | sw({f1 + f2}, f1 + f2)", ["Y ~ X1 | {f1 + f2}", "Y ~ X1 | f1 + f2"]),
        (
            "Y ~ X1 | sw(`f1 + f2`, {f1 + f2})",
            ["Y ~ X1 | `f1 + f2`", "Y ~ X1 | {f1 + f2}"],
        ),
        ("Y ~ sw({X1 + X2}, X1 + X2) | f1", ["Y ~ {X1 + X2} | f1", "Y ~ X1 + X2 | f1"]),
    ],
)
def test_expanded_models_have_unambiguous_string_keys(fml, names):
    data = pf.get_data(N=200, seed=17).dropna()
    data["f1 + f2"] = data.f2
    multi = pf.feols(fml, data=data, fixef_rm="none")
    assert list(multi.all_fitted_models) == names
    for fit, name in zip(multi.to_list(), names, strict=True):
        assert fit.model.model_name == name
        assert multi.all_fitted_models[name] is fit


@pytest.mark.xfail(strict=True, reason="Pending faithful public formula rendering")
@pytest.mark.parametrize(
    "estimator, fml, preserved",
    [
        (pf.feols, "Y ~ X1 - 1", "0 + X1"),
        (pf.fepois, "Y ~ X1 - 1", "0 + X1"),
        (pf.feglm, "Y ~ X1 - 1", "0 + X1"),
        (pf.quantreg, "Y ~ X1 - 1", "0 + X1"),
        (pf.feols, "Y ~ {X1 + X2}", "{X1 + X2}"),
        (pf.feols, "`out come` ~ `X1 + X2`", "`out come` ~ 1 + `X1 + X2`"),
        (pf.feols, "Y ~ X1 | `f1 + f2`", "`f1 + f2`"),
        (pf.feols, "Y ~ X1 | {f1 + f2}", "{f1 + f2}"),
        (pf.feols, "Y ~ X1 - 1 | X2 ~ Z1 - 1", "0 + X1 + [X2 ~ 0 + Z1]"),
        (pf.feols, "Y ~ {X1 + X1 ** 2} | f1 | X2 ~ {Z1 + Z2}", "{Z1 + Z2}"),
    ],
)
def test_public_formula_roundtrip_preserves_estimation(estimator, fml, preserved):
    """#1735/#1759: a public formula must replay the expanded specification."""
    data = pf.get_data(N=400, seed=8123).dropna()
    data["out come"] = data.Y
    data["X1 + X2"] = data.X1
    data["f1 + f2"] = data.f2
    options = {"fixef_rm": "none"}
    if estimator is pf.quantreg:
        options = {}
    elif estimator is pf.fepois:
        data["Y"] = np.round(np.abs(data.Y))
    elif estimator is pf.feglm:
        options["family"] = "gaussian"
    fit = estimator(fml, data=data, **options)
    assert preserved in fit.model.formula
    replay = estimator(fit.model.formula, data=data, **options)
    assert list(replay.coef().index) == list(fit.coef().index)
    np.testing.assert_allclose(replay.coef(), fit.coef(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(replay.se(), fit.se(), rtol=1e-10, atol=1e-10)
