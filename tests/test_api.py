import duckdb
import numpy as np
import pandas as pd
import pytest
from formulaic.errors import FactorEvaluationError

import pyfixest as pf
from pyfixest.utils.utils import get_data


def test_api():
    df1 = get_data()
    df2 = get_data(model="Fepois")

    fit1 = pf.feols("Y ~ X1 + X2 | f1", data=df1)
    fit2 = pf.estimation.fepois(
        "Y ~ X1 + X2 + f2 | f1", data=df2, vcov={"CRV1": "f1+f2"}
    )
    fit_multi = pf.feols("Y + Y2 ~ X1", data=df2)

    pf.summary(fit1)
    pf.report.summary(fit2)
    pf.etable([fit1, fit2])
    pf.coefplot([fit1, fit2])

    pf.summary(fit_multi)
    pf.etable(fit_multi)
    pf.coefplot(fit_multi)


def test_feols_args():
    """
    Check feols function arguments.

    Arguments to check:
    - copy_data
    - store_data
    - demeaner
    - solver
    """
    df = pf.get_data()

    fit1 = pf.feols(fml="Y ~ X1 | f1 + f2", data=df, copy_data=True)
    fit2 = pf.feols(fml="Y ~ X1 | f1 + f2", data=df, copy_data=False)

    assert (fit1.coef() == fit2.coef()).all()

    fit3 = pf.feols(
        fml="Y ~ X1 | f1 + f2",
        data=df,
        store_data=False,
        demeaner=pf.MapDemeaner(fixef_tol=1e-02),
    )
    if hasattr(fit3, "_data"):
        raise AttributeError(
            "The 'fit3' object has the attribute '_data', which should not be present."
        )

    assert fit1.coef().xs("X1") != fit3.coef().xs("X1")
    assert np.abs(fit1.coef().xs("X1") - fit3.coef().xs("X1")) < 0.01

    fit4 = pf.feols(fml="Y ~ X1 | f1 + f2", data=df, solver="np.linalg.solve")
    fit5 = pf.feols(fml="Y ~ X1 | f1 + f2", data=df, solver="np.linalg.lstsq")

    assert np.allclose(fit4.coef().values, fit5.coef().values)


def test_fepois_args():
    """
    Check feols function arguments.

    Arguments to check:
    - copy_data
    - store_data
    - demeaner
    - solver
    """
    df = pf.get_data(model="Fepois")

    fit1 = pf.fepois(fml="Y ~ X1 | f1 + f2", data=df, copy_data=True)
    fit2 = pf.fepois(fml="Y ~ X1 | f1 + f2", data=df, copy_data=False)

    assert (fit1.coef() == fit2.coef()).all()

    fit3 = pf.fepois(
        fml="Y ~ X1 | f1 + f2",
        data=df,
        store_data=False,
        demeaner=pf.MapDemeaner(fixef_tol=1e-02),
    )
    if hasattr(fit3, "_data"):
        raise AttributeError(
            "The 'fit3' object has the attribute '_data', which should not be present."
        )

    assert fit1.coef().xs("X1") != fit3.coef().xs("X1")
    assert np.abs(fit1.coef().xs("X1") - fit3.coef().xs("X1")) < 0.01

    fit4 = pf.feols(fml="Y ~ X1 | f1 + f2", data=df, solver="np.linalg.solve")
    fit5 = pf.feols(fml="Y ~ X1 | f1 + f2", data=df, solver="np.linalg.lstsq")

    np.testing.assert_allclose(fit4.coef(), fit5.coef(), rtol=1e-12)


def test_map_demeaner_defaults_to_rust():
    data = pf.get_data()

    assert pf.MapDemeaner().backend == "rust"

    fit = pf.feols("Y ~ X1 | f1", data=data)

    assert isinstance(fit._demeaner, pf.MapDemeaner)
    assert fit._demeaner.backend == "rust"


def _run_with_deprecated_kwargs(estimator_name, **kwargs):
    if estimator_name == "feols":
        return pf.feols("Y ~ X1 | f1", data=pf.get_data(), **kwargs)
    if estimator_name == "fepois":
        return pf.fepois("Y ~ X1 | f1", data=pf.get_data(model="Fepois"), **kwargs)
    if estimator_name == "feglm":
        return pf.feglm("Y ~ X1 | f1", data=pf.get_data(), family="gaussian", **kwargs)
    raise ValueError(estimator_name)


_DEPRECATION_ESTIMATORS = ["feols", "fepois", "feglm"]


def test_demeaner_backend_cupy_emits_deprecation_warning():
    from pyfixest.estimation.internals.demeaner_options import (
        _warn_if_deprecated_demeaner_backend,
    )

    with pytest.warns(DeprecationWarning, match=r"`cupy` LSMR demeaner backend") as rec:
        _warn_if_deprecated_demeaner_backend(pf.LsmrDemeaner(backend="cupy"))
    assert any("torch', device='cuda" in str(r.message) for r in rec)
    assert any("default within backend" in str(r.message) for r in rec)


def test_demeaner_backend_cupy_cuda_emits_gpu_replacement_warning():
    from pyfixest.estimation.internals.demeaner_options import (
        _warn_if_deprecated_demeaner_backend,
    )

    with pytest.warns(DeprecationWarning, match=r"`cupy` LSMR demeaner backend") as rec:
        _warn_if_deprecated_demeaner_backend(
            pf.LsmrDemeaner(backend="cupy", device="cuda")
        )
    assert any("torch', device='cuda" in str(r.message) for r in rec)


def test_demeaner_backend_scipy_emits_deprecation_warning():
    from pyfixest.estimation.internals.demeaner_options import (
        _warn_if_deprecated_demeaner_backend,
    )

    with pytest.warns(
        DeprecationWarning, match=r"`scipy` LSMR demeaner backend"
    ) as rec:
        _warn_if_deprecated_demeaner_backend(
            pf.LsmrDemeaner(backend="cupy", device="cpu")
        )
    assert any("default within backend" in str(r.message) for r in rec)


@pytest.mark.parametrize(
    "builder, invalid_name",
    [
        (lambda: pf.MapDemeaner(fixef_tol=1.0), "fixef_tol"),
        (lambda: pf.LsmrDemeaner(fixef_atol=1.0), "fixef_atol"),
        (lambda: pf.LsmrDemeaner(fixef_btol=1.0), "fixef_btol"),
    ],
)
def test_typed_demeaners_reject_tolerances_ge_one(builder, invalid_name):
    with pytest.raises(ValueError, match=invalid_name):
        builder()


@pytest.mark.parametrize("requested", ["auto", "off", "additive", "diagonal"])
def test_lsmr_demeaner_accepts_within_preconditioner_strings(requested):
    """All four documented string options round-trip through ``LsmrDemeaner``."""
    demeaner = pf.LsmrDemeaner(preconditioner=requested)
    assert demeaner.backend == "within"
    assert demeaner.preconditioner == requested


@pytest.mark.parametrize(
    ("bad", "exc", "match"),
    [
        ("bogus", ValueError, "`preconditioner`"),
        (42, TypeError, "Preconditioner"),
        (object(), TypeError, "Preconditioner"),
    ],
)
def test_lsmr_demeaner_rejects_invalid_preconditioner(bad, exc, match):
    """Unknown strings raise ValueError; non-string/non-Preconditioner raises TypeError."""
    with pytest.raises(exc, match=match):
        pf.LsmrDemeaner(preconditioner=bad)  # type: ignore[arg-type]


def _glm_binary_data():
    data = pf.get_data()
    data["Y_bin"] = (data["Y"] > data["Y"].median()).astype(int)
    return data


def _feglm_logit(data, demeaner):
    return pf.feglm(
        "Y_bin ~ X1 | f1 + f2", data=data, family="logit", demeaner=demeaner
    )


_REUSE_ESTIMATORS = [
    pytest.param(
        lambda: pf.get_data(),
        lambda data, demeaner: pf.feols(
            "Y ~ X1 | f1 + f2", data=data, demeaner=demeaner
        ),
        id="feols",
    ),
    pytest.param(
        lambda: pf.get_data(model="Fepois"),
        lambda data, demeaner: pf.fepois(
            "Y ~ X1 | f1 + f2", data=data, demeaner=demeaner
        ),
        id="fepois",
    ),
    pytest.param(_glm_binary_data, _feglm_logit, id="feglm"),
]


@pytest.mark.parametrize(("data_fn", "fit_fn"), _REUSE_ESTIMATORS)
def test_within_preconditioner_reuse_across_estimators(data_fn, fit_fn):
    """Every Feols-family estimator exposes a Preconditioner that round-trips.

    Pinned down for ``feols``, ``fepois`` (IWLS), and ``feglm`` (IWLS):
    - the fit exposes a non-None ``Preconditioner`` after a 2-FE solve, and
    - feeding it back via ``LsmrDemeaner(preconditioner=...)`` reproduces the
      original coefficients to machine precision.
    """
    data = data_fn()
    fit = fit_fn(data, pf.LsmrDemeaner(backend="within"))

    pre = fit.preconditioner
    assert isinstance(pre, pf.Preconditioner)

    fit_reused = fit_fn(data, pf.LsmrDemeaner(backend="within", preconditioner=pre))

    # Load-bearing correctness check: the reused factorization must produce
    # the same solve as the original.
    np.testing.assert_allclose(fit_reused.coef(), fit.coef(), rtol=1e-10, atol=1e-10)
    assert isinstance(fit_reused.preconditioner, pf.Preconditioner)
    assert fit_reused.preconditioner.variant == pre.variant
    assert fit_reused.preconditioner.nrows == pre.nrows


def test_fixest_multi_shares_preconditioners_by_na_index():
    data = pf.get_data().copy()
    data["X3"] = data["X2"] ** 2 + 0.1 * data["X1"]
    complete = data[["Y", "X1", "X2", "X3", "f1", "f2"]].notna().all(axis=1)
    extra_na_index = data.index[complete][0]
    data.loc[extra_na_index, "X3"] = np.nan

    fit = pf.feols(
        "Y ~ csw(X1, X2, X3) | f1 + f2",
        data=data,
        demeaner=pf.LsmrDemeaner(backend="within"),
    )
    models = list(fit.all_fitted_models.values())

    assert len(models) == 3
    assert len({id(model._demean_cache.lookup_preconditioner) for model in models}) == 1

    first, second, third = models
    assert first._na_index == second._na_index
    assert extra_na_index not in first._na_index
    assert extra_na_index in third._na_index

    assert isinstance(first.preconditioner, pf.Preconditioner)
    assert first.preconditioner is second.preconditioner
    assert isinstance(third.preconditioner, pf.Preconditioner)
    assert third.preconditioner is not first.preconditioner
    assert set(first._demean_cache.lookup_preconditioner) == {
        first._na_index,
        third._na_index,
    }


def test_within_preconditioner_off_yields_no_cached_preconditioner():
    """``preconditioner='off'`` disables Schwarz, so ``fit.preconditioner`` is None."""
    data = pf.get_data()
    fit = pf.feols(
        "Y ~ X1 | f1 + f2",
        data=data,
        demeaner=pf.LsmrDemeaner(backend="within", preconditioner="off"),
    )
    assert fit.preconditioner is None


@pytest.mark.parametrize("backend", ["torch"])
def test_preconditioner_rejected_on_non_within_backend(backend):
    """A built ``Preconditioner`` is only valid for ``backend='within'``."""
    data = pf.get_data()
    pre = pf.feols(
        "Y ~ X1 | f1 + f2",
        data=data,
        demeaner=pf.LsmrDemeaner(backend="within"),
    ).preconditioner
    assert pre is not None
    with pytest.raises(ValueError, match="Preconditioner"):
        pf.LsmrDemeaner(backend=backend, preconditioner=pre)


def test_feiv_first_stage_reuses_within_preconditioner():
    data = pf.get_data()
    fit = pf.feols(
        "Y ~ 1 | f1 + f2 | X1 ~ Z1",
        data=data,
        demeaner=pf.LsmrDemeaner(backend="within"),
    )

    preconditioner = fit.preconditioner
    assert isinstance(preconditioner, pf.Preconditioner)
    assert isinstance(fit._model_1st_stage._demeaner, pf.LsmrDemeaner)
    # The 1st-stage demeaner's config stores the 2nd-stage's preconditioner
    # verbatim (identity preserved on assignment).
    assert fit._model_1st_stage._demeaner.preconditioner is preconditioner
    # The 1st-stage model's preconditioner is what came back from the solve;
    # a fresh pyo3 wrapper around the same factorization (identity differs;
    # value semantics match upstream — compare structurally).
    assert isinstance(fit._model_1st_stage.preconditioner, pf.Preconditioner)
    assert fit._model_1st_stage.preconditioner.variant == preconditioner.variant
    assert fit._model_1st_stage.preconditioner.nrows == preconditioner.nrows


def test_lean():
    data = pf.get_data()
    fit = pf.feols("Y ~ X1 + X2 | f1", data=data, lean=True)

    assert not hasattr(fit, "_data")
    assert not hasattr(fit, "_X")
    assert not hasattr(fit, "_Y")


def test_duckdb_input():
    data_pandas = pf.get_data()
    data_duckdb = duckdb.query("SELECT * FROM data_pandas")
    fit_pandas = pf.feols("Y ~ X1 | f1 + f2", data=data_pandas)
    fit_duckdb = pf.feols("Y ~ X1 | f1 + f2", data=data_duckdb)
    assert type(fit_pandas) is type(fit_duckdb)
    np.testing.assert_allclose(fit_pandas.coef(), fit_duckdb.coef(), rtol=1e-12)
    np.testing.assert_allclose(fit_pandas.se(), fit_duckdb.se(), rtol=1e-12)


def _lspline(series: pd.Series, knots: list[float]) -> np.array:
    """Generate a linear spline design matrix for the input series based on knots."""
    vector = series.values
    columns = []

    for i, knot in enumerate(knots):
        column = np.minimum(vector, knot if i == 0 else knot - knots[i - 1])
        columns.append(column)
        vector = vector - column

    # Add the remainder as the last column
    columns.append(vector)

    # Combine columns into a design matrix
    return np.column_stack(columns)


@pytest.fixture
def spline_data():
    """Fixture to prepare data with spline splits."""
    data = pf.get_data()
    data["Y"] = np.where(data["Y"] > data["Y"].median(), 1, 0)
    spline_split = _lspline(data["X2"], [0, 1])
    data["X2_0"], data["0_X2_1"], data["1_X2"] = spline_split.T
    return data


@pytest.mark.parametrize(
    "method,family",
    [
        ("feols", None),
        ("feglm", "logit"),
        ("feglm", "probit"),
        ("feglm", "gaussian"),
    ],
)
def test_context_capture(spline_data, method, family):
    method_kwargs = {"data": spline_data}
    if family:
        method_kwargs["family"] = family

    explicit_fit = getattr(pf, method)("Y ~ X2_0 + 0_X2_1 + 1_X2", **method_kwargs)
    context_captured_fit = getattr(pf, method)(
        "Y ~ _lspline(X2,[0,1])", context=0, **method_kwargs
    )
    context_captured_fit_map = getattr(pf, method)(
        "Y ~ _lspline(X2,[0,1])", context={"_lspline": _lspline}, **method_kwargs
    )

    for context_fit in [context_captured_fit, context_captured_fit_map]:
        np.testing.assert_allclose(context_fit.coef(), explicit_fit.coef(), rtol=1e-12)
        np.testing.assert_allclose(context_fit.se(), explicit_fit.se(), rtol=1e-12)

    # FactorEvaluationError for `feols` when context is not set
    if method == "feols":
        with pytest.raises(
            FactorEvaluationError, match="Unable to evaluate factor `_lspline"
        ):
            pf.feols("Y ~ _lspline(X2,[0,1]) | f1 + f2", data=spline_data)
