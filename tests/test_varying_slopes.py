"""Support boundaries of varying-slope fixed effects such as `f1[z]`.

Numerical results are compared with R fixest in
`tests/test_varying_slopes_vs_fixest.py`.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.errors import FormulaSyntaxError, VcovTypeNotSupportedError


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pf.get_data(N=500, seed=2026)


@pytest.fixture(scope="module", params=["feols", "fepois"])
def multiple_estimation_data(request):
    """Create estimator-specific data shared by multiple-estimation cases."""
    if request.param == "feols":
        data = pf.get_data(N=1000, seed=76540251, beta_type="2", error_type="2")
        data.loc[4, "Z1"] = np.nan
        fit_model = pf.feols
        kwargs = {}
    else:
        data = pf.get_data(
            N=1000, seed=7651, beta_type="2", error_type="2", model="Fepois"
        )
        fit_model = pf.fepois
        kwargs = {"iwls_tol": 1e-10, "iwls_maxiter": 100}
    data.loc[[3, 17], "X2"] = np.nan
    return data, fit_model, kwargs


@pytest.mark.parametrize(
    "demeaner",
    [pf.MapDemeaner(), pf.LsmrDemeaner(backend="torch")],
    ids=["map", "torch"],
)
@pytest.mark.parametrize("estimator", [pf.feols, pf.fepois])
def test_varying_slopes_require_within_demeaner(data, demeaner, estimator):
    count_data = data.assign(Y=data.Y.abs().round())
    with pytest.raises(ValueError, match=r"require `LsmrDemeaner\(backend='within'\)`"):
        estimator("Y ~ X1 | f1[X2]", data=count_data, demeaner=demeaner)


@pytest.mark.parametrize(
    "call",
    [
        lambda fit: fit.predict(),
        lambda fit: fit.fixef(),
        lambda fit: fit.decompose(param="X1", x1_vars=["X2"]),
        lambda fit: fit.ritest(resampvar="X1", reps=10),
        lambda fit: fit.wildboottest(param="X1", reps=9),
    ],
    ids=["predict", "fixef", "decompose", "ritest", "wildboottest"],
)
def test_varying_slopes_reject_unsupported_post_estimation(data, call):
    fit = pf.feols("Y ~ X1 + X2 | f1[Z1]", data=data)

    with pytest.raises(NotImplementedError, match=r"(fits|models) with varying slopes"):
        call(fit)


@pytest.mark.parametrize("slope", ["C(f2)", "f2_str"])
def test_varying_slopes_reject_categorical_slopes(data, slope):
    data = data.assign(f2_str=data["f2"].astype(str))

    with pytest.raises(
        FormulaSyntaxError, match=rf"`{re.escape(slope)}` must evaluate to one numeric"
    ):
        pf.feols(f"Y ~ X1 | f1[{slope}]", data=data)


@pytest.mark.parametrize(
    ("vcov", "vcov_kwargs", "error"),
    [
        ({"CRV3": "f1"}, None, VcovTypeNotSupportedError),
        ("NW", {"time_id": "year", "panel_id": "unit", "lag": 1}, NotImplementedError),
    ],
    ids=["CRV3", "NW"],
)
def test_varying_slopes_reject_unsupported_vcov(data, vcov, vcov_kwargs, error):
    panel = data.assign(unit=data.f1, year=data.groupby("f1").cumcount())

    with pytest.raises(error, match="fits with varying slopes"):
        pf.feols("Y ~ X1 | f1[Z1]", data=panel, vcov=vcov, vcov_kwargs=vcov_kwargs)


@pytest.mark.parametrize("retention", [{"lean": True}, {"store_data": False}])
def test_varying_slopes_with_reduced_retention(data, retention):
    full = pf.feols("Y ~ X1 | f1[X2]", data=data, vcov={"CRV1": "f2"})
    reduced = pf.feols("Y ~ X1 | f1[X2]", data=data, vcov={"CRV1": "f2"}, **retention)

    np.testing.assert_allclose(reduced.coef(), full.coef(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(reduced.se(), full.se(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("slope_type", ["float", "int"])
@pytest.mark.parametrize("level_type", ["float", "str", "categorical"])
def test_varying_slopes_column_type_invariance(data, level_type, slope_type):
    """Equivalent level and slope values give the same fit across dtypes."""
    data = data.dropna().reset_index(drop=True)
    levels = data["f1"].astype(int)
    slopes = np.round(3 * data["X2"]).astype(int)
    baseline_data = data.assign(f1=levels, s=slopes)
    variant_data = baseline_data.assign(
        f1={
            "float": levels.astype(float),
            "str": levels.astype(str),
            "categorical": pd.Categorical(levels),
        }[level_type],
        s=slopes.astype(slope_type),
    )
    fml = "Y ~ X1 | f1[s] + f2[[s]]"

    baseline = pf.feols(fml, data=baseline_data)
    variant = pf.feols(fml, data=variant_data)

    assert variant.coef().index.equals(baseline.coef().index)
    assert variant.sample_info.n_obs == baseline.sample_info.n_obs
    assert variant.variance_covariance.df_k == baseline.variance_covariance.df_k
    np.testing.assert_allclose(variant.coef(), baseline.coef(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(variant.se(), baseline.se(), rtol=0, atol=1e-10)


@pytest.mark.parametrize(
    ("fml", "single_fmls"),
    [
        (
            "Y ~ X1 | sw(f1, f1[X2], f1[[X2]], f1[X2, Z1])",
            [
                "Y ~ X1 | f1",
                "Y ~ X1 | f1[X2]",
                "Y ~ X1 | f1[[X2]]",
                "Y ~ X1 | f1[X2, Z1]",
            ],
        ),
        (
            "Y ~ X1 | csw0(f1[[X2]], f2)",
            ["Y ~ X1", "Y ~ X1 | f1[[X2]]", "Y ~ X1 | f1[[X2]] + f2"],
        ),
        (
            "Y ~ X1 | csw(f1, f1[X2])",
            ["Y ~ X1 | f1", "Y ~ X1 | f1[X2]"],
        ),
        (
            "Y ~ X1 | csw(f1[[X2]], f1)",
            ["Y ~ X1 | f1[[X2]]", "Y ~ X1 | f1[X2]"],
        ),
        (
            "Y ~ X1 | csw(f1[X2], f1[Z1])",
            ["Y ~ X1 | f1[X2]", "Y ~ X1 | f1[X2, Z1]"],
        ),
        ("Y ~ sw(X1, X1 + Z1) | f1[X2]", ["Y ~ X1 | f1[X2]", "Y ~ X1 + Z1 | f1[X2]"]),
    ],
)
def test_varying_slopes_multiple_estimation_matches_single_fits(
    fml, single_fmls, multiple_estimation_data
):
    """Multiple fits match single fits, including slopes sharing group levels."""
    data, fit_model, kwargs = multiple_estimation_data
    multi = fit_model(fml, data=data, **kwargs)
    singles = [fit_model(single, data=data, **kwargs) for single in single_fmls]

    for fit, single in zip(multi.to_list(), singles, strict=True):
        assert fit.coef().index.equals(single.coef().index)
        assert fit.sample_info.n_obs == single.sample_info.n_obs
        assert fit.variance_covariance.df_k == single.variance_covariance.df_k
        np.testing.assert_allclose(fit.coef(), single.coef(), rtol=0, atol=1e-8)
        np.testing.assert_allclose(fit.se(), single.se(), rtol=0, atol=1e-7)


def test_varying_slopes_boolean_slope_equals_zero_one_slope(data):
    """A boolean slope is treated as its 0/1 coding."""
    data = data.assign(b=data["X2"] > 0)
    data["b_float"] = data["b"].astype(float)

    fit_bool = pf.feols("Y ~ X1 | f1[b]", data=data)
    fit_float = pf.feols("Y ~ X1 | f1[b_float]", data=data)

    np.testing.assert_allclose(fit_bool.coef(), fit_float.coef(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(fit_bool.se(), fit_float.se(), rtol=0, atol=1e-12)
