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


def test_varying_slopes_boolean_slope_equals_zero_one_slope(data):
    """A boolean slope is treated as its 0/1 coding."""
    data = data.assign(b=data["X2"] > 0)
    data["b_float"] = data["b"].astype(float)

    fit_bool = pf.feols("Y ~ X1 | f1[b]", data=data)
    fit_float = pf.feols("Y ~ X1 | f1[b_float]", data=data)

    np.testing.assert_allclose(fit_bool.coef(), fit_float.coef(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(fit_bool.se(), fit_float.se(), rtol=0, atol=1e-12)
