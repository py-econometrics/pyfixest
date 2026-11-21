from functools import partial

import numpy as np
import pandas as pd
import pytest

from pyfixest.estimation import feglm, feols, fepois


@pytest.fixture(scope="module")
def data_nested_fe() -> pd.DataFrame:
    """Create data where unit_fixed_effect is nested within unit_id (collinear with FE)."""
    rng = np.random.default_rng(42)
    n = 200
    units = 20
    unit_id = np.repeat(range(units), n // units)
    # unit_fixed_effect is nested within unit_id (each unit has one value)
    unit_fixed_effect = np.array([i % 3 for i in unit_id])
    year = rng.choice(range(2015, 2020), size=n)
    independent = rng.choice([0, 1], size=n)
    linear_pred = 0.5 * independent - 0.3 * unit_fixed_effect + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "Y": rng.poisson(np.exp(linear_pred)),
            "independent": independent,
            "unit_fixed_effect": unit_fixed_effect,
            "unit_id": unit_id,
            "year": year,
        }
    )


def test_multicollinearity_error():
    rng = np.random.default_rng(4)

    N = 10000
    X1 = rng.normal(0, 1, N)
    Y = rng.normal(0, 1, N)
    f1 = rng.choice([0, 1, 2, 3, 4], N, True)
    f2 = f1.copy()
    f3 = f1.copy()
    f3 = np.where(f3 == 1, 0, f3)

    data = pd.DataFrame(
        {
            "Y": Y,
            "X1": X1,
            "X2": X1 + rng.normal(0, 0.00000000001, N),
            "f1": f1,
            "f2": f2,
            "f3": f3,
        }
    )

    fit = feols("Y ~ X1 + X2", data=data)
    assert fit._coefnames == ["Intercept", "X1"]

    fit = feols("Y ~ X1 + f1 + X2 + f2", data=data)
    assert fit._coefnames == ["Intercept", "X1", "f1"]

    fit = feols("Y ~ X1 + f1 + X2 | f2", data=data)
    assert fit._coefnames == ["X1"]


def test_weighted_iv_drops_controls_and_instruments_consistently():
    data = pd.DataFrame(
        {
            "group": [1] * 4 + [2] * 4,
            "x": [0, 0, 0, 0, 1, 1, -1, -1],
            "z1": [1, -1, 1, -1, 0, 0, 0, 0],
            "z2": [0, 0, 0, 0, 1, -1, 1, -1],
            "d": [3, -1, 1, -3, 3, -1, 1, -3],
            "y": [5, -1, 3, -7, 6, -2, 2, -6],
            "w": np.repeat([1.0, 1e-12], 4),
        }
    )
    with pytest.warns(UserWarning, match="variables dropped due to multicollinearity"):
        fit = feols("y ~ x + [d ~ z1 + z2] | group", data, weights="w")
    reference = feols("y ~ 1 + [d ~ z1] | group", data, weights="w")
    assert fit.collinearity.dropped_coef_names == ("x",)
    assert fit.collinearity_instruments.dropped_coef_names == ("x", "z2")
    assert fit.first_stage.model.collinearity.dropped_coef_names == ("x", "z2")
    assert fit.first_stage.instruments == ("z1",)
    np.testing.assert_allclose(
        fit.coef(), reference.coef(), err_msg="Weighted IV coefficients"
    )


@pytest.mark.parametrize(
    "fml",
    ["Y ~ C(independent)*C(unit_fixed_effect) | unit_id + year"],
)
@pytest.mark.parametrize(
    "model",
    [feols, fepois, partial(feglm, family="gaussian")],
    ids=["feols", "fepois", "feglm-gaussian"],
)
def test_multicollinearity_with_fixed_effects(data_nested_fe, model, fml):
    """Test that models drop covariates collinear with fixed effects.

    This tests the case where a covariate is nested within a fixed effect
    (e.g., unit_fixed_effect is constant within each unit_id), making it
    collinear with the fixed effects.
    """
    fit = model(fml=fml, data=data_nested_fe)

    # The main effect terms of unit_fixed_effect should be dropped
    # as they are collinear with the unit_id fixed effects
    assert "C(unit_fixed_effect)[T.1]" in fit.collinearity.dropped_coef_names
    assert "C(unit_fixed_effect)[T.2]" in fit.collinearity.dropped_coef_names
