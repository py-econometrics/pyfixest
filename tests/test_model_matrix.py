import numpy as np
import pandas as pd
import pytest

from pyfixest.estimation.formula import FORMULAIC_TRANSFORMS
from pyfixest.estimation.formula.model_matrix import (
    _get_formulaic_formula,
    create_model_matrix,
)
from pyfixest.estimation.formula.parse import Formula


@pytest.fixture
def varying_slope_data():
    return pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [4.0, 5.0, 6.0, 7.0],
            "f1": ["a", "b", "a", "b"],
            "f2": [1, 1, 2, 2],
            "z1": [0.1, 0.2, 0.3, 0.4],
            "z2": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.mark.parametrize(
    "fixed_effects,expected_levels,expected_slopes",
    [
        ("f1", ["__fixed_effect__(f1)"], []),
        ("f1[z1]", ["__fixed_effect__(f1)"], ["z1"]),
        (
            "f1[z1, z2]",
            ["__fixed_effect__(f1)"],
            ["z1", "z2"],
        ),
        (
            "f1[[z1, z2]]",
            ["__fixed_effect__(f1)"],
            ["z1", "z2"],
        ),
        (
            "f1:f2[z1]",
            ["__fixed_effect__(f1, f2)"],
            ["z1"],
        ),
    ],
)
def test_materializes_fixed_effect_specifications(
    varying_slope_data, fixed_effects, expected_levels, expected_slopes
):
    formula = Formula.parse(f"y ~ x | {fixed_effects}")[0]

    model_matrix = create_model_matrix(formula, varying_slope_data)
    formulaic_matrix = _get_formulaic_formula(
        formula, varying_slope_data
    ).get_model_matrix(
        varying_slope_data,
        output="pandas",
        context=FORMULAIC_TRANSFORMS,
    )

    assert model_matrix.fixed_effects.columns.tolist() == expected_levels
    assert formulaic_matrix["fe_slopes"].columns.tolist() == expected_slopes


def test_shared_level_and_slope_columns_are_materialized_once(varying_slope_data):
    formula = Formula.parse("y ~ x | f1[z1] + f2[z1] + f1[[z2]]")[0]

    model_matrix = create_model_matrix(formula, varying_slope_data)
    formulaic_matrix = _get_formulaic_formula(
        formula, varying_slope_data
    ).get_model_matrix(
        varying_slope_data,
        output="pandas",
        context=FORMULAIC_TRANSFORMS,
    )

    assert model_matrix.fixed_effects.columns.tolist() == [
        "__fixed_effect__(f1)",
        "__fixed_effect__(f2)",
    ]
    assert formulaic_matrix["fe_slopes"].columns.tolist() == ["z1", "z2"]


def test_materializes_multicolumn_slope_terms(varying_slope_data):
    varying_slope_data["g"] = ["a", "b", "c", "a"]
    formula = Formula.parse("y ~ x | f1 + f1[[z1, C(g)]] + f2[z2] + f1[[z1]]")[0]

    model_matrix = create_model_matrix(formula, varying_slope_data)

    assert model_matrix.fixed_effect_slopes.columns.tolist() == [
        "z1",
        "C(g)[a]",
        "C(g)[b]",
        "C(g)[c]",
        "z2",
    ]


def test_slope_terms_share_missing_row_handling(varying_slope_data):
    varying_slope_data.loc[1, "z1"] = np.nan
    formula = Formula.parse("y ~ x | f1[z1] + f2")[0]

    model_matrix = create_model_matrix(formula, varying_slope_data)

    expected_index = pd.Index([0, 2, 3])
    assert model_matrix.dependent.index.equals(expected_index)
    assert model_matrix.fixed_effects.index.equals(expected_index)
    assert model_matrix.fixed_effect_slopes.index.equals(expected_index)


def test_slope_only_effect_retains_global_intercept(varying_slope_data):
    slope_only = create_model_matrix(
        Formula.parse("y ~ x | f1[[z1]]")[0], varying_slope_data
    )
    with_intercept = create_model_matrix(
        Formula.parse("y ~ x | f1[z1]")[0], varying_slope_data
    )

    assert "Intercept" in slope_only.independent
    assert "Intercept" not in with_intercept.independent


def test_formulaic_materializes_slope_transforms(varying_slope_data):
    formula = Formula.parse("y ~ x | f1[log(z2)]")[0]

    formulaic_matrix = _get_formulaic_formula(
        formula, varying_slope_data
    ).get_model_matrix(
        varying_slope_data,
        output="pandas",
        context=FORMULAIC_TRANSFORMS,
    )

    np.testing.assert_allclose(
        formulaic_matrix["fe_slopes"].iloc[:, 0],
        np.log(varying_slope_data["z2"]),
    )
