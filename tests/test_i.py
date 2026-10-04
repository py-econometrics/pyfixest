"""
Comprehensive tests for pyfixest i() syntax.

Tests cover:
- Simple i(var) with different factor types
- Factor x Continuous: i(var, continuous)
- Factor x Factor: i(var1, var2)
- Binning: bin and bin2 parameters
- Intercept control: 0+, -1, 1+ syntax
- Fixed effects combinations
- Multiple i() terms
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pyfixest.estimation import feols

fixest = importr("fixest")
stats = importr("stats")

# Tolerances for coefficient comparison
RTOL = 1e-5
ATOL = 1e-8


# =============================================================================
# Helper Functions
# =============================================================================


def normalize_coef_name(name: str) -> str:
    """Normalize coefficient name for comparison between R and Python."""
    name = str(name)
    # R uses (Intercept), Python uses Intercept
    if name == "(Intercept)":
        return "Intercept"

    # Normalize float formatting in factor levels (1.0 vs 1)
    def normalize_float_level(match):
        prefix = match.group(1)
        num = float(match.group(2))
        suffix = match.group(3) or ""
        if num == int(num):
            return f"{prefix}{int(num)}{suffix}"
        return match.group(0)

    name = re.sub(r"(::)(\d+\.0)(\b|:)", normalize_float_level, name)
    return name


def get_r_coef_names(fit_r) -> list[str]:
    """Extract coefficient names from R fixest fit."""
    ro.globalenv["fit_tmp"] = fit_r
    names = ro.r("names(coef(fit_tmp))")
    ro.r("rm(fit_tmp)")
    if names is ro.NULL or names is None:
        return []
    return [normalize_coef_name(n) for n in names]


def get_r_coef_values(fit_r) -> np.ndarray:
    """Extract coefficient values from R fixest fit."""
    ro.globalenv["fit_tmp"] = fit_r
    coefs = ro.r("as.numeric(coef(fit_tmp))")
    ro.r("rm(fit_tmp)")
    return np.array(coefs)


def assert_models_match(
    py_names: list[str],
    py_values: np.ndarray,
    r_names: list[str],
    r_values: np.ndarray,
    check_names: bool = True,
) -> None:
    """Assert pyfixest and R fixest models match."""
    assert len(py_names) == len(r_names), (
        f"Coefficient count mismatch: py={len(py_names)}, r={len(r_names)}"
    )
    if check_names:
        assert py_names == r_names, f"Name mismatch:\n  py={py_names}\n  r={r_names}"
    np.testing.assert_allclose(py_values, r_values, rtol=RTOL, atol=ATOL)


def compare_with_r(
    r_fml: str, df: pd.DataFrame, py_fml: str | None = None
) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """
    Compare pyfixest and R fixest models.

    Returns (py_names, py_values, r_names, r_values).
    """
    py_formula = py_fml if py_fml is not None else r_fml
    fit_py = feols(py_formula, df)
    py_names = [normalize_coef_name(str(n)) for n in fit_py._coefnames]
    py_values = fit_py.coef().values

    fit_r = fixest.feols(ro.Formula(r_fml), df)
    r_names = get_r_coef_names(fit_r)
    r_values = get_r_coef_values(fit_r)

    return py_names, py_values, r_names, r_values


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def df_het() -> pd.DataFrame:
    """Load heterogeneous treatment effects data."""
    np.random.seed(123)
    df = pd.read_csv("pyfixest/did/data/df_het.csv")
    df["X"] = np.random.normal(size=len(df))
    return df


@pytest.fixture(scope="module")
def df_test() -> pd.DataFrame:
    """Create test data with various factor types."""
    np.random.seed(42)
    n = 200

    return pd.DataFrame(
        {
            "Y": np.random.randn(n),
            "X1": np.random.randn(n),
            "X2": np.random.randn(n),
            # String factor
            "f_str": np.random.choice(["apple", "banana", "cherry"], n),
            # Integer factor
            "f_int": np.random.choice([1, 2, 3, 10, 20], n),
            # Float factor
            "f_float": np.random.choice([1.0, 2.0, 3.0], n),
            # Second string factor for interactions
            "g": np.random.choice(["X", "Y", "Z"], n),
            # Fixed effects
            "fe1": np.random.choice(range(10), n),
            "fe2": np.random.choice(range(5), n),
        }
    )


# =============================================================================
# Basic i() Tests (existing)
# =============================================================================


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "formula,excluded_coef",
    [
        ("dep_var ~ i(rel_year, ref=1.0)", "rel_year::1"),
        ("dep_var ~ i(rel_year, ref=-2.0)", "rel_year::-2"),
        ("dep_var ~ i(rel_year, treat, ref=1.0)", "rel_year::1:treat"),
        ("dep_var ~ i(rel_year, treat, ref=-2.0)", "rel_year::-2:treat"),
    ],
)
def test_i_reference_exclusion(df_het, formula, excluded_coef):
    """Test that reference levels are properly excluded."""
    fit = feols(formula, df_het)
    assert excluded_coef not in fit._coefnames, (
        f"{excluded_coef} should not be in coefficient names"
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "fml",
    [
        "dep_var ~ i(state)",
        "dep_var ~ i(state, ref = 1)",
        "dep_var ~ i(state, year)",
        "dep_var ~ i(state, year, ref = 1)",
        "dep_var ~ i(state, year) | state",
        "dep_var ~ i(state, year, ref = 1) | state",
    ],
)
def test_i_vs_fixest(fml):
    """Test i() against R fixest."""
    df = pd.read_csv("pyfixest/did/data/df_het.csv")
    df["X"] = np.random.normal(df.shape[0])

    fit_py = feols(fml, df)
    fit_r = fixest.feols(ro.Formula(fml), df)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )


# =============================================================================
# Intercept Control Tests (0+, -1, 1+)
# =============================================================================


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ 0 + i(f_str)",  # No intercept, keep all levels
        "Y ~ -1 + i(f_str)",  # Same as 0 +
        "Y ~ i(f_str) - 1",  # Alternative syntax
    ],
)
def test_no_intercept_all_levels(df_test, fml):
    """Test that without intercept, all levels are kept."""
    py_names, py_values, r_names, r_values = compare_with_r(fml, df_test)
    assert_models_match(py_names, py_values, r_names, r_values)


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ 0 + i(f_str, ref='apple')",  # No intercept + explicit ref
        "Y ~ -1 + i(f_str, ref='banana')",  # Same with different ref
    ],
)
def test_no_intercept_with_ref(df_test, fml):
    """Test no intercept with explicit reference level."""
    py_names, py_values, r_names, r_values = compare_with_r(fml, df_test)
    assert_models_match(py_names, py_values, r_names, r_values)


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ 1 + i(f_str)",  # With intercept, drop first level
        "Y ~ i(f_str)",  # Same (intercept implicit)
    ],
)
def test_with_intercept_drop_level(df_test, fml):
    """Test that with intercept, first level is dropped."""
    py_names, py_values, r_names, r_values = compare_with_r(fml, df_test)
    assert_models_match(py_names, py_values, r_names, r_values)


# =============================================================================
# Binning Tests
# =============================================================================


@pytest.mark.against_r_core
def test_binning_simple(df_test):
    """Test i() with bin parameter."""
    r_fml = "Y ~ i(f_str, bin=list(fruit=c('apple','banana')))"
    py_fml = "Y ~ i(f_str, bin={'fruit': ['apple','banana']})"
    py_names, py_values, r_names, r_values = compare_with_r(r_fml, df_test, py_fml)
    assert_models_match(py_names, py_values, r_names, r_values)


@pytest.mark.against_r_core
def test_binning_with_ref(df_test):
    """Test i() with bin and ref parameters."""
    r_fml = "Y ~ i(f_str, bin=list(fruit=c('apple','banana')), ref='fruit')"
    py_fml = "Y ~ i(f_str, bin={'fruit': ['apple','banana']}, ref='fruit')"
    py_names, py_values, r_names, r_values = compare_with_r(r_fml, df_test, py_fml)
    assert_models_match(py_names, py_values, r_names, r_values)


@pytest.mark.against_r_core
def test_binning_with_continuous(df_test):
    """Test i() with bin parameter and continuous interaction."""
    r_fml = "Y ~ i(f_str, X1, bin=list(fruit=c('apple','banana')))"
    py_fml = "Y ~ i(f_str, X1, bin={'fruit': ['apple','banana']})"
    py_names, py_values, r_names, r_values = compare_with_r(r_fml, df_test, py_fml)
    assert_models_match(py_names, py_values, r_names, r_values)


# =============================================================================
# Factor x Factor Tests
# =============================================================================


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "r_fml,py_fml",
    [
        ("Y ~ i(f_str, i.g)", "Y ~ i(f_str, g)"),
        ("Y ~ i(f_str, i.g, ref='apple')", "Y ~ i(f_str, g, ref='apple')"),
        (
            "Y ~ i(f_str, i.g, ref='apple', ref2='X')",
            "Y ~ i(f_str, g, ref='apple', ref2='X')",
        ),
    ],
)
def test_factor_x_factor(df_test, r_fml, py_fml):
    """Test i(factor1, factor2) interactions."""
    py_names, py_values, r_names, r_values = compare_with_r(r_fml, df_test, py_fml)
    assert_models_match(py_names, py_values, r_names, r_values)


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "r_fml,py_fml",
    [
        ("Y ~ i(f_str, i.g) | fe1", "Y ~ i(f_str, g) | fe1"),
        (
            "Y ~ i(f_str, i.g, ref='apple', ref2='X') | fe1",
            "Y ~ i(f_str, g, ref='apple', ref2='X') | fe1",
        ),
    ],
)
def test_factor_x_factor_with_fe(df_test, r_fml, py_fml):
    """Test i(factor1, factor2) with fixed effects."""
    py_names, py_values, r_names, r_values = compare_with_r(r_fml, df_test, py_fml)
    assert_models_match(py_names, py_values, r_names, r_values)


@pytest.mark.against_r_core
@pytest.mark.parametrize("with_fe", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("vcov", ["iid", "hetero", "cluster"])
def test_factor_x_factor_empty_first_cell(with_fe, weighted, vcov):
    """Accept a different default reference, verified against fixest 0.14.0."""
    rng = np.random.default_rng(1768)
    data = pd.DataFrame(
        [(a, b) for a in ["a1", "a2", "a3"] for b in ["b1", "b2", "b3", "b4"]] * 25,
        columns=["a", "b"],
    )
    data = data.loc[~((data.a == "a1") & (data.b == "b1"))].reset_index(drop=True)
    n = len(data)
    data["X1"] = rng.normal(size=n)
    data["Y"] = data.X1 + (data.a == "a2") + rng.normal(size=n)
    data["fe1"] = rng.integers(0, 10, size=n)
    data["fe2"] = rng.integers(0, 6, size=n)
    data["weights"] = rng.uniform(0.5, 2.0, size=n)
    suffix = " | fe1" if with_fe else ""
    py_vcov = {"CRV1": "fe2"} if vcov == "cluster" else vcov
    r_vcov = ro.Formula("~fe2") if vcov == "cluster" else vcov
    weights = "weights" if weighted else None

    with pytest.warns(UserWarning, match="a::a3:b::b4"):
        fit_py = feols(
            fml="Y ~ X1 + i(a, b)" + suffix,
            data=data,
            vcov=py_vcov,
            weights=weights,
        )
    fit_r = fixest.feols(
        ro.Formula("Y ~ X1 + i(a, i.b)" + suffix),
        data,
        vcov=r_vcov,
        weights=ro.Formula("~weights") if weighted else ro.NULL,
    )
    py_names = list(fit_py.coef().index)
    r_names = get_r_coef_names(fit_r)
    py_ref, r_ref = "a::a3:b::b4", "a::a1:b::b2"
    assert set(py_names) - set(r_names) == {r_ref}
    assert set(r_names) - set(py_names) == {py_ref}
    assert fit_py.collinearity.dropped_coef_names == (py_ref,)

    # Change from pyfixest's omitted cell to fixest's: subtract the new
    # reference from every cell coefficient and add it to the intercept.
    # With fixed effects the common shift is absorbed by their normalization.
    transform = np.zeros((len(r_names), len(py_names)))
    for row, name in enumerate(r_names):
        if name in py_names:
            transform[row, py_names.index(name)] = 1
        if name == "Intercept":
            transform[row, py_names.index(r_ref)] += 1
        elif name.startswith("a::"):
            transform[row, py_names.index(r_ref)] -= 1

    # Strict OLS bounds, allowing demeaning roundoff with fixed effects.
    np.testing.assert_allclose(
        transform @ fit_py.coef().to_numpy(),
        get_r_coef_values(fit_r),
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        transform @ fit_py.variance_covariance.vcov @ transform.T,
        np.asarray(stats.vcov(fit_r)),
        atol=1e-7,
        rtol=0,
    )
    np.testing.assert_allclose(
        fit_py.resid(), np.asarray(stats.residuals(fit_r)), atol=1e-7, rtol=0
    )
    r_pred = np.asarray(stats.predict(fit_r))
    np.testing.assert_allclose(fit_py.predict(), r_pred, atol=1e-7, rtol=0)
    # Tighten LSQR tolerances when recovering fixed effects for newdata.
    np.testing.assert_allclose(
        fit_py.predict(newdata=data, atol=1e-12, btol=1e-12),
        r_pred,
        atol=1e-7,
        rtol=0,
    )


# =============================================================================
# Multiple i() Terms
# =============================================================================


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ i(f_str) + i(g)",
        "Y ~ i(f_str, ref='apple') + i(g, ref='X')",
        "Y ~ X1 + i(f_str) + i(g)",
    ],
)
def test_multiple_i_terms(df_test, fml):
    """Test multiple i() terms in one formula."""
    py_names, py_values, r_names, r_values = compare_with_r(fml, df_test)
    assert_models_match(py_names, py_values, r_names, r_values)


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ i(f_str) + i(g) | fe1",
        "Y ~ i(f_str, ref='apple') + i(g, ref='X') | fe1",
    ],
)
def test_multiple_i_terms_with_fe(df_test, fml):
    """Test multiple i() terms with fixed effects."""
    py_names, py_values, r_names, r_values = compare_with_r(fml, df_test)
    assert_models_match(py_names, py_values, r_names, r_values)


# =============================================================================
# Different Factor Types
# =============================================================================


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ i(f_str)",
        "Y ~ i(f_str, ref='apple')",
        "Y ~ i(f_int)",
        "Y ~ i(f_int, ref=1)",
        "Y ~ i(f_float)",
        "Y ~ i(f_float, ref=1)",
    ],
)
def test_factor_types(df_test, fml):
    """Test i() with string, integer, and float factors."""
    py_names, py_values, r_names, r_values = compare_with_r(fml, df_test)
    assert_models_match(py_names, py_values, r_names, r_values)


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ i(f_str, X1)",
        "Y ~ i(f_str, X1, ref='apple')",
        "Y ~ i(f_int, X1)",
        "Y ~ i(f_int, X1, ref=1)",
    ],
)
def test_factor_x_continuous(df_test, fml):
    """Test i(factor, continuous) with different factor types."""
    py_names, py_values, r_names, r_values = compare_with_r(fml, df_test)
    assert_models_match(py_names, py_values, r_names, r_values)


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.against_r_core
def test_interacted_fixed_effects(df_test):
    """Test i() with interacted fixed effects."""
    r_fml = "Y ~ i(f_str) | fe1^fe2"
    py_fml = "Y ~ i(f_str) | fe1:fe2"
    py_names, py_values, r_names, r_values = compare_with_r(r_fml, df_test, py_fml)
    assert_models_match(py_names, py_values, r_names, r_values)


@pytest.mark.against_r_core
def test_i_with_same_var_standalone(df_test):
    """Test i(f, X) when X is also used standalone."""
    fml = "Y ~ X1 + i(f_str, X1)"
    py_names, py_values, r_names, r_values = compare_with_r(fml, df_test)
    assert_models_match(py_names, py_values, r_names, r_values, check_names=False)


# =============================================================================
# Null Value Handling Tests
# =============================================================================


@pytest.fixture(scope="module")
def df_with_nulls() -> pd.DataFrame:
    """Create test data with null values in various positions."""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame(
        {
            "Y": np.random.randn(n),
            "X1": np.random.randn(n),
            "X2": np.random.randn(n),
            "f_str": np.random.choice(["A", "B", "C"], n),
            "f_int": np.random.choice([1, 2, 3], n),
            "fe": np.random.choice(range(5), n),
        }
    )

    # Introduce nulls in different variables at different positions
    df.loc[[5, 15, 25, 35, 45], "Y"] = np.nan  # Nulls in dependent variable
    df.loc[[10, 20, 30], "X1"] = np.nan  # Nulls in continuous variable
    df.loc[[12, 22, 32], "f_str"] = np.nan  # Nulls in factor variable
    df.loc[[14, 24], "X2"] = np.nan  # Nulls in another continuous variable

    return df


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ i(f_str)",  # Simple i() with nulls in Y and f_str
        "Y ~ i(f_str, X1)",  # i() with continuous, nulls in Y, f_str, X1
        "Y ~ i(f_str) + X2",  # i() with covariate, nulls in multiple vars
        "Y ~ i(f_int)",  # i() with integer factor
        "Y ~ i(f_int, X1)",  # i() with integer factor and continuous
    ],
)
def test_null_handling(df_with_nulls, fml):
    """Test that null values are handled consistently between pyfixest and fixest."""
    py_names, py_values, r_names, r_values = compare_with_r(fml, df_with_nulls)
    assert_models_match(py_names, py_values, r_names, r_values)


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ i(f_str) | fe",  # With fixed effects
        "Y ~ i(f_str, X1) | fe",  # i() with continuous and FE
        "Y ~ i(f_str) + X2 | fe",  # i() with covariate and FE
    ],
)
def test_null_handling_with_fe(df_with_nulls, fml):
    """Test null handling with fixed effects."""
    py_names, py_values, r_names, r_values = compare_with_r(fml, df_with_nulls)
    assert_models_match(py_names, py_values, r_names, r_values)


@pytest.mark.against_r_core
def test_null_handling_with_ref(df_with_nulls):
    """Test null handling with explicit reference level."""
    fml = "Y ~ i(f_str, ref='A')"
    py_names, py_values, r_names, r_values = compare_with_r(fml, df_with_nulls)
    assert_models_match(py_names, py_values, r_names, r_values)


@pytest.mark.against_r_core
def test_null_handling_nobs(df_with_nulls):
    """Test that number of observations matches after null removal."""
    fml = "Y ~ i(f_str, X1) + X2"

    fit_py = feols(fml, df_with_nulls)
    fit_r = fixest.feols(ro.Formula(fml), df_with_nulls)

    # Extract number of observations from R
    ro.globalenv["fit_tmp"] = fit_r
    r_nobs = int(ro.r("fit_tmp$nobs")[0])
    ro.r("rm(fit_tmp)")

    # Compare number of observations
    assert r_nobs == fit_py.sample_info.n_obs, (
        f"Number of observations mismatch: py={fit_py.sample_info.n_obs}, r={r_nobs}"
    )
