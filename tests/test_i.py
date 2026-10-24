"""
Comprehensive tests for pyfixest i() syntax.

Tests cover:
- Simple i(var) with different factor types
- Factor x Continuous: i(var, continuous)
- Factor x Factor: i(var1, var2)
- Binning: bin and bin2 parameters
- Intercept control: 0+, -1, 1+ syntax
- Fixed effects combinations
- IV controls and instruments with pipe and bracket syntax
- Multiple i() terms
"""

import re
import warnings

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.packages import importr

from pyfixest.errors import FormulaSyntaxError
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

    name = re.sub(r"(::)(-?\d+\.0)(\b|:)", normalize_float_level, name)
    return name.replace("::-Inf", "::-inf")


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
    "formula",
    [
        "dep_var ~ i(rel_year, ref=1.0)",
        "dep_var ~ i(rel_year, ref=-2.0)",
        "dep_var ~ i(rel_year, treat, ref=1.0)",
        "dep_var ~ i(rel_year, treat, ref=-2.0)",
    ],
)
def test_i_reference_exclusion(df_het, formula):
    """Test that reference levels are properly excluded."""
    py_names, py_values, r_names, r_values = compare_with_r(formula, df_het)
    assert_models_match(py_names, py_values, r_names, r_values)


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
@pytest.mark.parametrize(
    "r_fml,py_fml",
    [
        (
            "Y ~ i(f_str, bin=list(fruit=c('apple','banana')))",
            "Y ~ i(f_str, bin={'fruit': ['apple','banana']})",
        ),
        # A scalar old level bins that level instead of its characters.
        (
            "Y ~ i(f_str, bin=list(fruit='apple'))",
            "Y ~ i(f_str, bin={'fruit': 'apple'})",
        ),
    ],
)
def test_binning_simple(df_test, r_fml, py_fml):
    """Test i() with bin parameter."""
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
        ("Y ~ i(f_str, i.g, ref2='X')", "Y ~ i(f_str, g, ref2='X')"),
        # Without an intercept, every cell is kept (#1767)
        ("Y ~ X1 + i(f_str, i.g) - 1", "Y ~ X1 + i(f_str, g) - 1"),
        ("Y ~ 0 + X1 + i(f_str, i.g)", "Y ~ 0 + X1 + i(f_str, g)"),
        ("Y ~ 0 + i(f_str, i.g, ref='apple')", "Y ~ 0 + i(f_str, g, ref='apple')"),
        ("Y ~ 0 + i(f_str, i.g, ref2='X')", "Y ~ 0 + i(f_str, g, ref2='X')"),
        (
            "Y ~ 0 + i(f_str, i.g, ref='apple', ref2='X')",
            "Y ~ 0 + i(f_str, g, ref='apple', ref2='X')",
        ),
    ],
)
def test_factor_x_factor(df_test, r_fml, py_fml):
    """Test i(factor1, factor2) interactions."""
    py_names, py_values, r_names, r_values = compare_with_r(r_fml, df_test, py_fml)
    assert_models_match(py_names, py_values, r_names, r_values)


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "arguments",
    [
        "f_str, ref='apple'",
        "f_str, g, ref='apple'",
        "f_str, g, ref2='X'",
        "f_str, g, ref='apple', ref2='X'",
    ],
)
@pytest.mark.parametrize("operator", ["+", "*"])
def test_factor_reference_with_ordinary_factor(df_test, arguments, operator):
    """Explicit i() references must not reduce a later ordinary factor."""
    data = df_test.assign(
        h=np.random.default_rng(1942).choice(["A", "B"], len(df_test))
    )
    fit_py = feols(f"Y ~ 0 + i({arguments}) {operator} h", data)
    r_arguments = arguments.replace(", g", ", i.g")
    r_formula = ro.Formula(f"Y ~ 0 + i({r_arguments}) {operator} h")
    r_formula.environment["i"] = fixest.i
    fit_r = fixest.feols(r_formula, data)

    # Ordinary factor labels differ between formulaic and R's model.matrix.
    def name_to_r(name):
        return re.sub(r"h\[(?:T\.)?([AB])\]", r"h\1", str(name))

    py_coef = fit_py.coef().rename(index=name_to_r)
    r_coef = pd.Series(get_r_coef_values(fit_r), index=get_r_coef_names(fit_r))
    assert set(py_coef.index) == set(r_coef.index)
    assert fit_py.sample_info.n_obs == int(stats.nobs(fit_r)[0])
    # Both are well-conditioned OLS fits of the same design; only floating-point
    # solve/accumulation order differs.
    np.testing.assert_allclose(
        py_coef,
        r_coef.reindex(py_coef.index),
        rtol=1e-10,
        atol=1e-12,
        err_msg="Explicit-reference interaction coefficients must match fixest",
    )
    py_se = fit_py.se().rename(index=name_to_r)
    r_se = pd.Series(np.asarray(fixest.se(fit_r)), index=r_coef.index)
    np.testing.assert_allclose(
        py_se,
        r_se.reindex(py_se.index),
        rtol=1e-10,
        atol=1e-12,
        err_msg="Explicit-reference interaction standard errors must match fixest",
    )
    np.testing.assert_allclose(
        fit_py.predict(newdata=data.iloc[:20]),
        np.asarray(stats.predict(fit_r, newdata=data.iloc[:20])),
        rtol=1e-10,
        atol=1e-12,
        err_msg="Explicit-reference interaction predictions must match fixest",
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "r_fml,py_fml",
    [
        ("Y ~ i(f_str, i.g) | fe1", "Y ~ i(f_str, g) | fe1"),
        ("Y ~ i(f_str, i.g, ref='apple') | fe1", "Y ~ i(f_str, g, ref='apple') | fe1"),
        ("Y ~ i(f_str, i.g, ref2='X') | fe1", "Y ~ i(f_str, g, ref2='X') | fe1"),
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


# =============================================================================
# IV Controls and Instruments
# =============================================================================


@pytest.mark.against_r_core
@pytest.mark.parametrize("syntax", ["pipe", "bracket"])
@pytest.mark.parametrize("role", ["control", "instrument"])
@pytest.mark.parametrize(
    "intercept,interaction,fixed_effects",
    [
        ("1", "i(f_str, g)", ""),
        ("0", "i(f_str, g)", ""),
        ("-1", "i(f_str, g)", ""),
        ("0", "i(f_str, g, ref='apple')", ""),
        ("0", "i(f_str, g, ref2='X')", ""),
        ("0", "i(f_str, g, ref='apple', ref2='X')", ""),
        ("1", "i(f_str, g)", "fe1"),
        ("0", "i(f_str, X2) + i(f_str, g)", ""),
    ],
    ids=[
        "intercept",
        "zero",
        "minus-one",
        "ref",
        "ref2",
        "both-refs",
        "fe",
        "slope-first",
    ],
)
def test_i_iv_vs_fixest(df_test, syntax, role, intercept, interaction, fixed_effects):
    """Both IV syntaxes preserve interaction coding in both estimation stages."""
    rng = np.random.default_rng(1942)
    data = df_test.assign(Z=rng.normal(size=len(df_test)))
    error = rng.normal(size=len(data))
    data["D"] = data["Z"] + (data["f_str"] == "apple") + 0.5 * error
    data["Y"] = 2 * data["D"] + data["X1"] + error + data["Y"]
    controls = f"{intercept} + X1"
    instruments = "Z"
    if role == "control":
        controls += f" + {interaction}"
    else:
        instruments += f" + {interaction}"

    fe = f" | {fixed_effects}" if fixed_effects else ""
    pipe_formula = f"Y ~ {controls}{fe} | D ~ {instruments}"
    formula = (
        pipe_formula
        if syntax == "pipe"
        else f"Y ~ {controls} + [D ~ {instruments}]{fe}"
    )
    r_formula = pipe_formula.replace("i(f_str, g", "i(f_str, i.g")
    if intercept in {"0", "-1"} and (
        role == "instrument" or "i(f_str, X2)" in interaction
    ):
        # fixest selects an implicit reference in the instrument block and in
        # later i() terms even without an intercept. Explicit full-level keep
        # matches the full-cell specification used by pyfixest (see #1917 and
        # test_i_reference_uses_intercept_span).
        r_formula = r_formula.replace(
            "i(f_str, i.g)",
            "i(f_str, i.g, keep=c('apple', 'banana', 'cherry'))",
        )

    fit_py = feols(formula, data=data, vcov="iid")
    fit_r = fixest.feols(ro.Formula(r_formula), data, vcov="iid")
    first_stage_r = ro.r("function(fit) summary(fit, stage = 1)")(fit_r)
    for model_py, model_r in (
        (fit_py, fit_r),
        (fit_py.first_stage.model, first_stage_r),
    ):
        r_names = [name.removeprefix("fit_") for name in get_r_coef_names(model_r)]
        py_coef = model_py.coef()
        assert set(py_coef.index) == set(r_names)
        assert model_py.sample_info.n_obs == int(stats.nobs(model_r)[0])
        # Use the existing IV comparison tolerances in tests/test_iv.py.
        np.testing.assert_allclose(
            py_coef.loc[r_names],
            get_r_coef_values(model_r),
            rtol=0,
            atol=1e-8,
            err_msg="IV interaction coefficients must match fixest",
        )
        np.testing.assert_allclose(
            model_py.se().loc[r_names],
            np.asarray(fixest.se(model_r)),
            rtol=0,
            atol=1e-7,
            err_msg="IV interaction standard errors must match fixest",
        )
        np.testing.assert_allclose(
            model_py.resid()[:20],
            np.asarray(stats.residuals(model_r))[:20],
            rtol=0,
            atol=1e-8,
            err_msg="IV interaction residuals must match fixest",
        )


# =============================================================================
# Multiple i() Terms
# =============================================================================


@pytest.mark.against_r_core
@pytest.mark.parametrize("preceding", ["i(f_str, X1)", "i(f_int, ref=1)"])
@pytest.mark.parametrize(
    "step", ["i(f_str, g)", "sw(X2, i(f_str, g))", "csw(X2, i(f_str, g))"]
)
def test_i_reference_uses_intercept_span(df_test, preceding, step):
    """Unlike fixest 0.14.0, non-spanning i() terms do not remove a later cell."""
    fml = f"Y ~ 0 + {preceding} + {step}"
    fit_py = feols(fml, data=df_test)
    py_models = fit_py.to_list() if "sw(" in step else [fit_py]
    assert len(py_models) == (2 if "sw(" in step else 1)
    r_fml = fml.replace("i(f_str, g)", "i(f_str, i.g)")
    # An explicit keep containing all levels suppresses fixest's positional
    # reference choice without excluding any interaction cells.
    r_full_fml = r_fml.replace(
        "i(f_str, i.g)", "i(f_str, i.g, keep=c('apple', 'banana', 'cherry'))"
    )
    r_default = fixest.feols(ro.Formula(r_fml), df_test)
    r_full = fixest.feols(ro.Formula(r_full_fml), df_test)
    if "sw(" in step:
        assert len(r_default) == len(r_full) == len(py_models)
    r_defaults = (
        [r_default.rx2(i + 1) for i in range(len(py_models))]
        if "sw(" in step
        else [r_default]
    )
    r_fulls = (
        [r_full.rx2(i + 1) for i in range(len(py_models))]
        if "sw(" in step
        else [r_full]
    )

    for model_py, model_r, model_default in zip(
        py_models, r_fulls, r_defaults, strict=True
    ):
        py_coef = model_py.coef()
        r_coef = pd.Series(get_r_coef_values(model_r), index=get_r_coef_names(model_r))
        assert set(py_coef.index) == set(r_coef.index)
        assert model_py.sample_info.n_obs == int(stats.nobs(model_r)[0])
        # Identical, well-conditioned OLS designs differ only in numerical solve
        # and accumulation order, including in the expanded stepwise models.
        np.testing.assert_allclose(
            py_coef,
            r_coef.reindex(py_coef.index),
            rtol=1e-10,
            atol=1e-12,
            err_msg="Full interaction-cell coefficients must match explicit fixest references",
        )
        r_se = pd.Series(np.asarray(fixest.se(model_r)), index=r_coef.index)
        np.testing.assert_allclose(
            model_py.se(),
            r_se.reindex(py_coef.index),
            rtol=1e-10,
            atol=1e-12,
            err_msg="Full interaction-cell standard errors must match fixest",
        )
        np.testing.assert_allclose(
            model_py.predict(newdata=df_test.iloc[:20]),
            np.asarray(stats.predict(model_r, newdata=df_test.iloc[:20])),
            rtol=1e-10,
            atol=1e-12,
            err_msg="Full interaction-cell predictions must match fixest",
        )
        default_names = get_r_coef_names(model_default)
        first_cell = "f_str::apple:g::X"
        if first_cell in py_coef.index:
            assert set(default_names) == set(py_coef.index) - {first_cell}
            assert np.sum(model_py.resid() ** 2) < np.sum(
                np.asarray(stats.residuals(model_default)) ** 2
            ), (
                "Dropping the first interaction cell in fixest must change the fitted model"
            )
        else:
            assert set(default_names) == set(py_coef.index)


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


@pytest.mark.against_r_core
@pytest.mark.parametrize("dtype", ["bool", "boolean", "object"])
@pytest.mark.parametrize("fml", ["Y ~ i(f_str, flag)", "Y ~ i(f_str, flag) | fe1"])
def test_factor_x_boolean(df_test, fml, dtype):
    """Test that a boolean second variable is a 0/1 slope, as in fixest."""
    rng = np.random.default_rng(1756)
    flag = pd.Series(rng.choice([True, False], len(df_test)), dtype=dtype)
    if dtype != "bool":
        flag.iloc[[3, 30]] = np.nan
    fit_py = feols(fml, df_test.assign(flag=flag))
    # rpy2 converts object NaNs to TRUE; nullable booleans preserve R logical NAs.
    fit_r = fixest.feols(ro.Formula(fml), df_test.assign(flag=flag.astype("boolean")))
    assert_models_match(
        [normalize_coef_name(str(n)) for n in fit_py.coef().index],
        fit_py.coef().values,
        get_r_coef_names(fit_r),
        get_r_coef_values(fit_r),
    )


@pytest.mark.against_r_core
def test_event_study_boolean_treatment(df_het):
    """Test the event study with the boolean `treat` column of df_het."""
    fml = "dep_var ~ i(rel_year, treat, ref=-1.0) | unit + year"
    py_names, py_values, r_names, r_values = compare_with_r(fml, df_het)
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


@pytest.fixture(scope="module")
def df_unused_categories(df_test) -> pd.DataFrame:
    """Categorical columns with categories absent from the data."""
    df = df_test.copy()
    # Unused first category must not become the reference level; unused last
    # category must not produce an all-zero column.
    df["f_cat"] = pd.Categorical(
        df["f_str"], categories=["avocado", "apple", "banana", "cherry", "zucchini"]
    )
    # The first observed category, not the alphabetically first, is the reference.
    df["f_cat_reordered"] = df["f_cat"].cat.reorder_categories(
        ["avocado", "cherry", "apple", "banana", "zucchini"]
    )
    df["g_cat"] = pd.Categorical(df["g"], categories=["W", "X", "Y", "Z"])
    return df


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "r_fml,py_fml",
    [
        ("Y ~ X1 + i(f_cat)", None),
        ("Y ~ X1 + i(f_cat_reordered) | fe1", None),
        ("Y ~ X1 + i(f_cat, X2)", None),
        ("Y ~ X1 + i(f_cat, i.g_cat)", "Y ~ X1 + i(f_cat, g_cat)"),
    ],
)
def test_unused_categories(df_unused_categories, r_fml, py_fml):
    """Unused categories of a categorical produce no columns, as in fixest."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r"\s*\d+ variables dropped")
        py_names, py_values, r_names, r_values = compare_with_r(
            r_fml, df_unused_categories, py_fml
        )
    assert_models_match(py_names, py_values, r_names, r_values)


@pytest.mark.against_r_core
def test_unused_category_reference(df_unused_categories):
    """An unused category cannot be the reference, matching fixest."""
    fml = "Y ~ i(f_cat, ref='avocado')"
    with pytest.raises(ValueError, match=r"avocado.*not among the provided levels"):
        feols(fml, data=df_unused_categories)
    with pytest.raises(RRuntimeError, match="no match was found for 'avocado'"):
        fixest.feols(ro.Formula(fml), df_unused_categories)


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "r_fml,py_fml",
    [
        ("Y ~ X1 + i(f_str)", None),
        ("Y ~ X1 + i(f_cat)", None),
        (
            "Y ~ X1 + i(f_cat, i.g_cat, ref2='X')",
            "Y ~ X1 + i(f_cat, g_cat, ref2='X')",
        ),
    ],
)
def test_levels_fixed_before_missing_row_filter(df_unused_categories, r_fml, py_fml):
    """Levels observed only on excluded rows still affect reference selection."""
    df = df_unused_categories.copy()
    df.loc[(df["f_str"] == "apple") | (df["g"] == "X"), "Y"] = np.nan
    with pytest.warns(UserWarning, match="variables dropped"):
        py_names, py_values, r_names, r_values = compare_with_r(r_fml, df, py_fml)
    assert_models_match(py_names, py_values, r_names, r_values)


@pytest.fixture(scope="module", params=["string", "category"])
def df_split_levels(request) -> pd.DataFrame:
    """Apple occurs only in g1; banana and cherry occur in both samples."""
    cells = [
        ("g1", "apple", 10.0),
        ("g1", "banana", 20.0),
        ("g1", "cherry", 30.0),
        ("g2", "banana", 20.0),
        ("g2", "cherry", 30.0),
    ]
    df = pd.DataFrame(
        [
            (group, level, mean + error)
            for group, level, mean in cells
            for error in [-1.5, -0.5, 0.5, 1.5]
        ],
        columns=["grp", "fc", "Y"],
    )
    df["fc"] = df["fc"].astype(request.param)
    return df


@pytest.mark.against_r_core
@pytest.mark.parametrize("split_arg", ["split", "fsplit"])
def test_split_levels_choose_sample_reference(df_split_levels, split_arg):
    """Split-local references intentionally differ from fixest 0.14.0."""
    data = df_split_levels
    fml = "Y ~ i(fc)"
    py_models = feols(fml, data=data, **{split_arg: "grp"}).to_list()
    r_models = fixest.feols(ro.Formula(fml), data, **{split_arg: ro.Formula("~grp")})
    samples = [None, "g1", "g2"] if split_arg == "fsplit" else ["g1", "g2"]
    assert [model.model.sample_split.value for model in py_models] == samples
    assert len(r_models) == len(samples)

    for i, (sample, fit_py) in enumerate(zip(samples, py_models, strict=True)):
        fit_r = r_models.rx2(i + 1)
        if sample == "g2":
            # Pyfixest uses banana; fixest keeps full-data levels and drops
            # cherry as collinear, giving an equivalent parameterization.
            py_names = ["Intercept", "fc::cherry"]
            py_values = [20.0, 10.0]
            r_names = ["Intercept", "fc::banana"]
            r_values = [30.0, -10.0]
        else:
            py_names = r_names = ["Intercept", "fc::banana", "fc::cherry"]
            py_values = r_values = [10.0, 10.0, 20.0]
        assert fit_py.coef().index.tolist() == py_names
        assert get_r_coef_names(fit_r) == r_names
        np.testing.assert_allclose(
            fit_py.coef().values,
            py_values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Pyfixest coefficients for sample {sample}",
        )
        np.testing.assert_allclose(
            get_r_coef_values(fit_r),
            r_values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"R fixest coefficients for sample {sample}",
        )
        sample_data = data if sample is None else data.loc[data["grp"] == sample]
        newdata = sample_data.drop_duplicates(subset="fc")
        np.testing.assert_allclose(
            fit_py.predict(newdata=newdata),
            np.asarray(stats.predict(fit_r, newdata=newdata)),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Split predictions must match R fixest for sample {sample}",
        )


@pytest.mark.against_r_core
@pytest.mark.parametrize("split_arg", ["split", "fsplit"])
def test_split_reference_validity(df_split_levels, split_arg):
    """Unlike fixest 0.14.0, references must exist in every split sample."""
    data = df_split_levels
    fml = "Y ~ i(fc, ref='apple')"
    r_models = fixest.feols(ro.Formula(fml), data, **{split_arg: ro.Formula("~grp")})
    # R accepts apple despite its absence from g2, for both split options.
    assert len(r_models) == (3 if split_arg == "fsplit" else 2)
    with pytest.raises(ValueError, match=r"apple.*not among the provided levels"):
        feols(fml, data=data, **{split_arg: "grp"})


@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ i(f_str, X1) + i(f_str, var2=X1)",
        "Y ~ i(f_str, X1) + i(f_str, np.power(X1, 2))",
    ],
)
def test_duplicate_column_names_raise(df_test, fml):
    """Equally named columns would overwrite each other in the model matrix."""
    with pytest.raises(FormulaSyntaxError, match="duplicate column names"):
        feols(fml, df_test)


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
