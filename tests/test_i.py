from typing import Optional

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# rpy2 imports
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.estimation.estimation import feols

pandas2ri.activate()

fixest = importr("fixest")
stats = importr("stats")
broom = importr("broom")


def i_name(
    var1: str,
    var2: Optional[str] = None,
    ref1: Optional[str] = None,
    ref2: Optional[str] = None,
) -> str:
    name = f"{var1}"
    if ref1 is not None:
        name = f"{name}::{ref1}"
    if var2 is not None:
        name = f"{name}:{var2}"
    if ref2 is not None:
        name = f"{name}:{ref2}"
    return name


def i_func(
    var1: str,
    var2: Optional[str] = None,
    ref1: Optional[str] = None,
    ref2: Optional[str] = None,
) -> str:
    name = f"{var1}"
    if var2 is not None:
        name = f"{name}, {var2}"
    if ref1 is not None:
        name = f"{name}, ref={ref1}"
    if ref2 is not None:
        name = f"{name}, ref2={ref2}"
    return f"i({name})"


@pytest.fixture(scope="module")
def df_het() -> pd.DataFrame:
    df_het = pd.read_csv("pyfixest/did/data/df_het.csv")
    df_het["X"] = np.random.normal(size=len(df_het))
    return df_het


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(var1="rel_year", ref1=1.0),
        dict(var1="rel_year", ref1=-2.0),
        dict(var1="rel_year", var2="treat", ref1=1.0),
        dict(var1="rel_year", var2="treat", ref1=-2.0),
    ],
)
def test_i(df_het, kwargs):
    n = i_name(**kwargs)
    formula = f"dep_var~{i_func(**kwargs)}"
    fit = feols(formula, df_het)
    if n in fit._coefnames:
        raise AssertionError(f"{n} should not be in the column names.")


@pytest.mark.against_r_core
def test_i_vs_fixest():
    df_het = pd.read_csv("pyfixest/did/data/df_het.csv")
    df_het = df_het[df_het["year"] >= 2010]
    # ------------------------------------------------------------------------ #
    # no fixed effects

    # TODO: fixest drops `treat::FALSE`, pyfixest drops `treat::True`
    # # no references
    # fit_py = feols("dep_var~i(treat)", df_het)
    # fit_r = fixest.feols(ro.Formula("dep_var~i(treat)"), df_het)
    # np.testing.assert_allclose(
    #     fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    # )

    # TODO: fixest keeps `rel_year::20.0`
    # fit_py = feols("dep_var~i(rel_year)", df_het)
    # fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year)"), df_het)
    # np.testing.assert_allclose(
    #     fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    # )

    # with references
    fit_py = feols("dep_var~i(treat, ref = False)", df_het)
    fit_r = fixest.feols(ro.Formula("dep_var~i(treat, ref = FALSE)"), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )

    # TODO: fixest adds coefficient `rel_year::-Inf`?
    # fit_py = feols("dep_var~i(rel_year, ref = 1.0)", df_het)
    # fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year, ref = c(1))"), df_het)
    # np.testing.assert_allclose(
    #     fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    # )

    # ------------------------------------------------------------------------ #
    # with fixed effects

    # TODO: fixest drops `treat::FALSE`, pyfixest drops `treat::True`
    # # no references
    # fit_py = feols("dep_var~i(treat) | year", df_het)
    # fit_r = fixest.feols(ro.Formula("dep_var~i(treat)|year"), df_het)
    # np.testing.assert_allclose(
    #     fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    # )

    # TODO: pyfixest drops `rel_year::11.0` to `rel_year::20.0` due to collinearity; fixest does not?
    # fit_py = feols("dep_var~i(rel_year) | year", df_het)
    # fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year)|year"), df_het)
    # np.testing.assert_allclose(
    #     fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    # )

    # with references
    fit_py = feols("dep_var~i(treat,ref=False) | year", df_het)
    fit_r = fixest.feols(ro.Formula("dep_var~i(treat, ref = FALSE)|year"), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )

    # TODO: pyfixest drops `rel_year::11.0` to `rel_year::20.0` due to collinearity; fixest does not?
    # fit_py = feols("dep_var~i(rel_year,ref=1.0) | year", df_het)
    # fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year, ref = c(1))|year"), df_het)
    # np.testing.assert_allclose(
    #     fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    # )


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
def test_i_interacted_fixest(fml):
    df_het = pd.read_csv("pyfixest/did/data/df_het.csv")
    df_het["X"] = np.random.normal(df_het.shape[0])

    fit_py = feols(fml, df_het)
    fit_r = fixest.feols(ro.Formula(fml), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )


@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ i(f1)",
        "Y ~ i(f1, ref = 1.0)",
        "Y ~ i(f1, X1)",
        "Y ~ i(f1, X1, ref = 2.0)",
        "Y ~ i(f1) + X2",
        "Y ~ i(f1, ref = 1.0) + X2",
        "Y ~ i(f1, X1) + X2",
        "Y ~ i(f1, X1, ref = 2.0) + X2",
    ],
)
def test_get_icovars(fml):
    # Use the data and fml from the fixture and parameterization
    fit = pf.feols(fml, data=pf.get_data())
    assert len(fit._icovars) > 0, "No icovars found"
    assert "X2" not in fit._icovars, "X2 is found in _icovars"
