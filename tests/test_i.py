import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# rpy2 imports
from rpy2.robjects.packages import importr

from pyfixest.estimation import feols
from pyfixest.exceptions import InvalidReferenceLevelError

pandas2ri.activate()

fixest = importr("fixest")
stats = importr("stats")
broom = importr("broom")


def test_i():
    df_het = pd.read_csv("pyfixest/did/data/df_het.csv")
    df_het["X"] = np.random.normal(size=len(df_het))

    if (
        "C(rel_year)[T.1.0]"
        in feols("dep_var~i(rel_year)", df_het, i_ref1=1.0)._coefnames
    ):
        raise AssertionError("C(rel_year)[T.1.0] should not be in the column names.")
    if (
        "C(rel_year)[T.-2.0]"
        in feols("dep_var~i(rel_year)", df_het, i_ref1=-2.0)._coefnames
    ):
        raise AssertionError("C(rel_year)[T.-2.0] should not be in the column names.")
    if (
        "C(rel_year)[T.1.0]"
        in feols("dep_var~i(rel_year)", df_het, i_ref1=[1.0, 2.0])._coefnames
    ):
        raise AssertionError("C(rel_year)[T.1.0] should not be in the column names.")
    if (
        "C(rel_year)[T.2.0]"
        in feols("dep_var~i(rel_year)", df_het, i_ref1=[1.0, 2.0])._coefnames
    ):
        raise AssertionError("C(rel_year)[T.2.0] should not be in the column names.")

    if (
        "C(rel_year)[T.1.0]:treat"
        in feols("dep_var~i(rel_year, treat)", df_het, i_ref1=1.0)._coefnames
    ):
        raise AssertionError(
            "C(rel_year)[T.1.0]:treat should not be in the column names."
        )
    if (
        "C(rel_year)[T.-2.0]:treat"
        in feols("dep_var~i(rel_year, treat)", df_het, i_ref1=-2.0)._coefnames
    ):
        raise AssertionError(
            "C(rel_year)[T.-2.0]:treat should not be in the column names."
        )
    if (
        "C(rel_year)[T.1.0]:treat"
        in feols("dep_var~i(rel_year, treat)", df_het, i_ref1=[1.0, 2.0])._coefnames
    ):
        raise AssertionError(
            "C(rel_year)[T.1.0]:treat should not be in the column names."
        )
    if (
        "C(rel_year)[T.2.0]:treat"
        in feols("dep_var~i(rel_year, treat)", df_het, i_ref1=[1.0, 2.0])._coefnames
    ):
        raise AssertionError(
            "C(rel_year)[T.2.0]:treat should not be in the column names."
        )

    with pytest.raises(InvalidReferenceLevelError):
        feols("dep_var~i(rel_year)", df_het, i_ref1="1.0")
    with pytest.raises(InvalidReferenceLevelError):
        feols("dep_var~i(rel_year)", df_het, i_ref1=[1])
    with pytest.raises(InvalidReferenceLevelError):
        feols("dep_var~i(rel_year)", df_het, i_ref1=[1, 2])
    with pytest.raises(AssertionError):
        feols("dep_var~i(rel_year)", df_het, i_ref1=[1.0, "a"])

    # i_ref2 currently not supported
    with pytest.raises(AssertionError):
        feols("dep_var~i(rel_year, treat)", df_het, i_ref2="1.0")


def test_i_vs_fixest():
    df_het = pd.read_csv("pyfixest/did/data/df_het.csv")

    # ------------------------------------------------------------------------ #
    # no fixed effects

    # no references
    fit_py = feols("dep_var~i(treat)", df_het)
    fit_r = fixest.feols(ro.Formula("dep_var~i(treat)"), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )

    fit_py = feols("dep_var~i(rel_year)", df_het)
    fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year)"), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )

    # with references
    fit_py = feols("dep_var~i(treat)", df_het, i_ref1=False)
    fit_r = fixest.feols(ro.Formula("dep_var~i(treat, ref = FALSE)"), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )

    fit_py = feols("dep_var~i(rel_year)", df_het, i_ref1=1.0)
    fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year, ref = c(1))"), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )

    fit_py = feols("dep_var~i(rel_year)", df_het, i_ref1=[1.0, 2.0])
    fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year, ref = c(1, 2))"), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )

    # ------------------------------------------------------------------------ #
    # with fixed effects

    # no references
    fit_py = feols("dep_var~i(treat) | year", df_het)
    fit_r = fixest.feols(ro.Formula("dep_var~i(treat)|year"), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )

    fit_py = feols("dep_var~i(rel_year) | year", df_het)
    fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year)|year"), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )

    # with references
    fit_py = feols("dep_var~i(treat) | year", df_het, i_ref1=False)
    fit_r = fixest.feols(ro.Formula("dep_var~i(treat, ref = FALSE)|year"), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )

    fit_py = feols("dep_var~i(rel_year) | year", df_het, i_ref1=1.0)
    fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year, ref = c(1))|year"), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )

    fit_py = feols("dep_var~i(rel_year) | year", df_het, i_ref1=[1.0, 2.0])
    fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year, ref = c(1, 2))|year"), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )


def test_i_interacted_fixest():
    df_het = pd.read_csv("pyfixest/did/data/df_het.csv")
    df_het["X"] = np.random.normal(df_het.shape[0])

    # ------------------------------------------------------------------------ #
    # no fixed effects

    # no references
    fit_py = feols("dep_var~i(state, year)", df_het)
    fit_r = fixest.feols(ro.Formula("dep_var~i(state, year)"), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )

    if True:
        # no reference one fixed effect
        fit_py = feols("dep_var~i(state, year) | state ", df_het)
        fit_r = fixest.feols(ro.Formula("dep_var~i(state, year) | state"), df_het)
        np.testing.assert_allclose(
            fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
        )

    if True:
        # one reference
        fit_py = feols("dep_var~i(state, year)  ", df_het, i_ref1=1)
        fit_r = fixest.feols(ro.Formula("dep_var~i(state, year, ref = 1)"), df_het)
        np.testing.assert_allclose(
            fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
        )

    if True:
        # one reference and fixed effect
        fit_py = feols("dep_var~i(state, year) | state ", df_het, i_ref1=1)
        fit_r = fixest.feols(
            ro.Formula("dep_var~i(state, year, ref = 1) | state"), df_het
        )
        np.testing.assert_allclose(
            fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
        )

    if True:
        # two references and no fixed effect
        fit_py = feols("dep_var~i(state, year)  ", df_het, i_ref1=[1, 2])
        fit_r = fixest.feols(ro.Formula("dep_var~i(state, year, ref = c(1,2))"), df_het)
        np.testing.assert_allclose(
            fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
        )

        # two references and one fixed effect
        fit_py = feols("dep_var~i(state, year) | state ", df_het, i_ref1=[1, 2])
        fit_r = fixest.feols(
            ro.Formula("dep_var~i(state, year, ref = c(1,2)) | state"), df_het
        )
        np.testing.assert_allclose(
            fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
        )
