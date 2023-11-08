import pytest
import numpy as np
import pandas as pd
from pyfixest.estimation import feols
from pyfixest.exceptions import InvalidReferenceLevelError

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()

fixest = importr("fixest")
stats = importr("stats")
broom = importr("broom")

def test_i():


    df_het = pd.read_csv("pyfixest/experimental/data/df_het.csv")
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
        feols("dep_var~i(rel_year, treat)", df_het, i_ref1="1.0")
    with pytest.raises(InvalidReferenceLevelError):
        feols("dep_var~i(rel_year, treat)", df_het, i_ref1=[1])
    with pytest.raises(InvalidReferenceLevelError):
        feols("dep_var~i(rel_year, X)", df_het, i_ref1=[1, 2])
    with pytest.raises(AssertionError):
        feols("dep_var~i(rel_year, X)", df_het, i_ref1=[1.0, "a"])

    # i_ref2 currently not supported
    with pytest.raises(AssertionError):
        feols("dep_var~i(rel_year, treat)", df_het, i_ref2="1.0")



def test_i_vs_fixest():

    df_het = pd.read_csv("pyfixest/experimental/data/df_het.csv")

    # ---------------------------------------------------------------------------------------#
    # no fixed effects

    # no references
    fit_py = feols("dep_var~i(treat)", df_het)
    fit_r = fixest.feols(ro.Formula("dep_var~i(treat)"), df_het)
    np.testing.assert_allclose(fit_py.coef().values, np.array(fit_r.rx2("coefficients")))

    fit_py = feols("dep_var~i(rel_year)", df_het)
    fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year)"), df_het)
    np.testing.assert_allclose(fit_py.coef().values, np.array(fit_r.rx2("coefficients")))

    # with references
    fit_py = feols("dep_var~i(treat)", df_het, i_ref1=False)
    fit_r = fixest.feols(ro.Formula("dep_var~i(treat, ref = FALSE)"), df_het)
    np.testing.assert_allclose(fit_py.coef().values, np.array(fit_r.rx2("coefficients")))

    fit_py = feols("dep_var~i(rel_year)", df_het, i_ref1=1.0)
    fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year, ref = c(1))"), df_het)
    np.testing.assert_allclose(fit_py.coef().values, np.array(fit_r.rx2("coefficients")))

    fit_py = feols("dep_var~i(rel_year)", df_het, i_ref1= [1.0, 2.0])
    fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year, ref = c(1, 2))"), df_het)
    np.testing.assert_allclose(fit_py.coef().values, np.array(fit_r.rx2("coefficients")))

    # ---------------------------------------------------------------------------------------#
    # with fixed effects

    # no references
    fit_py = feols("dep_var~i(treat) | year", df_het)
    fit_r = fixest.feols(ro.Formula("dep_var~i(treat)|year"), df_het)
    np.testing.assert_allclose(fit_py.coef().values, np.array(fit_r.rx2("coefficients")))

    fit_py = feols("dep_var~i(rel_year) | year", df_het)
    fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year)|year"), df_het)
    np.testing.assert_allclose(fit_py.coef().values, np.array(fit_r.rx2("coefficients")))

    # with references
    if True:
        fit_py = feols("dep_var~i(treat) | year", df_het, i_ref1=False)
        fit_r = fixest.feols(ro.Formula("dep_var~i(treat, ref = FALSE)|year"), df_het)
        np.testing.assert_allclose(fit_py.coef().values, np.array(fit_r.rx2("coefficients")))

        fit_py = feols("dep_var~i(rel_year) | year", df_het, i_ref1=1.0)
        fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year, ref = c(1))|year"), df_het)
        np.testing.assert_allclose(fit_py.coef().values, np.array(fit_r.rx2("coefficients")))

        fit_py = feols("dep_var~i(rel_year) | year", df_het, i_ref1= [1.0, 2.0])
        fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year, ref = c(1, 2))|year"), df_het)
        np.testing.assert_allclose(fit_py.coef().values, np.array(fit_r.rx2("coefficients")))













