import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# rpy2 imports
from rpy2.robjects.packages import importr

from pyfixest.estimation.estimation import feols

pandas2ri.activate()

fixest = importr("fixest")
stats = importr("stats")
broom = importr("broom")


@pytest.fixture(scope="module")
def df_het() -> pd.DataFrame:
    df_het = pd.read_csv("pyfixest/did/data/df_het.csv")
    df_het["X"] = np.random.normal(df_het.shape[0])
    return df_het


@pytest.mark.against_r_core
def test_i(df_het: pd.DataFrame):
    if (
        "C(rel_year)[T.1.0]"
        in feols("dep_var~i(rel_year, ref = 1.0)", df_het)._coefnames
    ):
        raise AssertionError("C(rel_year)[T.1.0] should not be in the column names.")
    if (
        "C(rel_year)[T.-2.0]"
        in feols("dep_var~i(rel_year,ref=-2.0)", df_het)._coefnames
    ):
        raise AssertionError("C(rel_year)[T.-2.0] should not be in the column names.")

    if (
        "C(rel_year)[T.1.0]:treat"
        in feols("dep_var~i(rel_year, treat, ref=1.0)", df_het)._coefnames
    ):
        raise AssertionError(
            "C(rel_year)[T.1.0]:treat should not be in the column names."
        )
    if (
        "C(rel_year)[T.-2.0]:treat"
        in feols("dep_var~i(rel_year, treat,ref=-2.0)", df_het)._coefnames
    ):
        raise AssertionError(
            "C(rel_year)[T.-2.0]:treat should not be in the column names."
        )

    with pytest.raises(ValueError):
        feols("dep_var~i(rel_year, ref = [1.0, 'a'])", df_het)


@pytest.mark.against_r_core
def test_i_vs_fixest(df_het: pd.DataFrame):
    df_het = df_het[df_het["year"] >= 2010]
    # ------------------------------------------------------------------------ #
    # no fixed effects

    # # no references
    # fit_py = feols("dep_var~i(treat)", df_het)
    # fit_r = fixest.feols(ro.Formula("dep_var~i(treat)"), df_het)
    # np.testing.assert_allclose(
    #     fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    # )
    #
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

    fit_py = feols("dep_var~i(rel_year, ref = 1.0)", df_het)
    fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year, ref = c(1))"), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )

    # ------------------------------------------------------------------------ #
    # with fixed effects

    # # no references
    # fit_py = feols("dep_var~i(treat) | year", df_het)
    # fit_r = fixest.feols(ro.Formula("dep_var~i(treat)|year"), df_het)
    # np.testing.assert_allclose(
    #     fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    # )
    #
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

    fit_py = feols("dep_var~i(rel_year,ref=1.0) | year", df_het)
    fit_r = fixest.feols(ro.Formula("dep_var~i(rel_year, ref = c(1))|year"), df_het)
    np.testing.assert_allclose(
        fit_py.coef().values, np.array(fit_r.rx2("coefficients"))
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "fml",
    [
        # TODO: If no reference level is specified, fixest drops first but pyfixest last coefficient when there is collinearity?
        # "dep_var ~ i(state)",
        # "dep_var ~ i(state, group)",
        "dep_var ~ i(state, ref = 1)",
        "dep_var ~ i(state, year)",
        "dep_var ~ i(state, year, ref = 1)",
        "dep_var ~ i(state, year) | state",
        "dep_var ~ i(state, year, ref = 1) | state",
        "dep_var ~ i(state, group, ref=1)",
        "dep_var ~ i(state, group, ref2='Group 1')",
        "dep_var ~ i(state, group, ref=1, ref2='Group 1')",
    ],
)
def test_i_interacted_fixest(fml: str, df_het: pd.DataFrame):
    fit_py = feols(fml, df_het)
    fit_r = fixest.feols(ro.Formula(fml), df_het)
    py_coef = fit_py.coef()
    r_coef = (
        pd.DataFrame(broom.tidy_fixest(fit_r, conf_int=ro.BoolVector([True])))
        .T.set_index(0)[1]
        .astype(float)
    )
    # map coefficient names
    pattern = r"C\(([^,]+), .*?\)\[([^\]]+)\]"
    replacement = r"\1::\2"
    py_coef.index = py_coef.index.str.replace(pattern, replacement, regex=True)
    r_coef.rename(index={"(Intercept)": "Intercept"}, inplace=True)
    coefs = py_coef.to_frame("py").join(r_coef.rename("r"), how="outer")
    np.testing.assert_allclose(coefs["py"], coefs["r"])
