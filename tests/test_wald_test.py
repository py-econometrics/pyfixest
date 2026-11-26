import re
import warnings

import numpy as np
import pandas as pd
import pytest

# rpy2 imports
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.estimation import feols
from pyfixest.utils.utils import ssc

# Core R packages
fixest = importr("fixest")
stats = importr("stats")
base = importr("base")
broom = importr("broom")
car = importr("car")


@pytest.mark.against_r_core
def test_F_test_single_equation_no_clustering_with_auto_computed_wald_test():
    # Test Identity_2 * \beta = 0 with single equation.
    # Generate correlated data for dep_var and treat
    np.random.seed(50)
    n = 3000
    treat = np.random.choice([0, 1], size=n)
    X1 = 1 + np.random.randn(n)
    dep_var = 0.0 + 0.0 * treat + 0.0 * X1 + np.random.randn(n)

    data = pd.DataFrame(
        {
            "dep_var": dep_var,
            "treat": treat,
            "X1": X1,
            "year": np.random.choice(range(2000, 2023), size=n),
        }
    )

    fml = "dep_var ~ treat + X1"
    fit = feols(fml, data, vcov=None, ssc=ssc(k_adj=False))

    # Joint test of all three coefficients, including the intercept.
    wald = fit.wald_test(R=np.eye(3))
    f_stat = wald.f_statistic
    p_stat = wald.pvalue

    # Compare with R

    r_fit = stats.lm(fml, data=data)
    r_wald = car.linearHypothesis(r_fit, base.matrix(np.identity(2), 3, 3), test="F")

    r_fstat = r_wald.rx2("F")[1]
    r_pvalue = r_wald.rx2("Pr(>F)")[1]

    np.testing.assert_allclose(f_stat, r_fstat, rtol=1e-03, atol=1e-02)
    np.testing.assert_allclose(p_stat, r_pvalue, rtol=1e-03, atol=1e-02)


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "R",
    [
        np.array([1, 0]),
        np.array([1, 2]),
        np.array([1, 1]),
        np.array([-1, 1]),
        # np.eye(2) * 2,
    ],
)
def test_F_test_single_equation(R):
    # Test R * \beta = 0 with single equation.
    data = pd.read_csv("pyfixest/did/data/df_het.csv")
    data = data.iloc[1:3000]
    fml = "dep_var ~ treat"
    fit = feols(fml, data, vcov={"CRV1": "year"}, ssc=ssc(k_adj=False))

    # Wald test
    wald = fit.wald_test(R=R)
    f_stat = wald.f_statistic
    p_value = wald.pvalue

    # Compare with R
    r_fit = fixest.feols(
        ro.Formula(fml),
        data=data,
        vcov=ro.Formula("~year"),
        ssc=fixest.ssc(False),
    )

    r_wald = car.linearHypothesis(r_fit, base.matrix(R, 1, 2), test="F")

    r_fstat = r_wald.rx2("F")[1]
    r_pvalue = r_wald.rx2("Pr(>F)")[1]

    np.testing.assert_allclose(f_stat, r_fstat, rtol=1e-03, atol=1e-03)
    np.testing.assert_allclose(p_value, r_pvalue, rtol=1e-03, atol=1e-03)


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "seedn",
    [
        51,
        20,
        100,
    ],
)
def test_F_test_multiple_equation(seedn):
    # Test R * \beta = 0 with single equation.

    R = np.eye(3)
    # Generate correlated data for dep_var and treat
    np.random.seed(seedn)
    n = 3000
    treat = np.random.choice([0, 1], size=n)
    X1 = 1 + np.random.randn(n)
    dep_var = 0.0 + 1.0 * treat + 0.0 * X1 + np.random.randn(n)

    data = pd.DataFrame(
        {
            "dep_var": dep_var,
            "treat": treat,
            "X1": X1,
            "year": np.random.choice(range(2000, 2023), size=n),
        }
    )

    fml = "dep_var ~ treat + X1"
    fit = feols(fml, data, vcov={"CRV1": "year"}, ssc=ssc(k_adj=False))

    # Wald test
    wald = fit.wald_test(R=R)
    f_stat = wald.f_statistic
    p_value = wald.pvalue

    r_fit = fixest.feols(
        ro.Formula(fml),
        data=data,
        vcov=ro.Formula("~year"),
        ssc=fixest.ssc(False),
    )

    r_wald = car.linearHypothesis(r_fit, base.matrix(R, 3, 3), test="F")

    r_fstat = r_wald.rx2("F")[1]
    r_pvalue = r_wald.rx2("Pr(>F)")[1]

    np.testing.assert_allclose(f_stat, r_fstat, rtol=1e-03, atol=1e-03)
    np.testing.assert_allclose(p_value, r_pvalue, rtol=1e-03, atol=1e-03)


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "R, fml",
    [
        (np.eye(3), "dep_var ~ treat + X1"),
        (np.array([[-1, 0, 1], [0, 1, 1]]), "dep_var ~ treat + X1"),
        (np.eye(4), "dep_var ~ treat + X1 + X2"),
    ],
)
def test_F_test_multiple_equations_pvalue(R, fml):
    # Test R * \beta = 0 with multiple equations.
    # In this test, we test p-values that are quite larger than 0.0
    # Generate correlated data for dep_var and treat
    Rsize1 = R.shape[0]
    Rsize2 = R.shape[1]
    np.random.seed(0)
    n = 3000
    treat = np.random.choice([0, 1], size=n)
    X1 = 1 + np.random.randn(n)

    if Rsize2 == 3:
        dep_var = 1.0 + 0.5 * treat + 2.0 * X1 + np.random.randn(n)
        data = pd.DataFrame(
            {
                "dep_var": dep_var,
                "treat": treat,
                "X1": X1,
                "year": np.random.choice(range(2000, 2023), size=n),
            }
        )
    else:
        X2 = 2 + np.random.randn(n)
        dep_var = 1.0 + 0.5 * treat + 2.0 * X1 + 3.0 * X2 + np.random.randn(n)
        data = pd.DataFrame(
            {
                "dep_var": dep_var,
                "treat": treat,
                "X1": X1,
                "X2": X2,
                "year": np.random.choice(range(2000, 2023), size=n),
            }
        )

    fit = feols(fml, data, vcov={"CRV1": "year"}, ssc=ssc(k_adj=False))

    # Wald test
    f_stat = fit.wald_test(R=R).f_statistic

    r_fit = fixest.feols(
        ro.Formula(fml),
        data=data,
        vcov=ro.Formula("~year"),
        ssc=fixest.ssc(False),
    )

    r_wald = car.linearHypothesis(r_fit, base.matrix(R, Rsize1, Rsize2), test="F")

    r_fstat = r_wald.rx2("F")[1]

    np.testing.assert_allclose(f_stat, r_fstat, rtol=1e-02, atol=1e-02)


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "R, q, fml",
    [
        (np.array([[0, 1, 2], [1, 1, 0]]), np.array([-1.169, -0.104]), "Y ~ X1 + X2"),
        (np.array([[0, 1, 2], [1, 1, 0]]), np.array([-1.4, -0.11]), "Y ~ X1 + X2"),
        (
            np.array([[1, 1, 1, 0], [1, 2, 0, 0]]),
            np.array([-0.200, -1.131]),
            "Y ~ X1 + X2 + Z1",
        ),
    ],
)
def test_wald_test_multiple_equations(R, q, fml):
    # Test R * \beta = q with multiple equations.
    # In this test, we test p-values that are quite larger than 0.0
    # Generate correlated data

    R_nrows = R.shape[0]
    R_ncloumns = R.shape[1]

    data = pf.get_data()
    fit_r = stats.lm(fml, data=data)
    R_r = ro.r.matrix(R.reshape(1, R_nrows * R_ncloumns), nrow=R_nrows, byrow=True)

    fit2 = pf.feols(fml, data=data)
    # Define the hypothesis matrix R
    Rpf = R

    # Define the hypothesis values q (both zero)

    # Perform the Wald test
    wald = fit2.wald_test(R=Rpf, q=q, distribution="chi2")

    r_result = car.linearHypothesis(fit_r, R_r, rhs=ro.FloatVector(q), test="Chisq")

    # Extracting p-value from the result
    wald_stat = wald.wald_statistic
    p_value = wald.pvalue

    r_wald_stat = r_result[4][1]
    r_p_value = r_result[5][1]

    np.testing.assert_allclose(wald_stat, r_wald_stat, rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(p_value, r_p_value, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize(
    "R, q, expected_message",
    [
        pytest.param(
            [[0, 1, -1]],
            [0],
            "Distribution changed to chi2, as R is not an identity matrix.",
            id="non-identity-R",
        ),
        pytest.param(
            np.eye(3),
            [1.0, 0.0, 0.0],
            "Distribution changed to chi2, as q is not a zero vector.",
            id="nonzero-q",
        ),
        pytest.param(
            [[0, 1, -1], [1, 0, 0]],
            [0, 1.0],
            "Distribution changed to chi2, as R is not an identity matrix and q is not a zero vector.",
            id="non-identity-R-nonzero-q",
        ),
    ],
)
def test_wald_test_chi2_warning_names_the_triggered_reason(R, q, expected_message):
    # The F fallback warning must only claim the restrictions that actually
    # forced the switch to chi2.
    data = pf.get_data()
    fit = pf.feols("Y ~ X1 + X2", data)

    with pytest.warns(UserWarning, match=re.escape(expected_message)) as record:
        wald = fit.wald_test(R=R, q=q)

    assert len(record) == 1
    assert str(record[0].message) == expected_message
    assert wald.distribution == "chi2"


def test_wald_test_joint_null_keeps_F_without_warning():
    data = pf.get_data()
    fit = pf.feols("Y ~ X1 + X2", data)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        wald = fit.wald_test(R=np.eye(3), q=[0, 0, 0])

    assert not [w for w in record if "Distribution changed to chi2" in str(w.message)]
    assert wald.distribution == "F"


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "fml, vcov",
    [
        ("Y ~ X1", "iid"),
        ("Y ~ X1", {"CRV1": "f1"}),
        ("Y ~ X1 + X2", "hetero"),
        ("Y ~ X1 + X2 | f1", {"CRV1": "f1"}),
    ],
)
def test_default_wald_test_excludes_intercept_like_fixest(fml, vcov):
    # A weak signal keeps the p-values away from zero.
    rng = np.random.default_rng(2096)
    n = 500
    data = pd.DataFrame(
        {
            "f1": np.repeat(np.arange(20), 25),
            "X1": rng.normal(size=n),
            "X2": rng.normal(size=n),
        }
    )
    data["Y"] = 1.0 + 0.1 * data["X1"] + rng.normal(size=n)
    fit = pf.feols(fml, data, vcov=vcov)

    r_vcov = ro.Formula("~f1") if isinstance(vcov, dict) else vcov
    fit_r = fixest.feols(ro.Formula(fml), data=data, vcov=r_vcov)
    wald_r = fixest.wald(fit_r, print=False)

    assert fit.wald.df1 == wald_r.rx2("df1")[0]
    assert fit.wald.df2 == wald_r.rx2("df2")[0]
    np.testing.assert_allclose(fit.wald.stat, wald_r.rx2("stat")[0], rtol=1e-6)
    np.testing.assert_allclose(fit.wald.pvalue, wald_r.rx2("p")[0], rtol=1e-6)


def test_intercept_only_model_has_no_default_wald_test():
    fit = pf.feols("Y ~ 1", pf.get_data())

    assert not hasattr(fit, "wald")
    with pytest.raises(ValueError, match="no coefficients besides the intercept"):
        fit.wald_test()
    assert fit.wald_test(R=np.eye(1)).df1 == 1
