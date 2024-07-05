import numpy as np
import pandas as pd
import pytest
from rpy2.robjects import pandas2ri

# rpy2 imports
from rpy2.robjects.packages import importr

from pyfixest.estimation.estimation import feols
from pyfixest.utils.utils import ssc

pandas2ri.activate()
clubSandwich = importr("clubSandwich")
stats = importr("stats")
base = importr("base")


@pytest.mark.parametrize(
    "R",
    [
        np.eye(3),
    ],
)
def test_wald_test_single_equation_no_clustering(R):
    # Test R * \beta = 0 with single equation.
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
    fit = feols(fml, data, vcov=None, ssc=ssc(adj=False))

    # Wald test
    fit.wald_test(R=R, distribution="F")
    f_stat = fit._f_statistic
    p_stat = fit._p_value
    # Compare with R

    r_fit = stats.lm(fml, data=data)

    unique_cluster = base.seq_len(len(data))

    r_wald = clubSandwich.Wald_test(
        r_fit,
        constraints=base.matrix(R, 3, 3),
        vcov="CR0",
        cluster=unique_cluster,
        test="Naive-F",
    )

    r_fstat = pd.DataFrame(r_wald).T[1].values[0]
    r_pvalue = pd.DataFrame(r_wald).T[5].values[0]

    np.testing.assert_allclose(f_stat, r_fstat, rtol=1e-02, atol=1e-02)
    np.testing.assert_allclose(p_stat, r_pvalue, rtol=1e-02, atol=1e-02)


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
def test_wald_test_single_equation(R):
    # Test R * \beta = 0 with single equation.
    data = pd.read_csv("pyfixest/did/data/df_het.csv")
    data = data.iloc[1:3000]
    fml = "dep_var ~ treat"
    fit = feols(fml, data, vcov={"CRV1": "year"}, ssc=ssc(adj=False))

    # Wald test
    fit.wald_test(R=R)
    f_stat = fit._f_statistic
    p_value = fit._p_value

    # Compare with R
    r_fit = stats.lm(fml, data=data)
    r_wald = clubSandwich.Wald_test(
        r_fit,
        constraints=base.matrix(R, 1, 2),
        vcov="CR1",
        cluster=data["year"],
        test="Naive-Fp",
    )
    r_fstat = pd.DataFrame(r_wald).T[1].values[0]
    r_pvalue = pd.DataFrame(r_wald).T[5].values[0]

    np.testing.assert_allclose(f_stat, r_fstat, rtol=1e-02, atol=1e-02)
    np.testing.assert_allclose(p_value, r_pvalue, rtol=1e-03, atol=1e-03)


@pytest.mark.parametrize(
    "seedn",
    [
        50,
        20,
        100,
    ],
)
def test_wald_test_multiple_equation(seedn):
    # Test R * \beta = 0 with single equation.

    R = np.eye(3)
    # Generate correlated data for dep_var and treat
    np.random.seed(seedn)
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
    fit = feols(fml, data, vcov={"CRV1": "year"}, ssc=ssc(adj=False))

    # Wald test
    fit.wald_test(R=R)
    f_stat = fit._f_statistic
    p_value = fit._p_value

    # Compare with R
    r_fit = stats.lm(fml, data=data)
    r_wald = clubSandwich.Wald_test(
        r_fit,
        constraints=base.matrix(R, 3, 3),
        vcov="CR1",
        cluster=data["year"],
        test="Naive-F",
    )
    r_fstat = pd.DataFrame(r_wald).T[1].values[0]
    r_pvalue = pd.DataFrame(r_wald).T[5].values[0]

    np.testing.assert_allclose(f_stat, r_fstat, rtol=1e-03, atol=1e-03)
    np.testing.assert_allclose(p_value, r_pvalue, rtol=1e-03, atol=1e-03)


@pytest.mark.parametrize(
    "R, fml",
    [
        (np.eye(3), "dep_var ~ treat + X1"),
        (np.array([[-1, 0, 1], [0, 1, 1]]), "dep_var ~ treat + X1"),
        (np.eye(4), "dep_var ~ treat + X1 + X2"),
    ],
)
def test_wald_test_multiple_equations_pvalue(R, fml):
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

    fit = feols(fml, data, vcov={"CRV1": "year"}, ssc=ssc(adj=False))

    # Wald test
    fit.wald_test(R=R)
    f_stat = fit._f_statistic

    # Compare with R
    r_fit = stats.lm(fml, data=data)
    r_wald = clubSandwich.Wald_test(
        r_fit,
        constraints=base.matrix(R, Rsize1, Rsize2),
        vcov="CR1",
        cluster=data["year"],
        test="Naive-F",
    )
    r_fstat = pd.DataFrame(r_wald).T[1].values[0]
    np.testing.assert_allclose(f_stat, r_fstat, rtol=1e-02, atol=1e-02)
