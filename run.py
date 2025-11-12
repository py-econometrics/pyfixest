# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "pyfixest",
#     "rpy2",
#     "numpy"
# ]
# ///

import numpy as np
import pandas as pd
from rpy2.robjects import Formula, pandas2ri, r
from rpy2.robjects.packages import importr

import pyfixest as pf

if __name__ == "__main__":
    # Load R packages and mtcars data
    fixest = importr("fixest")
    broom = importr("broom")
    stats = importr("stats")
    mtcars = r["mtcars"]
    mtcars_py = pandas2ri.rpy2py(mtcars)

    # feols comparison
    formula = Formula("mpg ~ hp + wt + factor(cyl)")

    # feglm comparison
    fit = fixest.feglm(formula, data=mtcars, family=stats.gaussian)
    fit_ols = fixest.feols(formula, data=mtcars)
    tidied = pd.DataFrame(broom.tidy_fixest(fit)).T
    tidied.columns = ["term", "estimate", "std.error", "statistic", "p.value"]
    tidied = tidied.set_index("term")
    coef = tidied.loc["wt", "estimate"]
    se = tidied.loc["wt", "std.error"]
    uhat = stats.residuals(fit)[0:5]
    r_vcov = np.array(stats.vcov(fit))
    r_vcov_ols = np.array(stats.vcov(fit_ols))

    fit = pf.feglm("mpg ~ hp + wt + C(cyl)", mtcars_py, family="gaussian")
    coef = fit.tidy().loc["wt", "Estimate"]
    se = fit.tidy().loc["wt", "Std. Error"]
    uhat = fit._u_hat[0:5]
    py_vcov = fit._vcov

    print(
        f"R fixest.feglm(mpg ~ hp + wt + factor(cyl), family=gaussian): Coefficient of wt: {coef:.8f}"
    )
    print(
        f'pyfixest.feglm(mpg ~ hp + wt + C(cyl), family="gaussian"): Coefficient of wt: {coef:.8f}'
    )

    print(
        f"R fixest.feglm(mpg ~ hp + wt + factor(cyl), family=gaussian): Standard Error of wt: {se:.8f}"
    )
    print(
        f'pyfixest.feglm(mpg ~ hp + wt + C(cyl), family="gaussian"): Standard Error of wt: {se:.8f}'
    )

    print(
        f"R fixest.feglm(mpg ~ hp + wt + factor(cyl), family=gaussian): First 5 residuals: {list(uhat)}"
    )
    print(
        f'pyfixest.feglm(mpg ~ hp + wt + C(cyl), family="gaussian"): First 5 residuals: {list(uhat)}'
    )

    print(
        "R fixest.feglm(mpg ~ hp + wt + factor(cyl), family=gaussian): Variance-Covariance Matrix:",
        np.diag(r_vcov),
    )
    print(
        "pyfixest.feglm(mpg ~ hp + wt + C(cyl), family='gaussian'): Variance-Covariance Matrix:",
        np.diag(py_vcov),
    )

    print("vcov ratio", r_vcov / py_vcov)

    print("vcov ratio fixest", r_vcov / r_vcov_ols)
