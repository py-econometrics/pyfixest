# FitStatistics

``` python
FitStatistics(
    rmse=np.nan,
    r2=np.nan,
    adj_r2=np.nan,
    r2_within=np.nan,
    adj_r2_within=np.nan,
    deviance=np.nan,
    loglik=np.nan,
    loglik_null=np.nan,
    pseudo_r2=np.nan,
    pearson_chi2=np.nan,
)
```

Goodness-of-fit measures of a fitted model.

Every field is a float, and `NaN` marks a measure the estimator does not define, as R `fixest::fitstat()` returns `NA`. `FitStatistics()` is the all-`NaN` value carried by IV and quantile regression fits.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| rmse | float | Root mean squared error. Defined for `feols()` and Gaussian `feglm()` fits. | `np.nan` |
| r2 | float | R-squared. Defined for `feols()` and Gaussian `feglm()` fits. | `np.nan` |
| adj_r2 | float | Adjusted R-squared. Defined for `feols()` and Gaussian `feglm()` fits. | `np.nan` |
| r2_within | float | R-squared on the dependent variable residualized on the fixed effects. Defined for `feols()` and Gaussian `feglm()` fits with fixed effects. | `np.nan` |
| adj_r2_within | float | Adjusted within R-squared. Defined for `feols()` and Gaussian `feglm()` fits with fixed effects. | `np.nan` |
| deviance | float | Deviance of the fitted model. Defined for `feglm()` and `fepois()` fits. | `np.nan` |
| loglik | float | Log-likelihood. Defined for `fepois()` fits. | `np.nan` |
| loglik_null | float | Log-likelihood of the intercept-only model. Defined for unweighted `fepois()` fits. | `np.nan` |
| pseudo_r2 | float | McFadden pseudo R-squared, `1 - loglik / loglik_null`. Defined for unweighted `fepois()` fits. | `np.nan` |
| pearson_chi2 | float | Pearson chi-squared statistic. Defined for `fepois()` fits. | `np.nan` |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 | f1", pf.get_data())
fit.fitstat.r2, fit.fitstat.r2_within
```

    (np.float64(0.43707702156880857), np.float64(0.16141769260928363))
