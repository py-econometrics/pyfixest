# Feols.tidy

``` python
tidy(alpha=0.05, inference_type='regular')
```

Tidy model outputs.

Return a tidy pd.DataFrame with the point estimates, standard errors, t-statistics, and p-values.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| alpha | float | The significance level for the confidence intervals. If None, computes a 95% confidence interval (`alpha = 0.05`). | `0.05` |
| inference_type | regular | Type of coefficient-wise inference to report. Only `"regular"` is currently available. Defaults to `"regular"`. | `"regular"` |

## Returns

| Name | Type | Description |
|----|----|----|
| tidy_df | pd.DataFrame | A tidy pd.DataFrame containing the regression results, including point estimates, standard errors, t-statistics, and p-values. |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
fit.tidy()
```

|             | Estimate  | Std. Error | t value    | Pr(\>\|t\|) | 2.5%      | 97.5%     |
|-------------|-----------|------------|------------|-------------|-----------|-----------|
| Coefficient |           |            |            |             |           |           |
| X1          | -0.949526 | 0.066373   | -14.305943 | 0.0         | -1.079777 | -0.819274 |
| X2          | -0.174225 | 0.017596   | -9.901590  | 0.0         | -0.208755 | -0.139695 |

Changing the variance estimator changes the standard errors, t-values and p-values reported by `tidy()`.

``` python
fit.vcov("hetero").tidy()
```

|             | Estimate  | Std. Error | t value    | Pr(\>\|t\|) | 2.5%      | 97.5%     |
|-------------|-----------|------------|------------|-------------|-----------|-----------|
| Coefficient |           |            |            |             |           |           |
| X1          | -0.949526 | 0.065020   | -14.603584 | 0.0         | -1.077123 | -0.821929 |
| X2          | -0.174225 | 0.018247   | -9.548359  | 0.0         | -0.210033 | -0.138418 |
