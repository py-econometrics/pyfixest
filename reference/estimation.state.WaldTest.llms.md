# WaldTest

``` python
WaldTest(
    stat,
    pvalue,
    df1,
    df2,
    distribution,
    vcov_type,
    wald_statistic,
    f_statistic,
)
```

Wald test of a linear hypothesis R @ beta = q.

Mirrors the return value of R `fixest`’s `wald()`: `stat` is the statistic of the reference distribution actually used, `df1` and `df2` are its degrees of freedom, and `vcov_type` names the covariance estimator the quadratic form was built with.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| stat | float | Test statistic under `distribution`: the F-scaled `f_statistic` for `"F"`, the unscaled `wald_statistic` for `"chi2"`. | *required* |
| pvalue | float | P-value of `stat` under `distribution`. | *required* |
| df1 | int | Numerator degrees of freedom, the number of restrictions in R. | *required* |
| df2 | int or float | Denominator degrees of freedom: the number of clusters minus one under clustered inference, otherwise the number of observations minus the number of estimated coefficients and fixed effects. | *required* |
| distribution | ('F', 'chi2') | Reference distribution. `"F"` is only used for the joint null that every coefficient is zero; any other restriction falls back to `"chi2"`. | `"F"` |
| vcov_type | str | Covariance estimator the test was computed with, as `VarianceCovariance.vcov_type_detail`. | *required* |
| wald_statistic | float | Wald quadratic form W = (R @ beta - q)’ (R V R’)^-1 (R @ beta - q). | *required* |
| f_statistic | float | F-scaled statistic W / `df1`, available under either distribution. | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data(), vcov={"CRV1": "f1"})
fit.wald
```

    WaldTest(stat=146.33077465377698, pvalue=6.661338147750939e-16, df1=2, df2=29, distribution='F', vcov_type='CRV1', wald_statistic=292.66154930755397, f_statistic=146.33077465377698)

`feols()` fits run the joint test on all coefficients automatically; other estimators publish `fit.wald` once `wald_test()` is called.

``` python
import numpy as np

fit.wald_test(R=np.array([[1.0, -1.0]]), q=np.array([0.0]), distribution="chi2")
```

    WaldTest(stat=126.15267641766282, pvalue=2.847143065521132e-29, df1=1, df2=29, distribution='chi2', vcov_type='CRV1', wald_statistic=126.15267641766282, f_statistic=126.15267641766282)
