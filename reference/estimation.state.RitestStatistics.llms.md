# RitestStatistics

``` python
RitestStatistics(statistics, sample_stat, pvalue)
```

Randomization-inference draws, the sample statistic, and the p-value.

`ritest(store_ritest_statistics=True)` publishes one value as `fit.ritest_statistics`.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| statistics | NDArray\[np.float64\] | Resampled test statistics, shape (reps + 1,). The first entry is the observed statistic under the original assignment; the remaining `reps` entries are the randomization draws. | *required* |
| sample_stat | float | The observed statistic, centered at the null hypothesis value. | *required* |
| pvalue | float | Randomization-inference p-value of the test. | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 + X2", pf.get_data())
fit.ritest("X1", reps=100, store_ritest_statistics=True)
fit.ritest_statistics.pvalue
```

    0.0
