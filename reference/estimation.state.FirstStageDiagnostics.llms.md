# FirstStageDiagnostics

``` python
FirstStageDiagnostics(f_stat, p_value, eff_f)
```

Instrument-strength diagnostics of a 2SLS first stage.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| f_stat | float | Wald F statistic of the joint null that every excluded instrument has a zero first-stage coefficient. It inherits the first stage’s covariance estimator, so it is heteroskedasticity- or cluster-robust whenever the second stage is. | *required* |
| p_value | float | P-value of `f_stat`. | *required* |
| eff_f | float or None | Effective F statistic of [Olea and Pflueger (2013)](https://doi.org/10.1080/00401706.2013.806694), computed against a heteroskedasticity-robust first stage. `None` until `IV_Diag()` or `eff_F()` computes it. | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X2 | f1 | X1 ~ Z1", pf.get_data())
fit.first_stage.diagnostics
```

    FirstStageDiagnostics(f_stat=647.2703851604911, p_value=8.76497162841603e-143, eff_f=None)
