# FirstStage

``` python
FirstStage(
    coefficients,
    fitted_values,
    residuals,
    model,
    instruments,
    diagnostics,
)
```

First-stage regression retained by a fitted 2SLS model.

The first stage regresses the endogenous regressor on the exogenous regressors and the excluded instruments, on the second stage’s retained rows and with its fixed effects, weights, covariance estimator, and small-sample correction.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| coefficients | NDArray\[np.float64\] | First-stage coefficients pi_hat, one per first-stage regressor. | *required* |
| fitted_values | NDArray\[np.float64\] | Within-scale fitted values `design @ coefficients`, shape (n_rows,). | *required* |
| residuals | NDArray\[np.float64\] | First-stage residuals v_hat, shape (n_rows,). | *required* |
| model | Feols | The fitted first-stage model. It follows the second stage’s `store_data` and `lean` policy, so it drops the same state. | *required* |
| instruments | tuple\[str, …\] | Names of the excluded instruments, in first-stage design order. | *required* |
| diagnostics | FirstStageDiagnostics | Instrument-strength statistics of that first stage. | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X2 | f1 | X1 ~ Z1", pf.get_data())
fit.first_stage.instruments
```

    ('Z1',)

``` python
fit.first_stage.model.tidy()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|) | 2.5%      | 97.5%    |
|-------------|-----------|------------|-----------|-------------|-----------|----------|
| Coefficient |           |            |           |             |           |          |
| X2          | -0.005292 | 0.006606   | -0.801163 | 0.423235    | -0.018255 | 0.007671 |
| Z1          | 0.395350  | 0.015540   | 25.441509 | 0.000000    | 0.364855  | 0.425845 |
