# EstimationOptions

``` python
EstimationOptions(
    ssc,
    drop_singletons,
    drop_intercept,
    weights,
    weights_type,
    offset,
    collin_tol,
    solver,
    demeaner,
    store_data,
    copy_data,
    lean,
    context,
)
```

Estimation options a fitted model was built with.

Every model class publishes one value in its constructor as `fit.options`, so the options a result was produced with can be read back from the result itself. GLM and quantile fits publish the estimator-specific extensions below. The value survives `store_data=False` and `lean=True`, which it describes through `retention`.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| ssc | Ssc | Small-sample correction options, as built by [ssc()](../reference/utils.utils.ssc.llms.md). | *required* |
| drop_singletons | bool | Whether singleton fixed-effect levels were dropped (`fixef_rm="singleton"`). | *required* |
| drop_intercept | bool | Whether the intercept was removed from the design. | *required* |
| weights | str or None | Name of the observation-weight column, or `None` for an unweighted fit. | *required* |
| weights_type | ('aweights', 'fweights') | Interpretation of `weights`: analytic or frequency weights. | `"aweights"` |
| offset | str or None | Name of the offset column of a GLM fit, or `None`. | *required* |
| collin_tol | float | Tolerance of the rank check that drops collinear columns. | *required* |
| solver | str | Linear solver used for the least-squares steps. | *required* |
| demeaner | MapDemeaner or LsmrDemeaner | Resolved fixed-effect demeaner configuration. | *required* |
| store_data | bool | Whether the estimation data and formula state were retained. | *required* |
| copy_data | bool | Whether the estimation data was copied before the fit, the documented exception to never mutating user input. | *required* |
| lean | bool | Whether the large fit products were dropped after estimation. | *required* |
| context | Mapping\[str, Any\] | Variables made available to formulaic when materializing the model matrix, as captured by `capture_context`. | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 | f1", pf.get_data(), weights="weights")
fit.options.weights, fit.options.weights_type, fit.options.demeaner
```

    ('weights',
     'aweights',
     LsmrDemeaner(fixef_maxiter=1000, backend='within', precision='float64', device='auto', fixef_atol=1e-08, fixef_btol=1e-08, warn_on_cpu_fallback=True, preconditioner='auto', local_size=None))

## Attributes

| Name | Description |
|----|----|
| [EstimationOptions.fixef_tol](#pyfixest.estimation.state.EstimationOptions.fixef_tol) | Stopping tolerance of the demeaning algorithm. |
| [EstimationOptions.has_weights](#pyfixest.estimation.state.EstimationOptions.has_weights) | Whether the fit used user-supplied observation weights. |
| [EstimationOptions.retention](#pyfixest.estimation.state.EstimationOptions.retention) | Storage policy the fitted model was cleaned up under. |
