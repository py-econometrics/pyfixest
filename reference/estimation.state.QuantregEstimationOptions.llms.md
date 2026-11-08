# QuantregEstimationOptions

``` python
QuantregEstimationOptions(
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
    quantile,
    method,
    quantile_tol,
    quantile_maxiter,
    seed,
)
```

Estimation options of a quantile-regression fit.

Published as `fit.options` by `quantreg()` fits.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| quantile | float | The quantile of the conditional distribution that was fitted. | *required* |
| method | ('fn', 'pfn') | Frisch-Newton interior point solver, with or without preprocessing. | `"fn"` |
| quantile_tol | float | Convergence tolerance of the interior point solver. | *required* |
| quantile_maxiter | int or None | Maximum number of solver iterations; `None` defers to the solver’s sample-size-dependent default. | *required* |
| seed | int or None | Seed of the subsampling draws of the `"pfn"` method. | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.quantreg("Y ~ X1", pf.get_data(), quantile=0.25)
fit.options.quantile, fit.options.method
```

    (0.25, 'fn')
