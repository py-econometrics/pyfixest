# GlmEstimationOptions

``` python
GlmEstimationOptions(
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
    maxiter,
    tol,
    separation_check,
    accelerate,
)
```

Estimation options of a GLM fit, including the IRLS options.

Published as `fit.options` by `feglm()` and `fepois()` fits. The family itself is not an option and is not carried here.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| maxiter | int | Maximum number of IRLS iterations. | *required* |
| tol | float | Convergence tolerance of the IRLS iteration. | *required* |
| separation_check | list\[str\] or None | Separation-detection methods run before estimation, or `None` when the check was skipped. | *required* |
| accelerate | bool | Whether the accelerated demeaning path of Stammann (2018) was used. | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.fepois("Y ~ X1 | f1", pf.get_data(model="Fepois"))
fit.options.maxiter, fit.options.tol
```

    (25, 1e-08)
