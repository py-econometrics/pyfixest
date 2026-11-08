# Fepois

``` python
Fepois(
    FixestFormula,
    data,
    *,
    options,
    lookup_demeaned_data,
    lookup_preconditioner=None,
    sample_split=None,
)
```

Estimate a Poisson regression model.

Non user-facing class to estimate a Poisson regression model via Iterated Weighted Least Squares (IWLS).

Inherits from the Feglm class. Users should not directly instantiate this class, but rather use the [fepois()](../reference/estimation.api.fepois.fepois.llms.md) function. IRLS residualization is orchestrated by `Feglm` through `DemeanCache`, reusing preconditioners while projecting with each solve’s current weights.

The method implements the algorithm from Stata’s `ppmlhdfe` module.

## Attributes

| Name | Type | Description |
|----|----|----|
| model_matrix | ModelMatrix | Model frames containing the response, regressors, and other model inputs. |
| observation_weights | ObservationWeights | User-scale observation weights. |
| working_state | GlmWorkingState | Final within-scale IRLS design, response, weights, predictors and residuals. |
| fitted_values | FittedValues | The linear predictor and the response mean of the final IRLS iteration. |
| sandwich | SandwichComponents | IRLS scores, the Hessian X’ W X with the final working weights, and its inverse. |
| coefnames | list\[str\] | Names of the coefficients in the design matrix X. |
| options | GlmEstimationOptions | The estimation options the model was built with, including the IRLS `maxiter` and `tol`, the separation check, and the offset. |
| \_data | pd.DataFrame | The data frame used in the estimation. None if arguments `lean = True` or `store_data = False`. |

## Examples

`Fepois` is returned by [fepois()](../reference/estimation.api.fepois.fepois.llms.md) and is not constructed directly. Post-estimation methods are inherited from [Feols](../reference/estimation.models.feols_.Feols.llms.md).

``` python
import pyfixest as pf

data = pf.get_data(model="Fepois")
fit = pf.fepois("Y ~ X1 + X2 | f1", data)

fit.tidy()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|) | 2.5%      | 97.5%    |
|-------------|-----------|------------|-----------|-------------|-----------|----------|
| Coefficient |           |            |           |             |           |          |
| X1          | 0.001864  | 0.040712   | 0.045784  | 0.963482    | -0.077930 | 0.081658 |
| X2          | -0.014261 | 0.010903   | -1.307958 | 0.190888    | -0.035631 | 0.007109 |
