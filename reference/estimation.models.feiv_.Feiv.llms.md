# Feiv

``` python
Feiv(
    FixestFormula,
    data,
    *,
    options,
    lookup_demeaned_data,
    lookup_preconditioner=None,
    sample_split=None,
)
```

Non user-facing class to estimate an IV model using a 2SLS estimator.

Inherits from the Feols class. Users should not directly instantiate this class, but rather use the [feols()](../reference/estimation.api.feols.feols.llms.md) function. This class constructs the second-stage and instrument within arrays through the shared `DemeanCache` supplied by the estimation runner.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| FixestFormula | Formula | Parsed fixest formula, including the first stage. | *required* |
| data | pd.DataFrame | Estimation data, already converted to pandas and reindexed. | *required* |
| options | EstimationOptions | Every estimation option the fit is built with, as built by the estimation function. | *required* |
| lookup_demeaned_data | dict\[frozenset\[int\], DemeanedData\] | Demeaning cache shared across the models of one cache block. | *required* |
| lookup_preconditioner | Optional\[dict\[frozenset\[int\], Preconditioner\]\] | Preconditioner cache shared across the models of one cache block. | `None` |
| sample_split | SampleSplit or None | The variable and value by which the estimation sample was split. `None` if the model was fit on the entire input data set (minus dropping of missings etc). For all model classes. | `None` |

## Attributes

| Name | Type | Description |
|----|----|----|
| \_Z | np.ndarray | Processed instruments after handling multicollinearity. |
| \_weights_type_feiv | str | Type of the weights variable defined in Feiv class. Either “aweights” for analytic weights or “fweights” for frequency weights. |
| \_coefnames_z | list\[str\] | Names of coefficients for Z after handling multicollinearity. |
| collinearity_instruments | CollinearityCheck | Names and column mask of the instruments dropped by the rank check, set in get_fit(). |
| capabilities | Capabilities | Inference and post-estimation features this model class supports. |
| sandwich | SandwichComponents | Weighted scores of the first-stage projection X_hat, the 2SLS Hessian X_hat’ W X_hat, and its inverse, set in get_fit(). |
| \_beta_hat | np.ndarray | Estimated regression coefficients. |
| fitted_values | FittedValues | In-sample predictions on the link and the response scale, set in get_fit(). |
| \_u_hat | np.ndarray | Residuals of the regression model. |
| first_stage | FirstStage | First-stage regression fitted after second-stage inference: the coefficients pi_hat, the fitted values X_hat, the residuals v_hat, the fitted first-stage model, the excluded instruments, and the instrument-strength `diagnostics`. |
| \_data | pd.DataFrame | The data frame used in the estimation. None if arguments `lean = True` or `store_data = False`. |

## Raises

| Name | Type | Description |
|----|----|----|
|  | IVIdentificationError | If the exogenous variables and instruments fully explain an endogenous variable, or the instruments add no explanatory power for it. |

## Examples

`Feiv` is returned by [feols()](../reference/estimation.api.feols.feols.llms.md) when the formula includes an instrument block, i.e. `depvar ~ exog + [endog ~ instruments] | fe`.

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X2 + [X1 ~ Z1] | f1", pf.get_data())
fit.tidy()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|) | 2.5%      | 97.5%     |
|-------------|-----------|------------|-----------|-------------|-----------|-----------|
| Coefficient |           |            |           |             |           |           |
| X2          | -0.174227 | 0.017599   | -9.899554 | 0.0         | -0.208764 | -0.139689 |
| X1          | -0.992187 | 0.104775   | -9.469663 | 0.0         | -1.197801 | -0.786574 |

The first stage F-statistic is stored on the fitted object.

``` python
fit.first_stage.diagnostics.f_stat
```

    647.2703851604911

See the [instrumental variables tutorial](../tutorials/instrumental-variables.llms.md) for details.

## Methods

| Name | Description |
|----|----|
| [Feiv.IV_Diag](#pyfixest.estimation.models.feiv_.Feiv.IV_Diag) | Implement IV diagnostic tests. |
| [Feiv.IV_weakness_test](#pyfixest.estimation.models.feiv_.Feiv.IV_weakness_test) | Implement IV weakness test (F-test). |
| [Feiv.eff_F](#pyfixest.estimation.models.feiv_.Feiv.eff_F) | Compute Effective F stat (Olea and Pflueger 2013). |
| [Feiv.get_fit](#pyfixest.estimation.models.feiv_.Feiv.get_fit) | Fit a IV model using a 2SLS estimator. |

### Feiv.IV_Diag

``` python
IV_Diag(statistics=None)
```

Implement IV diagnostic tests.

#### Notes

This method covers diagnostic tests related with IV regression. We currently have IV weak tests only. More test will be updated in future updates!

#### Parameters

| Name       | Type        | Description                      | Default |
|------------|-------------|----------------------------------|---------|
| statistics | list\[str\] | List of IV diagnostic statistics | `None`  |

#### Example

The following is an example usage of this method:

    ::: {#3ccb8b88 .cell execution_count=3}
    ``` {.python .cell-code}
    import numpy as np
    import pandas as pd
    from pyfixest.estimation import feols

    # Set random seed for reproducibility
    np.random.seed(1)

    # Number of observations
    n = 1000

    # Simulate the data
    # Instrumental variable
    z = np.random.binomial(1, 0.5, size=n)
    z2 = np.random.binomial(1, 0.5, size=n)

    # Endogenous variable
    d = 0.5 * z + 1.5 * z2 + np.random.normal(size=n)

    # Control variables
    c1 = np.random.normal(size=n)
    c2 = np.random.normal(size=n)

    # Outcome variable
    y = 1.0 + 1.5 * d + 0.8 * c1 + 0.5 * c2 + np.random.normal(size=n)

    # Cluster variable
    cluster = np.random.randint(1, 50, size=n)
    weights = np.random.uniform(1, 3, size=n)

    # Create a DataFrame
    data = pd.DataFrame({
        'd': d,
        'y': y,
        'z': z,
        'z2': z2,
        'c1': c1,
        'c2': c2,
        'cluster': cluster,
        'weights': weights
    })

    vcov_detail = "iid"

    # Fit OLS model
    fit_ols = feols("y ~ 1 + d + c1 + c2", data=data, vcov=vcov_detail)

    # Fit IV model
    fit_iv = feols("y ~ 1 + c1 + c2 + [d ~ z]", data=data,
             vcov=vcov_detail,
             weights="weights")
    F_stat_pf = fit_iv.first_stage.diagnostics.f_stat
    fit_iv.IV_Diag()
    F_stat_eff_pf = fit_iv.first_stage.diagnostics.eff_f

    print("(Unadjusted) F stat :", F_stat_pf)
    print("Effective F stat :", F_stat_eff_pf)
    ```

    ::: {.cell-output .cell-output-stdout}
    ```
    (Unadjusted) F stat : 52.81535560457478
    Effective F stat : 48.612881198586784
    ```
    :::
    :::

### Feiv.IV_weakness_test

``` python
IV_weakness_test(iv_diag_statistics=None)
```

Implement IV weakness test (F-test).

This method covers hetero-robust and clustered-robust F statistics. It republishes `first_stage` with two updated statistics:

- `first_stage.diagnostics.f_stat`: F statistic of the first stage
- `first_stage.diagnostics.eff_f`: effective F statistic (Olea and Pflueger 2013) of the first stage

#### Notes

`f_stat` is adjusted to the specification of vcov.

#### Parameters

| Name               | Type | Description                    | Default |
|--------------------|------|--------------------------------|---------|
| iv_diag_statistics | list | List of IV weakness statistics | `None`  |

### Feiv.eff_F

``` python
eff_F()
```

Compute Effective F stat (Olea and Pflueger 2013).

### Feiv.get_fit

``` python
get_fit()
```

Fit a IV model using a 2SLS estimator.
