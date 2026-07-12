<!-- Generated from docs/reference/estimation.FixestMulti_.FixestMulti.qmd; do not edit. -->

# estimation.FixestMulti_.FixestMulti

```python
estimation.FixestMulti_.FixestMulti(config, parsed, data, context)
```

Container for models produced by one multiple-estimation call.

`feols()`, `fepois()`, `feglm()`, and `quantreg()` return this class when a
formula, quantile list, `split`, or `fsplit` expands into multiple models.
Public APIs first create an `EstimationConfig`; `parse_formula()` expands the
call, and `runner.run_estimation()` fits each planned model while sharing
compatible demeaning caches. `FixestMulti` is only the result container: it
does not parse formulas, demean data, or fit models itself.

Use `to_list()` to obtain every model or `fetch_model()` to select one. The
reporting methods `summary()`, `etable()`, `coefplot()`, and `iplot()` apply
to the complete collection.

## Attributes

| Name | Description |
| --- | --- |
| [FixestFormulaDict](#pyfixest.estimation.FixestMulti_.FixestMulti.FixestFormulaDict) | Parsed formula dict keyed by fixed-effects spec. |

## Methods

| Name | Description |
| --- | --- |
| [confint](#pyfixest.estimation.FixestMulti_.FixestMulti.confint) | Obtain confidence intervals for the fitted models. |
| [fetch_model](#pyfixest.estimation.FixestMulti_.FixestMulti.fetch_model) | Fetch one fitted model by its zero-based position. |
| [tidy](#pyfixest.estimation.FixestMulti_.FixestMulti.tidy) | Combine every model's coefficient table into one tidy DataFrame. |
| [to_list](#pyfixest.estimation.FixestMulti_.FixestMulti.to_list) | Return all fitted models in estimation order. |
| [vcov](#pyfixest.estimation.FixestMulti_.FixestMulti.vcov) | Update inference for every fitted model in place. |
| [wildboottest](#pyfixest.estimation.FixestMulti_.FixestMulti.wildboottest) | Run a wild cluster bootstrap for all regressions in the Fixest object. |

### confint

```python
estimation.FixestMulti_.FixestMulti.confint()
```

Obtain confidence intervals for the fitted models.

#### Returns

| Name   | Type             | Description                                                           |
|--------|------------------|-----------------------------------------------------------------------|
|        | pandas.DataFrame | Lower and upper confidence bounds indexed by formula and coefficient. |

### fetch_model

```python
estimation.FixestMulti_.FixestMulti.fetch_model(i, print_fml=True)
```

Fetch one fitted model by its zero-based position.

#### Parameters

| Name      | Type       | Description                                                 | Default    |
|-----------|------------|-------------------------------------------------------------|------------|
| i         | int or str | The index of the model to fetch.                            | _required_ |
| print_fml | bool       | Whether to print the formula of the model. Default is True. | `True`     |

#### Returns

| Name   | Type        | Description                                             |
|--------|-------------|---------------------------------------------------------|
|        | ModelResult | The selected OLS, IV, GLM, Poisson, or quantile result. |

### tidy

```python
estimation.FixestMulti_.FixestMulti.tidy()
```

Combine every model's coefficient table into one tidy DataFrame.

#### Returns

| Name   | Type             | Description                                                     |
|--------|------------------|-----------------------------------------------------------------|
|        | pandas.DataFrame | Coefficient statistics indexed by formula and coefficient name. |

### to_list

```python
estimation.FixestMulti_.FixestMulti.to_list()
```

Return all fitted models in estimation order.

#### Returns

| Name   | Type                | Description                                                  |
|--------|---------------------|--------------------------------------------------------------|
|        | list\[ModelResult\] | Fitted OLS, IV, GLM, Poisson, or quantile-regression models. |

### vcov

```python
estimation.FixestMulti_.FixestMulti.vcov(vcov, vcov_kwargs=None)
```

Update inference for every fitted model in place.

#### Parameters

| Name        | Type                                                      | Description                                                                                                                                                                                                  | Default    |
|-------------|-----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| vcov        | RegressionVcovType, QuantregVcovType, or dict\[str, str\] | Covariance estimator accepted by every contained model. Regression models support iid, heteroskedastic, clustered, NW, and DK inference; quantile models support iid, nid, heteroskedastic, or one-way CRV1. | _required_ |
| vcov_kwargs | VcovKwargs                                                | HAC arguments such as `lag`, `time_id`, and `panel_id`.                                                                                                                                                      | `None`     |

#### Returns

| Name   | Type        | Description                            |
|--------|-------------|----------------------------------------|
|        | FixestMulti | This container with updated inference. |

### wildboottest

```python
estimation.FixestMulti_.FixestMulti.wildboottest(
    reps,
    cluster=None,
    param=None,
    weights_type='rademacher',
    impose_null=True,
    bootstrap_type='11',
    seed=None,
    k_adj=True,
    G_adj=True,
)
```

Run a wild cluster bootstrap for all regressions in the Fixest object.

#### Parameters

| Name           | Type               | Description                                                                                                                                                                                                                | Default        |
|----------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| reps           | int                | The number of bootstrap iterations to run.                                                                                                                                                                                 | _required_     |
| param          | Union\[str, None\] | A string of length one, containing the test parameter of interest. Default is None.                                                                                                                                        | `None`         |
| cluster        | Union\[str, None\] | The name of the cluster variable. Default is None. If None, uses the `self._clustervar` attribute as the cluster variable. If the `self._clustervar` attribute is None, a heteroskedasticity-robust wild bootstrap is run. | `None`         |
| weights_type   | str                | The type of bootstrap weights. Either 'rademacher', 'mammen', 'webb', or 'normal'. Default is 'rademacher'.                                                                                                                | `'rademacher'` |
| impose_null    | bool               | Should the null hypothesis be imposed on the bootstrap dgp, or not? Default is True.                                                                                                                                       | `True`         |
| bootstrap_type | str                | A string of length one. Allows choosing the bootstrap type to be run. Either '11', '31', '13', or '33'. Default is '11'.                                                                                                   | `'11'`         |
| seed           | Union\[str, None\] | Option to provide a random seed. Default is None.                                                                                                                                                                          | `None`         |
| k_adj          | bool               | Whether to adjust the original coefficients with the bootstrap distribution. Default is True.                                                                                                                              | `True`         |
| G_adj          | bool               | Whether to adjust standard errors for clustering in the bootstrap. Default is True.                                                                                                                                        | `True`         |

#### Returns

| Name   | Type             | Description                                                                                                                     |
|--------|------------------|---------------------------------------------------------------------------------------------------------------------------------|
|        | pandas.DataFrame | A pd.DataFrame with bootstrapped t-statistic and p-value. The index indicates which model the estimated statistic derives from. |
