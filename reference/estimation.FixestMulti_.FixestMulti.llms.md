# estimation.FixestMulti\_.FixestMulti

``` python
estimation.FixestMulti_.FixestMulti(config, parsed, data, context)
```

Results container holding every model fitted by one public-API call.

## Attributes

| Name | Description |
|----|----|
| [FixestFormulaDict](#pyfixest.estimation.FixestMulti_.FixestMulti.FixestFormulaDict) | Parsed formula dict keyed by fixed-effects spec. |

## Methods

| Name | Description |
|----|----|
| [confint](#pyfixest.estimation.FixestMulti_.FixestMulti.confint) | Obtain confidence intervals for the fitted models. |
| [fetch_model](#pyfixest.estimation.FixestMulti_.FixestMulti.fetch_model) | Fetch a model of class Feols from the Fixest class. |
| [tidy](#pyfixest.estimation.FixestMulti_.FixestMulti.tidy) | Return the results of an estimation using `feols()` as a tidy Pandas DataFrame. |
| [to_list](#pyfixest.estimation.FixestMulti_.FixestMulti.to_list) | Return a list of all fitted models. |
| [vcov](#pyfixest.estimation.FixestMulti_.FixestMulti.vcov) | Update regression inference “on the fly”. |
| [wildboottest](#pyfixest.estimation.FixestMulti_.FixestMulti.wildboottest) | Run a wild cluster bootstrap for all regressions in the Fixest object. |

### confint

``` python
estimation.FixestMulti_.FixestMulti.confint()
```

Obtain confidence intervals for the fitted models.

#### Returns

| Name | Type | Description |
|----|----|----|
|  | pandas.Series | A pd.Series with coefficient names and confidence intervals. The key indicates which models the estimated statistic derives from. |

### fetch_model

``` python
estimation.FixestMulti_.FixestMulti.fetch_model(i, print_fml=True)
```

Fetch a model of class Feols from the Fixest class.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| i | int or str | The index of the model to fetch. | *required* |
| print_fml | bool | Whether to print the formula of the model. Default is True. | `True` |

#### Returns

| Name | Type            | Description |
|------|-----------------|-------------|
|      | A Feols object. |             |

### tidy

``` python
estimation.FixestMulti_.FixestMulti.tidy()
```

Return the results of an estimation using `feols()` as a tidy Pandas DataFrame.

#### Returns

| Name | Type | Description |
|----|----|----|
|  | pandas.DataFrame or str | A tidy DataFrame with the following columns: - fml: the formula used to generate the results - Coefficient: the names of the coefficients - Estimate: the estimated coefficients - Std. Error: the standard errors of the estimated coefficients - t value: the t-values of the estimated coefficients - Pr(\>\|t\|): the p-values of the estimated coefficients - 2.5%: the lower bound of the 95% confidence interval - 97.5%: the upper bound of the 95% confidence interval If `type` is set to “markdown”, the resulting DataFrame will be returned as a markdown-formatted string with three decimal places. |

### to_list

``` python
estimation.FixestMulti_.FixestMulti.to_list()
```

Return a list of all fitted models.

#### Returns

| Name | Type                                                  | Description |
|------|-------------------------------------------------------|-------------|
|      | A list of all fitted models of types Feols or Fepois. |             |

### vcov

``` python
estimation.FixestMulti_.FixestMulti.vcov(vcov, vcov_kwargs=None)
```

Update regression inference “on the fly”.

By calling vcov() on a “Fixest” object, all inference procedures applied to the “Fixest” object are replaced with the variance-covariance matrix specified via the method.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| vcov | Union\[str, dict\[str, str\]\]) | A string or dictionary specifying the type of variance-covariance matrix to use for inference. - If a string, can be one of “iid”, “hetero”, “HC1”, “HC2”, “HC3”. - If a dictionary, it should have the format {“CRV1”: “clustervar”} for CRV1 inference or {“CRV3”: “clustervar”} for CRV3 inference. | *required* |
| vcov_kwargs | Optional\[dict\[str, any\]\] | Additional keyword arguments for the variance-covariance matrix. | `None` |

#### Returns

| Name | Type                                                      | Description |
|------|-----------------------------------------------------------|-------------|
|      | An instance of the "Fixest" class with updated inference. |             |

### wildboottest

``` python
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

| Name | Type | Description | Default |
|----|----|----|----|
| B | int | The number of bootstrap iterations to run. | *required* |
| param | Union\[str, None\] | A string of length one, containing the test parameter of interest. Default is None. | `None` |
| cluster | Union\[str, None\] | The name of the cluster variable. Default is None. If None, uses the `self._clustervar` attribute as the cluster variable. If the `self._clustervar` attribute is None, a heteroskedasticity-robust wild bootstrap is run. | `None` |
| weights_type | str | The type of bootstrap weights. Either ‘rademacher’, ‘mammen’, ‘webb’, or ‘normal’. Default is ‘rademacher’. | `'rademacher'` |
| impose_null | bool | Should the null hypothesis be imposed on the bootstrap dgp, or not? Default is True. | `True` |
| bootstrap_type | str | A string of length one. Allows choosing the bootstrap type to be run. Either ‘11’, ‘31’, ‘13’, or ‘33’. Default is ‘11’. | `'11'` |
| seed | Union\[str, None\] | Option to provide a random seed. Default is None. | `None` |
| k_adj | bool | Whether to adjust the original coefficients with the bootstrap distribution. Default is True. | `True` |
| G_adj | bool | Whether to adjust standard errors for clustering in the bootstrap. Default is True. | `True` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | pandas.DataFrame | A pd.DataFrame with bootstrapped t-statistic and p-value. The index indicates which model the estimated statistic derives from. |
