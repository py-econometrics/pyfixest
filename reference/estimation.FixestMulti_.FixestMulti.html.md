# estimation.FixestMulti_.FixestMulti { #pyfixest.estimation.FixestMulti_.FixestMulti }

```python
estimation.FixestMulti_.FixestMulti(
    data,
    copy_data,
    store_data,
    lean,
    fixef_tol,
    fixef_maxiter,
    weights_type,
    use_compression,
    reps,
    seed,
    split,
    fsplit,
    separation_check=None,
    context=0,
    quantreg_method='fn',
    quantreg_multi_method='cfm1',
)
```

A class to estimate multiple regression models with fixed effects.

## Methods

| Name | Description |
| --- | --- |
| [coef](#pyfixest.estimation.FixestMulti_.FixestMulti.coef) | Obtain the coefficients of the fitted models. |
| [confint](#pyfixest.estimation.FixestMulti_.FixestMulti.confint) | Obtain confidence intervals for the fitted models. |
| [fetch_model](#pyfixest.estimation.FixestMulti_.FixestMulti.fetch_model) | Fetch a model of class Feols from the Fixest class. |
| [pvalue](#pyfixest.estimation.FixestMulti_.FixestMulti.pvalue) | Obtain the p-values of the fitted models. |
| [se](#pyfixest.estimation.FixestMulti_.FixestMulti.se) | Obtain the standard errors of the fitted models. |
| [tidy](#pyfixest.estimation.FixestMulti_.FixestMulti.tidy) | Return the results of an estimation using `feols()` as a tidy Pandas DataFrame. |
| [to_list](#pyfixest.estimation.FixestMulti_.FixestMulti.to_list) | Return a list of all fitted models. |
| [tstat](#pyfixest.estimation.FixestMulti_.FixestMulti.tstat) | Obtain the t-statistics of the fitted models. |
| [vcov](#pyfixest.estimation.FixestMulti_.FixestMulti.vcov) | Update regression inference "on the fly". |
| [wildboottest](#pyfixest.estimation.FixestMulti_.FixestMulti.wildboottest) | Run a wild cluster bootstrap for all regressions in the Fixest object. |

### coef { #pyfixest.estimation.FixestMulti_.FixestMulti.coef }

```python
estimation.FixestMulti_.FixestMulti.coef()
```

Obtain the coefficients of the fitted models.

#### Returns {.doc-section .doc-section-returns}

| Name   | Type          | Description                                                                                                            |
|--------|---------------|------------------------------------------------------------------------------------------------------------------------|
|        | pandas.Series | A pd.Series with coefficient names and Estimates. The key indicates which models the estimated statistic derives from. |

### confint { #pyfixest.estimation.FixestMulti_.FixestMulti.confint }

```python
estimation.FixestMulti_.FixestMulti.confint()
```

Obtain confidence intervals for the fitted models.

#### Returns {.doc-section .doc-section-returns}

| Name   | Type          | Description                                                                                                                       |
|--------|---------------|-----------------------------------------------------------------------------------------------------------------------------------|
|        | pandas.Series | A pd.Series with coefficient names and confidence intervals. The key indicates which models the estimated statistic derives from. |

### fetch_model { #pyfixest.estimation.FixestMulti_.FixestMulti.fetch_model }

```python
estimation.FixestMulti_.FixestMulti.fetch_model(i, print_fml=True)
```

Fetch a model of class Feols from the Fixest class.

#### Parameters {.doc-section .doc-section-parameters}

| Name      | Type       | Description                                                 | Default    |
|-----------|------------|-------------------------------------------------------------|------------|
| i         | int or str | The index of the model to fetch.                            | _required_ |
| print_fml | bool       | Whether to print the formula of the model. Default is True. | `True`     |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type            | Description   |
|--------|-----------------|---------------|
|        | A Feols object. |               |

### pvalue { #pyfixest.estimation.FixestMulti_.FixestMulti.pvalue }

```python
estimation.FixestMulti_.FixestMulti.pvalue()
```

Obtain the p-values of the fitted models.

#### Returns {.doc-section .doc-section-returns}

| Name   | Type          | Description                                                                                                           |
|--------|---------------|-----------------------------------------------------------------------------------------------------------------------|
|        | pandas.Series | A pd.Series with coefficient names and p-values. The key indicates which models the estimated statistic derives from. |

### se { #pyfixest.estimation.FixestMulti_.FixestMulti.se }

```python
estimation.FixestMulti_.FixestMulti.se()
```

Obtain the standard errors of the fitted models.

#### Returns {.doc-section .doc-section-returns}

| Name   | Type          | Description                                                                                                                           |
|--------|---------------|---------------------------------------------------------------------------------------------------------------------------------------|
|        | pandas.Series | A pd.Series with coefficient names and standard error estimates. The key indicates which models the estimated statistic derives from. |

### tidy { #pyfixest.estimation.FixestMulti_.FixestMulti.tidy }

```python
estimation.FixestMulti_.FixestMulti.tidy()
```

Return the results of an estimation using `feols()` as a tidy Pandas DataFrame.

#### Returns {.doc-section .doc-section-returns}

| Name   | Type                    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|--------|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | pandas.DataFrame or str | A tidy DataFrame with the following columns: - fml: the formula used to generate the results - Coefficient: the names of the coefficients - Estimate: the estimated coefficients - Std. Error: the standard errors of the estimated coefficients - t value: the t-values of the estimated coefficients - Pr(>\|t\|): the p-values of the estimated coefficients - 2.5%: the lower bound of the 95% confidence interval - 97.5%: the upper bound of the 95% confidence interval If `type` is set to "markdown", the resulting DataFrame will be returned as a markdown-formatted string with three decimal places. |

### to_list { #pyfixest.estimation.FixestMulti_.FixestMulti.to_list }

```python
estimation.FixestMulti_.FixestMulti.to_list()
```

Return a list of all fitted models.

#### Returns {.doc-section .doc-section-returns}

| Name   | Type                                                  | Description   |
|--------|-------------------------------------------------------|---------------|
|        | A list of all fitted models of types Feols or Fepois. |               |

### tstat { #pyfixest.estimation.FixestMulti_.FixestMulti.tstat }

```python
estimation.FixestMulti_.FixestMulti.tstat()
```

Obtain the t-statistics of the fitted models.

#### Returns {.doc-section .doc-section-returns}

| Name   | Type                                                           | Description                                                          |
|--------|----------------------------------------------------------------|----------------------------------------------------------------------|
|        | A pd.Series with coefficient names and estimated t-statistics. | The key indicates which models the estimated statistic derives from. |

### vcov { #pyfixest.estimation.FixestMulti_.FixestMulti.vcov }

```python
estimation.FixestMulti_.FixestMulti.vcov(vcov, vcov_kwargs=None)
```

Update regression inference "on the fly".

By calling vcov() on a "Fixest" object, all inference procedures applied
to the "Fixest" object are replaced with the variance-covariance matrix
specified via the method.

#### Parameters {.doc-section .doc-section-parameters}

| Name        | Type                            | Description                                                                                                                                                                                                                                                                                            | Default    |
|-------------|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| vcov        | Union\[str, dict\[str, str\]\]) | A string or dictionary specifying the type of variance-covariance matrix to use for inference. - If a string, can be one of "iid", "hetero", "HC1", "HC2", "HC3". - If a dictionary, it should have the format {"CRV1": "clustervar"} for CRV1 inference or {"CRV3": "clustervar"} for CRV3 inference. | _required_ |
| vcov_kwargs | Optional\[dict\[str, any\]\]    | Additional keyword arguments for the variance-covariance matrix.                                                                                                                                                                                                                                       | `None`     |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type                                                        | Description   |
|--------|-------------------------------------------------------------|---------------|
|        | An instance of the \"Fixest\" class with updated inference. |               |

### wildboottest { #pyfixest.estimation.FixestMulti_.FixestMulti.wildboottest }

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

#### Parameters {.doc-section .doc-section-parameters}

| Name           | Type               | Description                                                                                                                                                                                                                | Default        |
|----------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| B              | int                | The number of bootstrap iterations to run.                                                                                                                                                                                 | _required_     |
| param          | Union\[str, None\] | A string of length one, containing the test parameter of interest. Default is None.                                                                                                                                        | `None`         |
| cluster        | Union\[str, None\] | The name of the cluster variable. Default is None. If None, uses the `self._clustervar` attribute as the cluster variable. If the `self._clustervar` attribute is None, a heteroskedasticity-robust wild bootstrap is run. | `None`         |
| weights_type   | str                | The type of bootstrap weights. Either 'rademacher', 'mammen', 'webb', or 'normal'. Default is 'rademacher'.                                                                                                                | `'rademacher'` |
| impose_null    | bool               | Should the null hypothesis be imposed on the bootstrap dgp, or not? Default is True.                                                                                                                                       | `True`         |
| bootstrap_type | str                | A string of length one. Allows choosing the bootstrap type to be run. Either '11', '31', '13', or '33'. Default is '11'.                                                                                                   | `'11'`         |
| seed           | Union\[str, None\] | Option to provide a random seed. Default is None.                                                                                                                                                                          | `None`         |
| k_adj          | bool               | Whether to adjust the original coefficients with the bootstrap distribution. Default is True.                                                                                                                              | `True`         |
| G_adj          | bool               | Whether to adjust standard errors for clustering in the bootstrap. Default is True.                                                                                                                                        | `True`         |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type             | Description                                                                                                                     |
|--------|------------------|---------------------------------------------------------------------------------------------------------------------------------|
|        | pandas.DataFrame | A pd.DataFrame with bootstrapped t-statistic and p-value. The index indicates which model the estimated statistic derives from. |