<!-- Generated from docs/reference/did.saturated_twfe.SaturatedEventStudy.qmd; do not edit. -->

# did.saturated_twfe.SaturatedEventStudy

```python
did.saturated_twfe.SaturatedEventStudy(
    data,
    yname,
    idname,
    tname,
    gname,
    att=True,
    cluster=None,
    xfml=None,
    display_warning=True,
)
```

Saturated event study with cohort-specific effect curves.

## Attributes

| Name            | Type         | Description                                    |
|-----------------|--------------|------------------------------------------------|
| data            | pd.DataFrame | Dataframe containing the data.                 |
| yname           | str          | Name of the outcome variable.                  |
| idname          | str          | Name of the unit identifier variable.          |
| tname           | str          | Name of the time variable.                     |
| gname           | str          | Name of the treatment variable.                |
| cluster         | str          | The name of the cluster variable.              |
| xfml            | str          | Additional covariates to include in the model. |
| att             | bool         | Whether to use the average treatment effect.   |
| display_warning | bool         | Whether to display (some) warning messages.    |

## Methods

| Name | Description |
| --- | --- |
| [aggregate](#pyfixest.did.saturated_twfe.SaturatedEventStudy.aggregate) | Aggregate the fully interacted event study estimates by relative time, cohort, and time. |
| [estimate](#pyfixest.did.saturated_twfe.SaturatedEventStudy.estimate) | Estimate the model. |
| [iplot](#pyfixest.did.saturated_twfe.SaturatedEventStudy.iplot) | Plot DID estimates. |
| [iplot_aggregate](#pyfixest.did.saturated_twfe.SaturatedEventStudy.iplot_aggregate) | Plot the aggregated estimates. |
| [summary](#pyfixest.did.saturated_twfe.SaturatedEventStudy.summary) | Get summary table. |
| [test_treatment_heterogeneity](#pyfixest.did.saturated_twfe.SaturatedEventStudy.test_treatment_heterogeneity) | Test for treatment heterogeneity in the event study design. |
| [tidy](#pyfixest.did.saturated_twfe.SaturatedEventStudy.tidy) | Tidy result dataframe. |
| [vcov](#pyfixest.did.saturated_twfe.SaturatedEventStudy.vcov) | Get the covariance matrix. |

### aggregate

```python
did.saturated_twfe.SaturatedEventStudy.aggregate(
    agg='period',
    weighting='shares',
)
```

Aggregate the fully interacted event study estimates by relative time, cohort, and time.

#### Parameters

| Name      | Type   | Description                                                                                                                                                                                                                                                                                                | Default    |
|-----------|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| agg       | str    | The type of aggregation to perform. Can be either "att" or "cohort" or "period".     Default is "att". If "att", computes the average treatment effect on the treated.     If "cohort", computes the average treatment effect by cohort. If "period",     computes the average treatment effect by period. | `'period'` |
| weighting | str    | The type of weighting to use. Can be either 'shares' or 'variance'.                                                                                                                                                                                                                                        | `'shares'` |

#### Returns

| Name   | Type      | Description                                   |
|--------|-----------|-----------------------------------------------|
|        | pd.Series | A Series containing the aggregated estimates. |

### estimate

```python
did.saturated_twfe.SaturatedEventStudy.estimate()
```

Estimate the model.

#### Returns

| Name   | Type   | Description                    |
|--------|--------|--------------------------------|
|        | Feols  | The fitted Feols model object. |

### iplot

```python
did.saturated_twfe.SaturatedEventStudy.iplot()
```

Plot DID estimates.

### iplot_aggregate

```python
did.saturated_twfe.SaturatedEventStudy.iplot_aggregate(
    agg='period',
    weighting='shares',
)
```

Plot the aggregated estimates.

#### Parameters

| Name      | Type   | Description                                                                                                                                                                                                                                                                                    | Default    |
|-----------|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| agg       | str    | The type of aggregation to perform. Can be either "att" or "cohort" or "period". Default is "att". If "att", computes the average treatment effect on the treated. If "cohort", computes the average treatment effect by cohort. If "period", computes the average treatment effect by period. | `'period'` |
| weighting | str    | The type of weighting to use. Can be either 'shares' or 'variance'.                                                                                                                                                                                                                            | `'shares'` |

#### Returns

| Name   | Type   | Description   |
|--------|--------|---------------|
|        | None   |               |

### summary

```python
did.saturated_twfe.SaturatedEventStudy.summary()
```

Get summary table.

### test_treatment_heterogeneity

```python
did.saturated_twfe.SaturatedEventStudy.test_treatment_heterogeneity()
```

Test for treatment heterogeneity in the event study design.

#### Parameters

| Name   | Type   | Description                                                                                                                                                                                                                       | Default    |
|--------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| by     | str    | The type of test to perform. Can be either "cohort" or "time".     Default is "cohort". If "cohort", tests for treatment heterogeneity     across cohorts as in Lal (2025). See https://arxiv.org/abs/2503.05125     for details. | _required_ |

### tidy

```python
did.saturated_twfe.SaturatedEventStudy.tidy()
```

Tidy result dataframe.

### vcov

```python
did.saturated_twfe.SaturatedEventStudy.vcov()
```

Get the covariance matrix.

#### Returns

| Name   | Type         | Description                                   |
|--------|--------------|-----------------------------------------------|
|        | pd.DataFrame | A DataFrame containing the covariance matrix. |
