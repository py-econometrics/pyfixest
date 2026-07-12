<!-- Generated from docs/reference/report.qplot.qmd; do not edit. -->

# report.qplot

```python
report.qplot(models, rename_models=None, figsize=None, ncol=None, nrow=None)
```

Plot regression quantiles.

## Parameters

| Name          | Type                                                                     | Description                                                                                            | Default    |
|---------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|------------|
| models        | A supported model object (Feols, Fepois, Feiv, FixestMulti) or a list of | Feols, Fepois & Feiv models.                                                                           | _required_ |
| figsize       | tuple or None                                                            | The size of the figure. If None, the default size is (10, 6).                                          | `None`     |
| rename_models | dict                                                                     | A dictionary to rename the models. The keys are the original model names and the values the new names. | `None`     |
| ncol          | int                                                                      | Number of columns of subplots. Default is None. Note: cannot be set jointly with nrow argument.        | `None`     |
| nrow          | int                                                                      | Number of rows of subplots. Default is None. Note: cannot be set jointly with ncol argument.           | `None`     |

## Returns

| Name   | Type   | Description           |
|--------|--------|-----------------------|
|        | object | A matplotplit figure. |
