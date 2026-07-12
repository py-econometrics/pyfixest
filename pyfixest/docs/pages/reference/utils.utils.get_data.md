<!-- Generated from docs/reference/utils.utils.get_data.qmd; do not edit. -->

# utils.utils.get_data

```python
utils.utils.get_data(
    N=1000,
    seed=1234,
    beta_type='1',
    error_type='1',
    model='Feols',
)
```

Create a random example data set.

## Parameters

| Name       | Type   | Description                                                                 | Default   |
|------------|--------|-----------------------------------------------------------------------------|-----------|
| N          | int    | Number of observations. Default is 1000.                                    | `1000`    |
| seed       | int    | Seed for the random number generator. Default is 1234.                      | `1234`    |
| beta_type  | str    | Type of beta coefficients. Must be one of '1', '2', or '3'. Default is '1'. | `'1'`     |
| error_type | str    | Type of error term. Must be one of '1', '2', or '3'. Default is '1'.        | `'1'`     |
| model      | str    | Type of the DGP. Must be either 'Feols' or 'Fepois'. Default is 'Feols'.    | `'Feols'` |

## Returns

| Name   | Type             | Description                             |
|--------|------------------|-----------------------------------------|
|        | pandas.DataFrame | A pandas DataFrame with simulated data. |

## Raises

| Name   | Type       | Description                                                                                                             |
|--------|------------|-------------------------------------------------------------------------------------------------------------------------|
|        | ValueError | If beta_type is not '1', '2', or '3', or if error_type is not '1', '2', or '3', or if model is not 'Feols' or 'Fepois'. |
