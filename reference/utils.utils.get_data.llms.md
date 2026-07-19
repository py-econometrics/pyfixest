# get_data

``` python
get_data(N=1000, seed=1234, beta_type='1', error_type='1', model='Feols')
```

Create a random example data set.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| N | int | Number of observations. Default is 1000. | `1000` |
| seed | int | Seed for the random number generator. Default is 1234. | `1234` |
| beta_type | str | Type of beta coefficients. Must be one of ‘1’, ‘2’, or ‘3’. Default is ‘1’. | `'1'` |
| error_type | str | Type of error term. Must be one of ‘1’, ‘2’, or ‘3’. Default is ‘1’. | `'1'` |
| model | str | Type of the DGP. Must be either ‘Feols’ or ‘Fepois’. Default is ‘Feols’. | `'Feols'` |

## Returns

| Name | Type             | Description                             |
|------|------------------|-----------------------------------------|
|      | pandas.DataFrame | A pandas DataFrame with simulated data. |

## Raises

| Name | Type | Description |
|----|----|----|
|  | ValueError | If beta_type is not ‘1’, ‘2’, or ‘3’, or if error_type is not ‘1’, ‘2’, or ‘3’, or if model is not ‘Feols’ or ‘Fepois’. |

## Examples

``` python
import pyfixest as pf

data = pf.get_data()
data.head()
```

|  | Y | Y2 | X1 | X2 | f1 | f2 | f3 | group_id | Z1 | Z2 | weights |
|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | NaN | 2.357103 | 0.0 | 0.457858 | 15.0 | 0.0 | 7.0 | 9.0 | -0.330607 | 1.054826 | 0.661478 |
| 1 | -1.458643 | 5.163147 | NaN | -4.998406 | 6.0 | 21.0 | 4.0 | 8.0 | NaN | -4.113690 | 0.772732 |
| 2 | 0.169132 | 0.751140 | 2.0 | 1.558480 | NaN | 1.0 | 7.0 | 16.0 | 1.207778 | 0.465282 | 0.990929 |
| 3 | 3.319513 | -2.656368 | 1.0 | 1.560402 | 1.0 | 10.0 | 11.0 | 3.0 | 2.869997 | 0.467570 | 0.021123 |
| 4 | 0.134420 | -1.866416 | 2.0 | -3.472232 | 19.0 | 20.0 | 6.0 | 14.0 | 0.835819 | -3.115669 | 0.790815 |

The data set contains a continuous outcome `Y`, covariates `X1` and `X2`, fixed effects `f1`, `f2` and `f3`, an instrument `Z1`, and some missing values. Set `model="Fepois"` for a count outcome.

``` python
pf.get_data(model="Fepois")["Y"].head()
```

    0    NaN
    1    0.0
    2    2.0
    3    0.0
    4    2.0
    Name: Y, dtype: float64
