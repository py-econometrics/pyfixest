# Feols.wildboottest

``` python
wildboottest(
    reps,
    cluster=None,
    param=None,
    weights_type='rademacher',
    impose_null=True,
    bootstrap_type='11',
    seed=None,
    k_adj=True,
    G_adj=True,
    parallel=False,
    return_bootstrapped_t_stats=False,
)
```

Run a wild cluster bootstrap based on an object of type “Feols”.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| reps | int | The number of bootstrap iterations to run. | *required* |
| cluster | Union\[str, None\] | The variable used for clustering. Defaults to None. If None, then uses the variable specified in the model’s `clustervar` attribute. If no `_clustervar` attribute is found, runs a heteroskedasticity- robust bootstrap. | `None` |
| param | Union\[str, None\] | A string of length one, containing the test parameter of interest. Defaults to None. | `None` |
| weights_type | str | The type of bootstrap weights. Options are ‘rademacher’, ‘mammen’, ‘webb’, or ‘normal’. Defaults to ‘rademacher’. | `'rademacher'` |
| impose_null | bool | Indicates whether to impose the null hypothesis on the bootstrap DGP. Defaults to True. | `True` |
| bootstrap_type | str | A string of length one to choose the bootstrap type. Options are ‘11’, ‘31’, ‘13’, or ‘33’. Defaults to ‘11’. | `'11'` |
| seed | Union\[int, None\] | An option to provide a random seed. Defaults to None. | `None` |
| k_adj | bool | Indicates whether to apply a small sample adjustment for the number of observations and covariates. Defaults to True. | `True` |
| G_adj | bool | Indicates whether to apply a small sample adjustment for the number of clusters. Defaults to True. | `True` |
| parallel | bool | Indicates whether to run the bootstrap in parallel. Defaults to False. | `False` |
| seed | Union\[str, None\] | An option to provide a random seed. Defaults to None. | `None` |
| return_bootstrapped_t_stats | bool, optional: | If True, the method returns a tuple of the regular output and the bootstrapped t-stats. Defaults to False. | `False` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | pd.DataFrame | A DataFrame with the original, non-bootstrapped t-statistic and bootstrapped p-value, along with the bootstrap type, inference type (HC vs CRV), and whether the null hypothesis was imposed on the bootstrap DGP. If `return_bootstrapped_t_stats` is True, the method returns a tuple of the regular output and the bootstrapped t-stats. |

## Examples

``` python
import re
import pyfixest as pf

data = pf.get_data()
fit = pf.feols("Y ~ X1 + X2 | f1", data)

fit.wildboottest(
    param = "X1",
    reps=1000,
    seed = 822
)

fit.wildboottest(
    param = "X1",
    reps=1000,
    seed = 822,
    bootstrap_type = "31"
)
```

    param                    X1
    t value          -14.843741
    Pr(>|t|)                0.0
    bootstrap_type           31
    inference                HC
    impose_null            True
    ssc                       1
    dtype: object
