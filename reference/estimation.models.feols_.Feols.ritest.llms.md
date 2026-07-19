# Feols.ritest

``` python
ritest(
    resampvar,
    cluster=None,
    reps=100,
    type='randomization-c',
    rng=None,
    choose_algorithm='auto',
    store_ritest_statistics=False,
    level=0.95,
)
```

Conduct Randomization Inference (RI) test against a null hypothesis of `resampvar = 0`.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| resampvar | str | The name of the variable to be resampled. | *required* |
| cluster | str | The name of the cluster variable in case of cluster random assignment. If provided, `resampvar` is held constant within each `cluster`. Defaults to None. | `None` |
| reps | int | The number of randomization iterations. Defaults to 100. | `100` |
| type | str | The type of the randomization inference test. Can be “randomization-c” or “randomization-t”. Note that the “randomization-c” is much faster, while the “randomization-t” is recommended by Wu & Ding (JASA, 2021). | `'randomization-c'` |
| rng | np.random.Generator | A random number generator. Defaults to None. | `None` |
| choose_algorithm | str | The algorithm to use for the computation. Defaults to “auto”. The alternatives are “fast” and “slow”. The fast algorithm requires the optional `numba` extra (install via `pip install pyfixest[numba]`); without it, the fast path raises an `ImportError`. The slow path does not require numba. | `'auto'` |
| include_plot |  | Whether to include a plot of the distribution p-values. Defaults to False. | *required* |
| store_ritest_statistics | bool | Whether to store the simulated statistics of the RI procedure. Defaults to False. If True, stores the simulated statistics in the model object via the `ritest_statistics` attribute as a numpy array. | `False` |
| level | float | The level for the confidence interval of the randomization inference p-value. Defaults to 0.95. | `0.95` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | A pd.Series with the regression coefficient of `resampvar` and the p-value |  |
|  | of the RI test. Additionally, reports the standard error and the confidence |  |
|  | interval of the p-value. |  |

## Examples

``` python
import pyfixest as pf
data = pf.get_data()
fit = pf.feols("Y ~ X1 + X2", data=data)

# Conduct a randomization inference test for the coefficient of X1
fit.ritest("X1", reps=1000)

# use randomization-t instead of randomization-c
fit.ritest("X1", reps=1000, type="randomization-t")

# store statistics for plotting
fit.ritest("X1", reps=1000, store_ritest_statistics=True)
```

    H0                                      X1=0
    ri-type                      randomization-c
    Estimate                 -0.9929357698186863
    Pr(>|t|)                                 0.0
    Std. Error (Pr(>|t|))                    0.0
    2.5% (Pr(>|t|))                          0.0
    97.5% (Pr(>|t|))                         0.0
    dtype: object
