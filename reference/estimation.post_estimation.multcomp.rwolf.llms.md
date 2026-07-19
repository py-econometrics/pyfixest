# rwolf

``` python
rwolf(models, param, reps, seed, sampling_method='wild-bootstrap')
```

Compute Romano-Wolf adjusted p-values for multiple hypothesis testing.

For each model, it is assumed that tests to adjust are of the form “param = 0”. This function uses the `wildboottest()` method for running the bootstrap, hence models of type `Feiv` or `Fepois` are not supported.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| models | list\[Feols\] or FixestMulti | A list of models for which the p-values should be computed, or a FixestMulti object. Models of type `Feiv` or `Fepois` are not supported. | *required* |
| param | str | The parameter for which the p-values should be computed. | *required* |
| reps | int | The number of bootstrap replications. | *required* |
| seed | int | The seed for the random number generator. | *required* |
| sampling_method | str | Sampling method for computing resampled statistics. Users can choose either bootstrap(‘wild-bootstrap’) or randomization inference(‘ri’) | `'wild-bootstrap'` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | pd.DataFrame | A DataFrame containing estimation statistics, including the Romano-Wolf adjusted p-values. |

## Examples

``` python
import pyfixest as pf
from pyfixest.utils import get_data

data = get_data().dropna()
fit = pf.feols("Y ~ Y2 + X1 + X2", data=data)
pf.rwolf(fit, "X1", reps=9999, seed=123)

fit1 = pf.feols("Y ~ X1", data=data)
fit2 = pf.feols("Y ~ X1 + X2", data=data)
rwolf_df = pf.rwolf([fit1, fit2], "X1", reps=9999, seed=123)

# use randomization inference - dontrun as too slow
# rwolf_df = pf.rwolf([fit1, fit2], "X1", reps=9999, seed=123, sampling_method = "ri")

rwolf_df
```

|                | est0       | est1       |
|----------------|------------|------------|
| Estimate       | -1.001930  | -0.995197  |
| Std. Error     | 0.084823   | 0.082194   |
| t value        | -11.811964 | -12.107957 |
| Pr(\>\|t\|)    | 0.000000   | 0.000000   |
| 2.5%           | -1.168383  | -1.156490  |
| 97.5%          | -0.835476  | -0.833904  |
| RW Pr(\>\|t\|) | 0.000100   | 0.000100   |
