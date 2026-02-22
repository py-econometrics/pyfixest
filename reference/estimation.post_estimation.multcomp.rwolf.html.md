# estimation.post_estimation.multcomp.rwolf { #pyfixest.estimation.post_estimation.multcomp.rwolf }

```python
estimation.post_estimation.multcomp.rwolf(
    models,
    param,
    reps,
    seed,
    sampling_method='wild-bootstrap',
)
```

Compute Romano-Wolf adjusted p-values for multiple hypothesis testing.

For each model, it is assumed that tests to adjust are of the form
"param = 0". This function uses the `wildboottest()` method for running the
bootstrap, hence models of type `Feiv` or `Fepois` are not supported.

## Parameters {.doc-section .doc-section-parameters}

| Name            | Type                         | Description                                                                                                                               | Default            |
|-----------------|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| models          | list\[Feols\] or FixestMulti | A list of models for which the p-values should be computed, or a FixestMulti object. Models of type `Feiv` or `Fepois` are not supported. | _required_         |
| param           | str                          | The parameter for which the p-values should be computed.                                                                                  | _required_         |
| reps            | int                          | The number of bootstrap replications.                                                                                                     | _required_         |
| seed            | int                          | The seed for the random number generator.                                                                                                 | _required_         |
| sampling_method | str                          | Sampling method for computing resampled statistics. Users can choose either bootstrap('wild-bootstrap') or randomization inference('ri')  | `'wild-bootstrap'` |

## Returns {.doc-section .doc-section-returns}

| Name   | Type         | Description                                                                                |
|--------|--------------|--------------------------------------------------------------------------------------------|
|        | pd.DataFrame | A DataFrame containing estimation statistics, including the Romano-Wolf adjusted p-values. |

## Examples {.doc-section .doc-section-examples}

```{python}
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