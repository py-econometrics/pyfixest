# estimation.post_estimation.multcomp.bonferroni

``` python
estimation.post_estimation.multcomp.bonferroni(models, param)
```

Compute Bonferroni adjusted p-values for multiple hypothesis testing.

For each model, it is assumed that tests to adjust are of the form “param = 0”.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| models | A supported model object (Feols, Fepois, Feiv, FixestMulti) or a list of | Feols, Fepois & Feiv models. | *required* |
| param | str | The parameter for which the p-values should be adjusted. | *required* |

## Returns

| Name | Type | Description |
|----|----|----|
|  | pd.DataFrame | A DataFrame containing estimation statistics, including the Bonferroni adjusted p-values. |

## Examples

``` python
import pyfixest as pf
from pyfixest.utils import get_data

data = get_data().dropna()
fit1 = pf.feols("Y ~ X1", data=data)
fit2 = pf.feols("Y ~ X1 + X2", data=data)
bonf_df = pf.bonferroni([fit1, fit2], param="X1")
bonf_df
```

|                        | est0       | est1       |
|------------------------|------------|------------|
| Estimate               | -1.001930  | -0.995197  |
| Std. Error             | 0.084823   | 0.082194   |
| t value                | -11.811964 | -12.107957 |
| Pr(\>\|t\|)            | 0.000000   | 0.000000   |
| 2.5%                   | -1.168383  | -1.156490  |
| 97.5%                  | -0.835476  | -0.833904  |
| Bonferroni Pr(\>\|t\|) | 0.000000   | 0.000000   |
