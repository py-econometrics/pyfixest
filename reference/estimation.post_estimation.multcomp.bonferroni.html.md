# estimation.post_estimation.multcomp.bonferroni { #pyfixest.estimation.post_estimation.multcomp.bonferroni }

```python
estimation.post_estimation.multcomp.bonferroni(models, param)
```

Compute Bonferroni adjusted p-values for multiple hypothesis testing.

For each model, it is assumed that tests to adjust are of the form
"param = 0".

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type                                                                     | Description                                              | Default    |
|--------|--------------------------------------------------------------------------|----------------------------------------------------------|------------|
| models | A supported model object (Feols, Fepois, Feiv, FixestMulti) or a list of | Feols, Fepois & Feiv models.                             | _required_ |
| param  | str                                                                      | The parameter for which the p-values should be adjusted. | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type         | Description                                                                               |
|--------|--------------|-------------------------------------------------------------------------------------------|
|        | pd.DataFrame | A DataFrame containing estimation statistics, including the Bonferroni adjusted p-values. |

## Examples {.doc-section .doc-section-examples}

```{python}
import pyfixest as pf
from pyfixest.utils import get_data

data = get_data().dropna()
fit1 = pf.feols("Y ~ X1", data=data)
fit2 = pf.feols("Y ~ X1 + X2", data=data)
bonf_df = pf.bonferroni([fit1, fit2], param="X1")
bonf_df
```