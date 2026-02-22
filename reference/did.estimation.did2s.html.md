# did.estimation.did2s { #pyfixest.did.estimation.did2s }

```python
did.estimation.did2s(
    data,
    yname,
    first_stage,
    second_stage,
    treatment,
    cluster,
    weights=None,
)
```

Estimate a Difference-in-Differences model using Gardner's two-step DID2S estimator.

## Parameters {.doc-section .doc-section-parameters}

| Name         | Type         | Description                                          | Default    |
|--------------|--------------|------------------------------------------------------|------------|
| data         | pd.DataFrame | The DataFrame containing all variables.              | _required_ |
| yname        | str          | The name of the dependent variable.                  | _required_ |
| first_stage  | str          | The formula for the first stage, starting with '~'.  | _required_ |
| second_stage | str          | The formula for the second stage, starting with '~'. | _required_ |
| treatment    | str          | The name of the treatment variable.                  | _required_ |
| cluster      | str          | The name of the cluster variable.                    | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                                                            |
|--------|--------|----------------------------------------------------------------------------------------|
|        | object | A fitted model object of class [Feols](/reference/estimation.models.feols_.Feols.qmd). |

## Examples {.doc-section .doc-section-examples}

```{python}
import pandas as pd
import numpy as np
import pyfixest as pf

url = "https://raw.githubusercontent.com/py-econometrics/pyfixest/master/pyfixest/did/data/df_het.csv"
df_het = pd.read_csv(url)
df_het.head()
```

In a first step, we estimate a classical event study model:

```{python}
# estimate the model
fit = pf.did2s(
    df_het,
    yname="dep_var",
    first_stage="~ 0 | unit + year",
    second_stage="~i(rel_year, ref=-1.0)",
    treatment="treat",
    cluster="state",
)

fit.tidy().head()
```

We can also inspect the model visually:

```{python}
fit.iplot(figsize= [1200, 400], coord_flip=False).show()
```

To estimate a pooled effect, we need to slightly update the second stage formula:

```{python}
fit = pf.did2s(
    df_het,
    yname="dep_var",
    first_stage="~ 0 | unit + year",
    second_stage="~i(treat)",
    treatment="treat",
    cluster="state"
)
fit.tidy().head()
```