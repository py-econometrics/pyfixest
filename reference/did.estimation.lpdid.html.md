# did.estimation.lpdid { #pyfixest.did.estimation.lpdid }

```python
did.estimation.lpdid(
    data,
    yname,
    idname,
    tname,
    gname,
    vcov=None,
    pre_window=None,
    post_window=None,
    never_treated=0,
    att=True,
    xfml=None,
)
```

Local projections approach to estimation.

Estimate a Difference-in-Differences / Event Study Model via the Local
Projections Approach.

## Parameters {.doc-section .doc-section-parameters}

| Name          | Type        | Description                                                                                                                            | Default    |
|---------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------|------------|
| data          | DataFrame   | The DataFrame containing all variables.                                                                                                | _required_ |
| yname         | str         | The name of the dependent variable.                                                                                                    | _required_ |
| idname        | str         | The name of the id variable.                                                                                                           | _required_ |
| tname         | str         | Variable name for calendar period.                                                                                                     | _required_ |
| gname         | str         | Unit-specific time of initial treatment.                                                                                               | _required_ |
| vcov          | (str, dict) | The type of inference to employ. Defaults to {"CRV1": idname}. Options include "iid", "hetero", or a dictionary like {"CRV1": idname}. | `None`     |
| pre_window    | int         | The number of periods before the treatment to include in the estimation. Default is the minimum relative year in the data.             | `None`     |
| post_window   | int         | The number of periods after the treatment to include in the estimation. Default is the maximum relative year in the data.              | `None`     |
| never_treated | int         | Value in gname indicating units never treated. Default is 0.                                                                           | `0`        |
| att           | bool        | If True, estimates the pooled average treatment effect on the treated (ATT). Default is False.                                         | `True`     |
| xfml          | str         | Formula for the covariates. Not yet supported.                                                                                         | `None`     |

## Returns {.doc-section .doc-section-returns}

| Name   | Type      | Description                                  |
|--------|-----------|----------------------------------------------|
|        | DataFrame | A DataFrame with the estimated coefficients. |

## Examples {.doc-section .doc-section-examples}

```{python}
import pandas as pd
import pyfixest as pf

url = "https://raw.githubusercontent.com/py-econometrics/pyfixest/master/pyfixest/did/data/df_het.csv"
df_het = pd.read_csv(url)

fit = pf.lpdid(
    df_het,
    yname="dep_var",
    idname="unit",
    tname="year",
    gname="g",
    vcov={"CRV1": "state"},
    pre_window=-20,
    post_window=20,
    att=False
)

fit.tidy().head()
fit.iplot(figsize= [1200, 400], coord_flip=False).show()
```

To get the ATT, set `att=True`:

```{python}
fit = pf.lpdid(
    df_het,
    yname="dep_var",
    idname="unit",
    tname="year",
    gname="g",
    vcov={"CRV1": "state"},
    pre_window=-20,
    post_window=20,
    att=True
)
fit.tidy()
```