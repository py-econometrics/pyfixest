# did.estimation.did2s

``` python
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

Estimate a Difference-in-Differences model using Gardner’s two-step DID2S estimator.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| data | pd.DataFrame | The DataFrame containing all variables. | *required* |
| yname | str | The name of the dependent variable. | *required* |
| first_stage | str | The formula for the first stage, starting with ‘~’. | *required* |
| second_stage | str | The formula for the second stage, starting with ‘~’. | *required* |
| treatment | str | The name of the treatment variable. | *required* |
| cluster | str | The name of the cluster variable. | *required* |

## Returns

| Name | Type | Description |
|----|----|----|
|  | object | A fitted model object of class [Feols](../reference/estimation.models.feols_.Feols.llms.md). |

## Examples

``` python
import pandas as pd
import numpy as np
import pyfixest as pf

url = "https://raw.githubusercontent.com/py-econometrics/pyfixest/master/pyfixest/did/data/df_het.csv"
df_het = pd.read_csv(url)
df_het.head()
```

|  | unit | state | group | unit_fe | g | year | year_fe | treat | rel_year | rel_year_binned | error | te | te_dynamic | dep_var |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 1 | 33 | Group 2 | 7.043016 | 2010 | 1990 | 0.066159 | False | -20.0 | -6 | -0.086466 | 0 | 0.0 | 7.022709 |
| 1 | 1 | 33 | Group 2 | 7.043016 | 2010 | 1991 | -0.030980 | False | -19.0 | -6 | 0.766593 | 0 | 0.0 | 7.778628 |
| 2 | 1 | 33 | Group 2 | 7.043016 | 2010 | 1992 | -0.119607 | False | -18.0 | -6 | 1.512968 | 0 | 0.0 | 8.436377 |
| 3 | 1 | 33 | Group 2 | 7.043016 | 2010 | 1993 | 0.126321 | False | -17.0 | -6 | 0.021870 | 0 | 0.0 | 7.191207 |
| 4 | 1 | 33 | Group 2 | 7.043016 | 2010 | 1994 | -0.106921 | False | -16.0 | -6 | -0.017603 | 0 | 0.0 | 6.918492 |

In a first step, we estimate a classical event study model:

``` python
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

|  | Estimate | Std. Error | t value | Pr(\>\|t\|) | 2.5% | 97.5% |
|----|----|----|----|----|----|----|
| Coefficient |  |  |  |  |  |  |
| rel_year::-inf | 3.133615e-08 | 5.155857e-09 | 6.077778 | 1.228051e-09 | 2.123060e-08 | 4.144171e-08 |
| rel_year::-20.0 | -5.822583e-02 | 3.580900e-02 | -1.626011 | 1.039541e-01 | -1.284120e-01 | 1.196035e-02 |
| rel_year::-19.0 | -6.032235e-03 | 3.034072e-02 | -0.198816 | 8.424072e-01 | -6.550051e-02 | 5.343604e-02 |
| rel_year::-18.0 | -6.152322e-03 | 3.509400e-02 | -0.175310 | 8.608370e-01 | -7.493709e-02 | 6.263245e-02 |
| rel_year::-17.0 | -1.253329e-02 | 2.483369e-02 | -0.504689 | 6.137796e-01 | -6.120770e-02 | 3.614111e-02 |

We can also inspect the model visually:

``` python
fit.iplot(figsize= [1200, 400], coord_flip=False).show()
```

To estimate a pooled effect, we need to slightly update the second stage formula:

``` python
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

|             | Estimate | Std. Error | t value   | Pr(\>\|t\|) | 2.5%     | 97.5%    |
|-------------|----------|------------|-----------|-------------|----------|----------|
| Coefficient |          |            |           |             |          |          |
| treat::True | 2.230482 | 0.024709   | 90.271433 | 0.0         | 2.182052 | 2.278911 |
