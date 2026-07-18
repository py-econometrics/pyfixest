# Causal Inference for the Brave and True

``` python
import pandas as pd

import pyfixest as pf
```

### Chapter 14: Panel Data and Fixed Effects

In this example we replicate the results of the great (freely available reference!) Causal Inference for the Brave and True - Chapter 14. Please refer to the original text for a detailed explanation of the data.

``` python
data_path = "https://raw.githubusercontent.com/bashtage/linearmodels/main/linearmodels/datasets/wage_panel/wage_panel.csv.bz2"
data_df = pd.read_csv(data_path)

data_df.head()
```

|     | nr  | year | black | exper | hisp | hours | married | educ | union | lwage    | expersq | occupation |
|-----|-----|------|-------|-------|------|-------|---------|------|-------|----------|---------|------------|
| 0   | 13  | 1980 | 0     | 1     | 0    | 2672  | 0       | 14   | 0     | 1.197540 | 1       | 9          |
| 1   | 13  | 1981 | 0     | 2     | 0    | 2320  | 0       | 14   | 1     | 1.853060 | 4       | 9          |
| 2   | 13  | 1982 | 0     | 3     | 0    | 2940  | 0       | 14   | 0     | 1.344462 | 9       | 9          |
| 3   | 13  | 1983 | 0     | 4     | 0    | 2960  | 0       | 14   | 0     | 1.433213 | 16      | 9          |
| 4   | 13  | 1984 | 0     | 5     | 0    | 3071  | 0       | 14   | 0     | 1.568125 | 25      | 5          |

We have a classical panel data set with units (nr) and time (year).

We are interested in estimating the effect of marriage status on log wage, using a set of controls (union, hours) and individual (nr) and year fixed effects.

``` python
panel_fit = pf.feols(
    fml="lwage ~ married + expersq + union + hours | nr + year",
    data=data_df,
    vcov={"CRV1": "nr + year"},
    demeaner=pf.MapDemeaner(backend="rust"),
)
```

``` python
pf.etable(panel_fit)
```

[TABLE]

We obtain the same results as in the book!
