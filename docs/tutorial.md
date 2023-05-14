# Getting Started with PyFixest

In a first step, we load the module and some example data:

```py
from pyfixest import Fixest
from pyfixest.utils import get_data

data = get_data()
data.head()
>>> data.head()
#            Y        X1  X2  X3  ...        19  group_id         Y2        Z1
# 0        NaN  0.471435   0   6  ... -1.546906         3  -1.568085  0.971477
# 1  -1.470802       NaN   4   6  ...  2.390961        20  -2.418717       NaN
# 2  -6.429899  0.076200   4   8  ...  1.545659        21  -6.491542 -1.122705
# 3 -15.911375 -0.974236   4   8  ...  0.039513        22 -14.777766 -1.381387
# 4  -6.537525  0.464392   3   8  ... -0.511881         6  -7.470515  0.327149
#
# [5 rows x 24 columns]

```

We then initiate an object of type `Fixest`.

```py
fixest = Fixest(data = data)
#<pyfixest.fixest.Fixest object at 0x00000216D5873070>
```

For this object, we can now estimate a fixed effects regression via the `.feols()` method. `.feols()` has two arguments: a two-sided model formula, and the type of inference.

```py
fixest.feols("Y~X1 | X2", vcov = "HC1")
```
The first part of the formula contains the dependent variable and "regular" covariates, while the second part contains fixed effects.

Estimation results can be accessed via a `.summary()` or `.tidy()` method:

```py
fixest.summary()
# ###
#
# Model:  OLS
# Dep. var.:  Y
# Fixed effects:  X2
# Inference:  HC1
# Observations:  1998
#
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.260472    0.175458 -1.484525  0.137828
# ---
```

Supported covariance types are "iid", "HC1-3", CRV1 and CRV3 (one-way clustering). Inference can be adjusted "on-the-fly" via the
`.vcov()` method:

```py

fixest.vcov({'CRV1':'group_id'}).summary()
# >>> fixest.vcov({'CRV1':'group_id'}).summary()
#
# ###
#
# Model:  OLS
# Dep. var.:  Y
# Fixed effects:  X2
# Inference:  {'CRV1': 'group_id'}
# Observations:  1998
#
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.260472    0.163874 -1.589472  0.125042
# ---
```

It is also possible to run a wild (cluster) bootstrap after estimation (via the [wildboottest module](https://github.com/s3alfisc/wildboottest)):

```py
fixest = Fixest(data = data)
fixest.feols("Y~ csw(X1, X2, X3)", vcov = {"CRV1":"group_id"})
fixest.wildboottest(param = "X1", B = 999)

#              param   t value  Pr(>|t|)
# fml
# Y ~ X1          X1  -1.65358  0.108108
# Y ~ X1+X2       X1 -1.617177  0.113113
# Y ~ X1+X2+X3    X1  0.388201  0.707708
```

Note that the wild bootstrap currently does not support fixed effects in the regression model. Supporting fixed effects is work in progress.

It is also possible to estimate instrumental variable models with *one* endogenous and *one* exogeneous variable via three-part syntax:

```python
fixest = Fixest(data = data)
fixest.feols("Y~ X1 | X2 | X1 ~ Z1")
fixest.summary()
# ###
#
# Model:  IV
# Dep. var.:  Y
# Fixed effects:  X2
# Inference:  {'CRV1': 'X2'}
# Observations:  1998
#
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.259964     0.19729 -1.317671  0.258015
# ---
```

`PyFixest` supports a range of multiple estimation functionality: `sw`, `sw0`, `csw`, `csw0`, and multiple dependent variables. Note that every new call of `.feols()` attaches new regression results the `Fixest` object.

```py
fixest.feols("Y~X1 | csw0(X3, X4)", vcov = "HC1").summary()

# >>> fixest.feols("Y~X1 | csw0(X3, X4)", vcov = "HC1").summary()
#
# ###
#
# Model:  IV
# Dep. var.:  Y
# Fixed effects:  X2
# Inference:  {'CRV1': 'X2'}
# Observations:  1998
#
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.259964     0.19729 -1.317671  0.258015
# ---
# ###
#
# Model:  OLS
# Dep. var.:  Y
# Inference:  HC1
# Observations:  1998
#
#            Estimate  Std. Error    t value  Pr(>|t|)
# Intercept -3.941395    0.184974 -21.307836  0.000000
#        X1 -0.273096    0.175432  -1.556710  0.119698
# ---
# ###
#
# Model:  OLS
# Dep. var.:  Y
# Fixed effects:  X3
# Inference:  HC1
# Observations:  1998
#
#     Estimate  Std. Error  t value  Pr(>|t|)
# X1  0.034788    0.117487 0.296105   0.76718
# ---
# ###
#
# Model:  OLS
# Dep. var.:  Y
# Fixed effects:  X3+X4
# Inference:  HC1
# Observations:  1998
#
#     Estimate  Std. Error  t value  Pr(>|t|)
# X1  0.049263    0.106979 0.460492  0.645213
# ---

```

# TWFE Event Study

Here, we follow an example from the [LOST](https://lost-stats.github.io/Model_Estimation/Research_Design/event_study.html) library of statistical techniques.

```py
import pandas as pd
import numpy as np
from pyfixest import Fixest

# Read in data
df = pd.read_csv("https://raw.githubusercontent.com/LOST-STATS/LOST-STATS.github.io/master/Model_Estimation/Data/Event_Study_DiD/bacon_example.csv")

df['time_to_treat'] = (df['year'] - df['_nfd'] ).fillna(0).astype(int)
df['time_to_treat'] = pd.Categorical(df.time_to_treat, np.sort(df.time_to_treat.unique()))
df['treat'] = np.where(pd.isna(df['_nfd']), 0, 1)

fixest = Fixest(df)
fml = 'asmrs ~ i(time_to_treat, treat, ref = -1) + csw(pcinc, asmrh, cases) | stfips + year'
fixest.feols(fml, vcov = {'CRV1':'stfips'})
fixest.iplot()
```

![image](figures/event_study.png)