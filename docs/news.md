# News

## PyFixest 0.10.5

+ Adds an experimental `event_study()` function with support for two-way fixed effects difference-in differences
  and Gardner's two stage estimator to `pyfixest.experimental.did`.

```python
%load_ext autoreload
%autoreload 2
from pyfixest.experimental.did import event_study
from pyfixest.summarize import summary
import pandas as pd

df_het = pd.read_csv("pyfixest/data/df_het.csv")

fit_twfe = event_study(
    data = df_het,
    yname = "dep_var",
    idname= "state",
    tname = "year",
    gname = "g",
    estimator = "twfe"
)

fit_did2s = event_study(
    data = df_het,
    yname = "dep_var",
    idname= "state",
    tname = "year",
    gname = "g",
    estimator = "did2s"
)

summary([fit_twfe, fit_did2s])

# ###
#
# Estimation:  TWFE
# Dep. var.: dep_var, Fixed effects: state+year
# Inference:  CRV1
# Observations:  46500
#
# | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
# |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
# | zz00_treat    |      2.135 |        0.044 |    48.803 |      0.000 |   2.049 |    2.220 |
# ---
# ###
#
#   Estimation:  DID2S
# Dep. var.: dep_var
# Inference:  CRV1 (GMM)
# Observations:  46500
#
# | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
# |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
# | zz00_treat    |      2.152 |        0.048 |    45.208 |      0.000 |   2.059 |    2.246 |
# ---

```

## PyFixest `0.10.3`

- Allows for white space in the multiway clustering formula.
- Adds documentation for multiway clustering.

## PyFixest `0.10.2`

- Adds support for two-way clustering.
- Adds support for CRV3 inference for Poisson regression.

## PyFixest `0.10.1`

- Adapts the internal fixed effects demeaning criteron to match `PyHDFE`'s default.
- Adds Styfen as coauthor.

## PyFixest `0.10`

- Multiple performance improvements.
- Most importantly, implements a custom demeaning algorithm in `numba` - thanks to Styfen Schaer (@styfenschaer),
  which leads to performance improvements of 5x or more:

```python
%load_ext autoreload
%autoreload 2

import numpy as np
import time
import pyhdfe
from pyfixest.demean import demean

np.random.seed(1238)
N = 10_000_000
x = np.random.normal(0, 1, 10*N).reshape((N,10))
f1 = np.random.choice(list(range(1000)), N).reshape((N,1))
f2 = np.random.choice(list(range(1000)), N).reshape((N,1))

flist = np.concatenate((f1, f2), axis = 1)
weights = np.ones(N)

algorithm = pyhdfe.create(flist)

start_time = time.time()
res_pyhdfe = algorithm.residualize(x)
end_time = time.time()
print(end_time - start_time)
# 26.04527711868286


start_time = time.time()
res_pyfixest, success = demean(x, flist, weights, tol = 1e-10)
# Calculate the execution time
end_time = time.time()
print(end_time - start_time)
#4.334428071975708

np.allclose(res_pyhdfe , res_pyfixest)
# True
```



## PyFixest `0.9.11`

- Bump required `formulaic` version to `0.6.5`.
- Stop copying the data frame in `fixef()`.

## PyFixest `0.9.10`

- Fixes a big in the `wildboottest` method (see [#158](https://github.com/s3alfisc/pyfixest/issues/158)).
- Allows to run a wild bootstrap after fixed effect estimation.

## PyFixest `0.9.9`

- Adds support for `wildboottest` for Python `3.11`.

## PyFixest `0.9.8`

- Fixes a couple more bugs in the `predict()` and `fixef()` methods.
- The `predict()` argument `data` is renamed to `newdata`.

## PyFixest `0.9.7`

Fixes a bug in `predict()` produced when multicollinear variables are dropped.

## PyFixest `0.9.6`

Improved Collinearity handling. See [#145](https://github.com/s3alfisc/pyfixest/issues/145)

## PyFixest `0.9.5`


- Moves plotting from `matplotlin` to `lets-plot`.
- Fixes a few minor bugs in plotting and the `fixef()` method.


## PyFixest `0.9.1`

### Breaking API changes

It is no longer required to initiate an object of type `Fixest` prior to running `feols` or `fepois`. Instead,
you can now simply use `feols()` and `fepois()` as functions, just as in `fixest`. Both function can be found in an
`estimation` module and need to obtain a `pd.DataFrame` as a function argument:

```py
from pyfixest.estimation import fixest, fepois
from pyfixest.utils import get_data

data = get_data()
fit = feols("Y ~ X1 | f1", data = data, vcov = "iid")
```

Calling `feols()` will return an instance of class `Feols`, while calling `fepois()` will return an instance of class `Fepois`.
Multiple estimation syntax will return an instance of class `FixestMulti`.

Post processing works as before via `.summary()`, `.tidy()` and other methods.

### New Features

A summary function allows to compare multiple models:

```py
from pyfixest.summarize import summary
fit2 = feols("Y ~ X1 + X2| f1", data = data, vcov = "iid")
summary([fit, fit2])
```

Visualization is possible via custom methods (`.iplot()` & `.coefplot()`), but a new module allows to visualize
  a list of `Feols` and/or `Fepois` instances:

```py
from pyfixest.visualize import coefplot, iplot
coefplot([fit, fit2])
```

The documentation has been improved (though there is still room for progress), and the code has been cleaned up a
bit (also lots of room for improvements).