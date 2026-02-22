This vignette compares estimation results from `fixest` with `pyfixest` via the `rpy2` package.

## Setup


```{python}
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

import pyfixest as pf

# Activate pandas2ri
pandas2ri.activate()

# Import R packages
fixest = importr("fixest")
stats = importr("stats")
broom = importr("broom")

# IPython magic commands for autoreloading
%load_ext autoreload
%autoreload 2

# Get data using pyfixest
data = pf.get_data(model="Feols", N=10_000, seed=99292)
```


## Ordinary Least Squares (OLS)

### IID Inference

First, we estimate a model via `pyfixest. We compute "iid" standard errors.


```{python}
fit = pf.feols(fml="Y ~ X1 + X2 | f1 + f2", data=data, vcov="iid")
```

We estimate the same model with weights:


```{python}
fit_weights = pf.feols(
    fml="Y ~ X1 + X2 | f1 + f2", data=data, weights="weights", vcov="iid"
)
```

Via `r-fixest` and `rpy2`, we get


```{python}
r_fit = fixest.feols(
    ro.Formula("Y ~ X1 + X2 | f1 + f2"),
    data=data,
    vcov="iid",
)

r_fit_weights = fixest.feols(
    ro.Formula("Y ~ X1 + X2 | f1 + f2"),
    data=data,
    weights=ro.Formula("~weights"),
    vcov="iid",
)
```

    R[write to console]: NOTE: 3 observations removed because of NA values (LHS: 1, RHS: 1, Fixed-effects: 1).

    R[write to console]: NOTE: 3 observations removed because of NA values (LHS: 1, RHS: 1, Fixed-effects: 1).



Let's compare how close the covariance matrices are:


```{python}
fit_vcov = fit._vcov
r_vcov = stats.vcov(r_fit)
fit_vcov - r_vcov
```


And for WLS:


```{python}
fit_weights._vcov - stats.vcov(r_fit_weights)
```

We conclude by comparing all estimation results via the `tidy` methods:

```{python}
fit.tidy()
```

```{python}
pd.DataFrame(broom.tidy_fixest(r_fit)).T
```

```{python}
fit_weights.tidy()
```

```{python}
pd.DataFrame(broom.tidy_fixest(r_fit_weights)).T
```


### Heteroskedastic Errors

We repeat the same exercise with heteroskedastic (HC1) errors:


```{python}
fit = pf.feols(fml="Y ~ X1 + X2 | f1 + f2", data=data, vcov="hetero")
fit_weights = pf.feols(
    fml="Y ~ X1 + X2 | f1 + f2", data=data, vcov="hetero", weights="weights"
)
```


```{python}
r_fit = fixest.feols(
    ro.Formula("Y ~ X1 + X2 | f1 + f2"),
    data=data,
    vcov="hetero",
)

r_fit_weights = fixest.feols(
    ro.Formula("Y ~ X1 + X2 | f1 + f2"),
    data=data,
    weights=ro.Formula("~weights"),
    vcov="hetero",
)
```

As before, we compare the variance covariance matrices:


```{python}
fit._vcov - stats.vcov(r_fit)
```

```{python}
fit_weights._vcov - stats.vcov(r_fit_weights)
```

We conclude by comparing all estimation results via the `tidy` methods:

```{python}
fit.tidy()
```

```{python}
pd.DataFrame(broom.tidy_fixest(r_fit)).T
```

```{python}
fit_weights.tidy()
```

```{python}
pd.DataFrame(broom.tidy_fixest(r_fit_weights)).T
```


### Cluster-Robust Errors

We conclude with cluster robust errors.


```{python}
fit = pf.feols(fml="Y ~ X1 + X2 | f1 + f2", data=data, vcov={"CRV1": "f1"})
fit_weights = pf.feols(
    fml="Y ~ X1 + X2 | f1 + f2", data=data, vcov={"CRV1": "f1"}, weights="weights"
)

r_fit = fixest.feols(
    ro.Formula("Y ~ X1 + X2 | f1 + f2"),
    data=data,
    vcov=ro.Formula("~f1"),
)
r_fit_weights = fixest.feols(
    ro.Formula("Y ~ X1 + X2 | f1 + f2"),
    data=data,
    weights=ro.Formula("~weights"),
    vcov=ro.Formula("~f1"),
)
```

```{python}
fit._vcov - stats.vcov(r_fit)
```

```{python}
fit_weights._vcov - stats.vcov(r_fit_weights)
```

We conclude by comparing all estimation results via the `tidy` methods:

```{python}
fit.tidy()
```

```{python}
pd.DataFrame(broom.tidy_fixest(r_fit)).T
```

```{python}
fit_weights.tidy()
```

```{python}
pd.DataFrame(broom.tidy_fixest(r_fit_weights)).T
```

## Poisson Regression


```{python}
data = pf.get_data(model="Fepois")
```


```{python}
fit_iid = pf.fepois(fml="Y ~ X1 + X2 | f1 + f2", data=data, vcov="iid", iwls_tol=1e-10)
fit_hetero = pf.fepois(
    fml="Y ~ X1 + X2 | f1 + f2", data=data, vcov="hetero", iwls_tol=1e-10
)
fit_crv = pf.fepois(
    fml="Y ~ X1 + X2 | f1 + f2", data=data, vcov={"CRV1": "f1"}, iwls_tol=1e-10
)

fit_r_iid = fixest.fepois(
    ro.Formula("Y ~ X1 + X2 | f1 + f2"),
    data=data,
    vcov="iid",
)

fit_r_hetero = fixest.fepois(
    ro.Formula("Y ~ X1 + X2 | f1 + f2"),
    data=data,
    vcov="hetero",
)

fit_r_crv = fixest.fepois(
    ro.Formula("Y ~ X1 + X2 | f1 + f2"),
    data=data,
    vcov=ro.Formula("~f1"),
)
```

```{python}
fit_iid._vcov - stats.vcov(fit_r_iid)
```

```{python}
fit_hetero._vcov - stats.vcov(fit_r_hetero)
```

```{python}
fit_crv._vcov - stats.vcov(fit_r_crv)
```

We conclude by comparing all estimation results via the `tidy` methods:


```{python}
fit_iid.tidy()
```

```{python}
pd.DataFrame(broom.tidy_fixest(fit_r_iid)).T
```

```{python}
fit_hetero.tidy()
```

```{python}
pd.DataFrame(broom.tidy_fixest(fit_r_hetero)).T
```

```{python}
fit_crv.tidy()
```

```{python}
pd.DataFrame(broom.tidy_fixest(fit_r_crv)).T
```
