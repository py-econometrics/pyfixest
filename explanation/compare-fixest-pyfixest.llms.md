# Does `PyFixest` match `fixest`?

This vignette compares estimation results from `fixest` with `pyfixest` via the `rpy2` package.

## Intentional formula differences

### Covariate powers

For covariates, PyFixest prioritizes [Formulaic’s formula conventions](https://matthew.wardrop.casa/formulaic/latest/guides/grammar/) over `fixest`’s arithmetic-power convention. Outside a Python expression such as `I(...)`, both `^` and `**` expand interactions up to the specified order:

| Covariate formula  | PyFixest interpretation            |
|--------------------|------------------------------------|
| `Y ~ X1 + X1^2`    | `Y ~ X1`; no squared term is added |
| `Y ~ X1 + X1**2`   | `Y ~ X1`; no squared term is added |
| `Y ~ (X1 + X2)^2`  | `Y ~ X1 + X2 + X1:X2`              |
| `Y ~ (X1 + X2)**2` | `Y ~ X1 + X2 + X1:X2`              |

In particular, a single-factor power such as `X1^2` reduces to `X1` under these formula rules. It does **not** square the values of `X1`, and adding it alongside `X1` has no effect. This also applies when the model has fixed effects. In contrast, `fixest` 0.14.0 interprets covariate `X1^2` as the arithmetic transformation `I(X1^2)`, and `(X1 + X2)^2` as `I((X1 + X2)^2)`. This is an intentional syntax difference; see [\#1765](https://github.com/py-econometrics/pyfixest/issues/1765).

To include an arithmetic square in PyFixest, use `I(X1**2)`:

``` python
import pyfixest as pf

data = pf.get_data()
fit_squared = pf.feols("Y ~ X1 + I(X1**2)", data=data)
fit_squared_fe = pf.feols("Y ~ X1 + I(X1**2) | f1", data=data)
```

Inside `I(...)`, expressions follow Python arithmetic: `**` means exponentiation, while `^` means bitwise XOR. Use `I(X1**2)` when translating R’s `I(X1^2)`.

## Setup

``` python
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import Converter
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import ListVector
from rpy2.rinterface import ListSexpVector

import pyfixest as pf

# Import R packages
fixest = importr("fixest")
stats = importr("stats")
broom = importr("broom")

# Enable automatic pandas <-> R DataFrame conversion (rpy2 >= 3.6)
# Keep R list classes intact (avoid conversion to NamedList).
_list_guard = Converter("pandas2ri-list-guard")

@_list_guard.rpy2py.register(ListSexpVector)
def _keep_r_list_classes(obj):
    return ListVector(obj)

_converter_ctx = (
    ro.default_converter + numpy2ri.converter + pandas2ri.converter + _list_guard
).context()
_converter_ctx.__enter__()

# IPython magic commands for autoreloading
%load_ext autoreload
%autoreload 2

# Get data using pyfixest
data = pf.get_data(model="Feols", N=10_000, seed=99292)
```

## Ordinary Least Squares (OLS)

### IID Inference

First, we estimate a model via \`pyfixest. We compute “iid” standard errors.

``` python
fit = pf.feols(fml="Y ~ X1 + X2 | f1 + f2", data=data, vcov="iid")
```

We estimate the same model with weights:

``` python
fit_weights = pf.feols(
    fml="Y ~ X1 + X2 | f1 + f2", data=data, weights="weights", vcov="iid"
)
```

Via `r-fixest` and `rpy2`, we get

``` python
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

Let’s compare how close the covariance matrices are:

``` python
fit_vcov = fit.variance_covariance.vcov
r_vcov = stats.vcov(r_fit)
fit_vcov - r_vcov
```

    array([[-5.96311195e-19,  7.89957485e-23],
           [ 7.85821582e-23, -1.32137140e-19]])

And for WLS:

``` python
fit_weights.variance_covariance.vcov - stats.vcov(r_fit_weights)
```

    array([[ 1.73472348e-18,  3.07049443e-21],
           [ 3.07049443e-21, -1.35525272e-19]])

We conclude by comparing all estimation results via the `tidy` methods:

``` python
fit.tidy()
```

|             | Estimate | Std. Error | t value    | Pr(\>\|t\|)  | 2.5%     | 97.5%    |
|-------------|----------|------------|------------|--------------|----------|----------|
| Coefficient |          |            |            |              |          |          |
| X1          | 0.112019 | 0.017042   | 6.572948   | 5.181855e-11 | 0.078612 | 0.145425 |
| X2          | 0.732788 | 0.004621   | 158.578261 | 0.000000e+00 | 0.723730 | 0.741846 |

``` python
pd.DataFrame(broom.tidy_fixest(r_fit)).T
```

|     | 0   | 1        | 2        | 3          | 4   |
|-----|-----|----------|----------|------------|-----|
| 0   | X1  | 0.112019 | 0.017042 | 6.572948   | 0.0 |
| 1   | X2  | 0.732788 | 0.004621 | 158.578261 | 0.0 |

``` python
fit_weights.tidy()
```

|             | Estimate | Std. Error | t value    | Pr(\>\|t\|)  | 2.5%     | 97.5%    |
|-------------|----------|------------|------------|--------------|----------|----------|
| Coefficient |          |            |            |              |          |          |
| X1          | 0.123687 | 0.016975   | 7.286200   | 3.432810e-13 | 0.090411 | 0.156962 |
| X2          | 0.732244 | 0.004610   | 158.844322 | 0.000000e+00 | 0.723207 | 0.741280 |

``` python
pd.DataFrame(broom.tidy_fixest(r_fit_weights)).T
```

|     | 0   | 1        | 2        | 3          | 4   |
|-----|-----|----------|----------|------------|-----|
| 0   | X1  | 0.123687 | 0.016975 | 7.2862     | 0.0 |
| 1   | X2  | 0.732244 | 0.00461  | 158.844322 | 0.0 |

### Heteroskedastic Errors

We repeat the same exercise with heteroskedastic (HC1) errors:

``` python
fit = pf.feols(fml="Y ~ X1 + X2 | f1 + f2", data=data, vcov="hetero")
fit_weights = pf.feols(
    fml="Y ~ X1 + X2 | f1 + f2", data=data, vcov="hetero", weights="weights"
)
```

``` python
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

``` python
fit.variance_covariance.vcov - stats.vcov(r_fit)
```

    array([[ 1.84896586e-14,  4.01031841e-15],
           [ 4.01031841e-15, -4.47538328e-17]])

``` python
fit_weights.variance_covariance.vcov - stats.vcov(r_fit_weights)
```

    array([[ 2.45021288e-13, -8.42544020e-14],
           [-8.42544020e-14,  6.30967378e-15]])

We conclude by comparing all estimation results via the `tidy` methods:

``` python
fit.tidy()
```

|             | Estimate | Std. Error | t value    | Pr(\>\|t\|)  | 2.5%     | 97.5%    |
|-------------|----------|------------|------------|--------------|----------|----------|
| Coefficient |          |            |            |              |          |          |
| X1          | 0.112019 | 0.017105   | 6.548962   | 6.082002e-11 | 0.078490 | 0.145548 |
| X2          | 0.732788 | 0.004579   | 160.036098 | 0.000000e+00 | 0.723812 | 0.741763 |

``` python
pd.DataFrame(broom.tidy_fixest(r_fit)).T
```

|     | 0   | 1        | 2        | 3          | 4   |
|-----|-----|----------|----------|------------|-----|
| 0   | X1  | 0.112019 | 0.017105 | 6.548962   | 0.0 |
| 1   | X2  | 0.732788 | 0.004579 | 160.036098 | 0.0 |

``` python
fit_weights.tidy()
```

|             | Estimate | Std. Error | t value    | Pr(\>\|t\|)  | 2.5%     | 97.5%    |
|-------------|----------|------------|------------|--------------|----------|----------|
| Coefficient |          |            |            |              |          |          |
| X1          | 0.123687 | 0.019470   | 6.352618   | 2.210043e-10 | 0.085521 | 0.161852 |
| X2          | 0.732244 | 0.005169   | 141.653304 | 0.000000e+00 | 0.722111 | 0.742376 |

``` python
pd.DataFrame(broom.tidy_fixest(r_fit_weights)).T
```

|     | 0   | 1        | 2        | 3          | 4   |
|-----|-----|----------|----------|------------|-----|
| 0   | X1  | 0.123687 | 0.01947  | 6.352618   | 0.0 |
| 1   | X2  | 0.732244 | 0.005169 | 141.653304 | 0.0 |

### Cluster-Robust Errors

We conclude with cluster robust errors.

``` python
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

``` python
fit.variance_covariance.vcov - stats.vcov(r_fit)
```

    array([[ 1.33663100e-13,  5.16681678e-14],
           [ 5.16681678e-14, -5.09801552e-14]])

``` python
fit_weights.variance_covariance.vcov - stats.vcov(r_fit_weights)
```

    array([[2.75766177e-12, 9.88699462e-13],
           [9.88699462e-13, 9.33791754e-14]])

We conclude by comparing all estimation results via the `tidy` methods:

``` python
fit.tidy()
```

|             | Estimate | Std. Error | t value    | Pr(\>\|t\|)  | 2.5%     | 97.5%    |
|-------------|----------|------------|------------|--------------|----------|----------|
| Coefficient |          |            |            |              |          |          |
| X1          | 0.112019 | 0.015865   | 7.060624   | 4.823750e-09 | 0.080152 | 0.143885 |
| X2          | 0.732788 | 0.004490   | 163.215618 | 0.000000e+00 | 0.723770 | 0.741806 |

``` python
pd.DataFrame(broom.tidy_fixest(r_fit)).T
```

|     | 0   | 1        | 2        | 3          | 4   |
|-----|-----|----------|----------|------------|-----|
| 0   | X1  | 0.112019 | 0.015865 | 7.060624   | 0.0 |
| 1   | X2  | 0.732788 | 0.00449  | 163.215618 | 0.0 |

``` python
fit_weights.tidy()
```

|             | Estimate | Std. Error | t value    | Pr(\>\|t\|)  | 2.5%     | 97.5%    |
|-------------|----------|------------|------------|--------------|----------|----------|
| Coefficient |          |            |            |              |          |          |
| X1          | 0.123687 | 0.018368   | 6.733633   | 1.566958e-08 | 0.086792 | 0.160581 |
| X2          | 0.732244 | 0.005266   | 139.062210 | 0.000000e+00 | 0.721667 | 0.742820 |

``` python
pd.DataFrame(broom.tidy_fixest(r_fit_weights)).T
```

|     | 0   | 1        | 2        | 3         | 4   |
|-----|-----|----------|----------|-----------|-----|
| 0   | X1  | 0.123687 | 0.018368 | 6.733633  | 0.0 |
| 1   | X2  | 0.732244 | 0.005266 | 139.06221 | 0.0 |

## Poisson Regression

``` python
data = pf.get_data(model="Fepois")
```

``` python
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

``` python
fit_iid.variance_covariance.vcov - stats.vcov(fit_r_iid)
```

    array([[ 1.28417953e-08, -6.96933242e-10],
           [-6.96933242e-10,  1.80681045e-09]])

``` python
fit_hetero.variance_covariance.vcov - stats.vcov(fit_r_hetero)
```

    array([[ 2.31877603e-08, -7.85182460e-10],
           [-7.85182460e-10,  3.26995553e-09]])

``` python
fit_crv.variance_covariance.vcov - stats.vcov(fit_r_crv)
```

    array([[ 1.63388420e-08, -1.25332336e-10],
           [-1.25332336e-10,  3.27770983e-09]])

We conclude by comparing all estimation results via the `tidy` methods:

``` python
fit_iid.tidy()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|) | 2.5%      | 97.5%    |
|-------------|-----------|------------|-----------|-------------|-----------|----------|
| Coefficient |           |            |           |             |           |          |
| X1          | -0.006591 | 0.042025   | -0.156836 | 0.875374    | -0.088959 | 0.075777 |
| X2          | -0.014924 | 0.011336   | -1.316520 | 0.188000    | -0.037142 | 0.007294 |

``` python
pd.DataFrame(broom.tidy_fixest(fit_r_iid)).T
```

|     | 0   | 1         | 2        | 3         | 4        |
|-----|-----|-----------|----------|-----------|----------|
| 0   | X1  | -0.006591 | 0.042025 | -0.156836 | 0.875374 |
| 1   | X2  | -0.014924 | 0.011336 | -1.316529 | 0.187996 |

``` python
fit_hetero.tidy()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|) | 2.5%      | 97.5%    |
|-------------|-----------|------------|-----------|-------------|-----------|----------|
| Coefficient |           |            |           |             |           |          |
| X1          | -0.006591 | 0.040363   | -0.163297 | 0.870284    | -0.085700 | 0.072518 |
| X2          | -0.014924 | 0.010828   | -1.378277 | 0.168118    | -0.036146 | 0.006298 |

``` python
pd.DataFrame(broom.tidy_fixest(fit_r_hetero)).T
```

|     | 0   | 1         | 2        | 3         | 4        |
|-----|-----|-----------|----------|-----------|----------|
| 0   | X1  | -0.006591 | 0.040362 | -0.163298 | 0.870284 |
| 1   | X2  | -0.014924 | 0.010828 | -1.378296 | 0.168112 |

``` python
fit_crv.tidy()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|) | 2.5%      | 97.5%    |
|-------------|-----------|------------|-----------|-------------|-----------|----------|
| Coefficient |           |            |           |             |           |          |
| X1          | -0.006591 | 0.035302   | -0.186705 | 0.851892    | -0.075782 | 0.062600 |
| X2          | -0.014924 | 0.010468   | -1.425726 | 0.153947    | -0.035440 | 0.005592 |

``` python
pd.DataFrame(broom.tidy_fixest(fit_r_crv)).T
```

|     | 0   | 1         | 2        | 3         | 4        |
|-----|-----|-----------|----------|-----------|----------|
| 0   | X1  | -0.006591 | 0.035302 | -0.186706 | 0.851891 |
| 1   | X2  | -0.014924 | 0.010467 | -1.425747 | 0.153941 |
