## PyFixest

[![PyPI - Version](https://img.shields.io/pypi/v/pyfixest.svg)](https://pypi.org/project/pyfixest/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyfixest.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyfixest)
[![image](https://codecov.io/gh/s3alfisc/pyfixest/branch/master/graph/badge.svg)](https://codecov.io/gh/s3alfisc/pyfixest)

This is a draft package (no longer highly experimental) for a Python clone of the excellent [fixest](https://github.com/lrberge/fixest) package. The package aims to mimic `fixest` syntax and functionality as closely as possible. Fixed effects are projected out via the [PyHDFE](https://github.com/jeffgortmaker/pyhdfe) package. For a quick introduction, see the [tutorial](https://s3alfisc.github.io/pyfixest/tutorial/).

## Functionality

At the moment, `PyFixest` supports

- OLS and IV Regression
- Poisson Regression
- Multiple Estimation Syntax
- Several Robust and Cluster Robust Variance-Covariance Types
- Wild Cluster Bootstrap Inference (via [wildboottest](https://github.com/s3alfisc/wildboottest))
- Support for estimators of the "new" Difference-in-Difference literature is work in progress. `PyFixest` currently provides an
  experimental implementation of Gardner's Did2s estimtator (via the `pyfixest.experimental.did` module, only ATT estimation).

## Installation

You can install the release version from `PyPi` by running `pip install pyfixest` or the development version from github.

## Quickstart

```python
from pyfixest.estimation import feols
from pyfixest.utils import get_data

data = get_data()

# OLS Estimation
fit = feols("Y~X1 | csw0(f1, f2)", data = data, vcov = {'CRV1':'group_id'})
fit.summary()

# ###
#
# Model:  OLS
# Dep. var.:  Y
# Inference:  CRV1
# Observations:  998
#
# | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
# |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
# | Intercept     |      2.206 |        0.078 |    28.304 |      0.000 |   2.043 |    2.370 |
# | X1            |      0.358 |        0.051 |     6.962 |      0.000 |   0.250 |    0.466 |
# ---
# RMSE: 1.765  Adj. R2: 0.024  Adj. R2 Within: 0.024
# ###
#
# Model:  OLS
# Dep. var.:  Y
# Fixed effects:  f1
# Inference:  CRV1
# Observations:  997
#
# | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
# |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
# | X1            |      0.411 |        0.040 |    10.188 |      0.000 |   0.326 |    0.495 |
# ---
# RMSE: 1.421  Adj. R2: 0.048  Adj. R2 Within: 0.048
# ###
#
# Model:  OLS
# Dep. var.:  Y
# Fixed effects:  f1+f2
# Inference:  CRV1
# Observations:  997
#
# | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
# |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
# | X1            |      0.431 |        0.035 |    12.319 |      0.000 |   0.358 |    0.505 |
# ---
# RMSE: 1.2  Adj. R2: 0.07  Adj. R2 Within: 0.07

```

Standard Errors can be adjusted after estimation, "on-the-fly":

```python
fit1 = fit.fetch_model(0)
fit1.vcov("hetero").tidy()
# Model:  Y~X1
# ###
#
# Model:  OLS
# Dep. var.:  Y
# Inference:  hetero
# Observations:  998
#
# | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
# |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
# | Intercept     |      2.206 |        0.088 |    25.180 |      0.000 |   2.034 |    2.378 |
# | X1            |      0.358 |        0.068 |     5.254 |      0.000 |   0.224 |    0.491 |
# ---
# RMSE: 1.765  Adj. R2: 0.024  Adj. R2 Within: 0.024
```

Last, `PyFixest` also supports IV estimation via three part formula syntax:

```py
fit_iv = feols("Y ~ 1 | f1 | X1 ~ Z1", data = data)
fit_iv.summary()

# ###
#
# Model:  IV
# Dep. var.:  Y
# Fixed effects:  f1
# Inference:  CRV1
# Observations:  997
#
# | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
# |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
# | X1            |      0.479 |        0.096 |     4.979 |      0.000 |   0.282 |    0.676 |
# ---
```