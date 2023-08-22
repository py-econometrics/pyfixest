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

## Installation

You can install the release version from `PyPi` by running `pip install pyfixest` or the development version from github.

## News

`PyFixest` now supports Poisson regression!

```python
import pyfixest as pf
from pyfixest.utils import get_poisson_data

pdata = get_poisson_data()
fixest = pf.Fixest(data = pdata)
fixest.fepois("Y~X1 | X2+X3+X4", vcov = {'CRV1':'X4'})

fixest.summary()
# Model:  Poisson
# Dep. var.:  Y
# Fixed effects:  X2+X3+X4
# Inference:  {'CRV1': 'X4'}
# Observations:  1000

# | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
# |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
# | X1            |      0.874 |        0.037 |    23.780 |      0.000 |   0.802 |    0.946 |
# ---
# Deviance: 481157.824
```


## Quickstart

```python
import pyfixest as pf
import numpy as np
from pyfixest.utils import get_data

data = get_data()

fixest = pf.Fixest(data = data)
# OLS Estimation
fixest.feols("Y~X1 | csw0(f1, f2)", vcov = {'CRV1':'group_id'})
fixest.summary()
# ###
#
# Model:  OLS
# Dep. var.:  Y
# Inference:  {'CRV1': 'group_id'}
# Observations:  998
#
# | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
# |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
# | Intercept     |      2.204 |        0.054 |    40.495 |      0.000 |   2.096 |    2.312 |
# | X1            |      0.351 |        0.063 |     5.595 |      0.000 |   0.227 |    0.476 |
# ---
# RMSE: 1.751  Adj. R2: 0.037  Adj. R2 Within: 0.037
# ###
#
# Model:  OLS
# Dep. var.:  Y
# Fixed effects:  f1
# Inference:  {'CRV1': 'group_id'}
# Observations:  997
#
# | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
# |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
# | X1            |      0.326 |        0.048 |     6.756 |      0.000 |   0.230 |    0.422 |
# ---
# RMSE: 1.407  Adj. R2: 0.049  Adj. R2 Within: 0.049
# ###
#
# Model:  OLS
# Dep. var.:  Y
# Fixed effects:  f1+f2
# Inference:  {'CRV1': 'group_id'}
# Observations:  997
#
# | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
# |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
# | X1            |      0.355 |        0.039 |     9.044 |      0.000 |   0.277 |    0.433 |
# ---
# RMSE: 1.183  Adj. R2: 0.078  Adj. R2 Within: 0.078
```

`PyFixest` also supports IV (Instrumental Variable) Estimation:

```python
fixest = pf.Fixest(data = data)
fixest.feols("Y~ 1 | f2 + f3 | X1 ~ Z1", vcov = {'CRV1':'group_id'})
fixest.summary()
# ###
#
# Model:  IV
# Dep. var.:  Y
# Fixed effects:  f2+f3
# Inference:  {'CRV1': 'group_id'}
# Observations:  998
#
# | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
# |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
# | X1            |      0.309 |        0.058 |     5.306 |      0.000 |   0.193 |    0.424 |
# ---
```

Standard Errors can be adjusted after estimation, "on-the-fly":

```python
fixest.vcov("hetero").tidy()
# ###
#
# Model:  IV
# Dep. var.:  Y
# Fixed effects:  f2+f3
# Inference:  hetero
# Observations:  998
#
# | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
# |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
# | X1            |      0.309 |        0.063 |     4.877 |      0.000 |   0.184 |    0.433 |
```