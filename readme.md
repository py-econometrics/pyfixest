## PyFixest

[![PyPI - Version](https://img.shields.io/pypi/v/pyfixest.svg)](https://pypi.org/project/pyfixest/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyfixest.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyfixest)
[![image](https://codecov.io/gh/s3alfisc/pyfixest/branch/master/graph/badge.svg)](https://codecov.io/gh/s3alfisc/pyfixest)

This is a draft package (no longer highly experimental) for a Python clone of the excellent [fixest](https://github.com/lrberge/fixest) package.

The package aims to mimic `fixest` syntax and functionality as closely as possible.

Fixed effects are projected out via the [PyHDFE](https://github.com/jeffgortmaker/pyhdfe) package.

For a quick introduction, see the [tutorial](https://s3alfisc.github.io/pyfixest/tutorial/).

You can install the release version from `PyPi` by running `pip install pyfixest` (v0.7.3).

## News

The dev version of `PyFixest` (v0.8.0) now supports Poisson regression!

```python
import pyfixest as pf
from pyfixest.utils import get_poisson_data

pdata = get_poisson_data()
fixest = pf.Fixest(data = pdata)
fixest.fepois("Y~X1 | X2+X3+X4", vcov = {'CRV1':'X4'})

fixest.summary()

# ###
#
# Model:  Poisson
# Dep. var.:  Y
# Fixed effects:  X2+X3+X4
# Inference:  {'CRV1': 'X4'}
# Observations:  1000
#
#  Estimate  Std. Error  t value  Pr(>|t|)  ci_l  ci_u
#      0.87        0.04    23.78       0.0   0.8  0.95
# ---
```

Note that `v0.8.0` is not yet on `PyPi` - I first need to finalize a PR to [PyHDFE](https://github.com/jeffgortmaker/pyhdfe/pull/4) to support weights - which
is a requirement for Poisson regression via iterated weighted least squares. The github repo currently installs my PR fork of [PyHDFE](https://github.com/s3alfisc/pyhdfe).

## Quickstart

```python
import pyfixest as pf
import numpy as np
from pyfixest.utils import get_data

data = get_data()

fixest = pf.Fixest(data = data)
# OLS Estimation
fixest.feols("Y~X1 | csw0(X2, X3)", vcov = {'CRV1':'group_id'})
fixest.summary()

# ###
# #
# Model:  OLS
# Dep. var.:  Y
# Inference:  {'CRV1': 'group_id'}
# Observations:  1998
#
#  Estimate  Std. Error  t value  Pr(>|t|)  ci_l  ci_u
#     -3.94        0.22   -17.80      0.00 -4.40 -3.48
#     -0.27        0.17    -1.65      0.11 -0.61  0.07
# ---
# RMSE: 8.26  Adj. R2: 0.0  Adj. R2 Within: 0.0
# ###
#
# Model:  OLS
# Dep. var.:  Y
# Fixed effects:  X2
# Inference:  {'CRV1': 'group_id'}
# Observations:  1998
#
#  Estimate  Std. Error  t value  Pr(>|t|)  ci_l  ci_u
#     -0.26        0.16    -1.59      0.13  -0.6  0.08
# ---
# RMSE: 8.25  Adj. R2: 0.0  Adj. R2 Within: 0.0
# ###
# ...
#  Estimate  Std. Error  t value  Pr(>|t|)  ci_l  ci_u
#      0.04        0.11     0.37      0.71 -0.18  0.26
# ---
# RMSE: 5.5  Adj. R2: -0.0  Adj. R2 Within: -0.0
```

`PyFixest` also supports IV (Instrumental Variable) Estimation:

```python
fixest = pf.Fixest(data = data)
fixest.feols("Y~ 1 | X2 + X3 | X1 ~ Z1", vcov = {'CRV1':'group_id'})
fixest.summary()

# ###
#
# Model:  IV
# Dep. var.:  Y
# Fixed effects:  X2+X3
# Inference:  {'CRV1': 'group_id'}
# Observations:  1998
#
#  Estimate  Std. Error  t value  Pr(>|t|)  ci_l  ci_u
#      0.04         0.2      0.2      0.84 -0.38  0.46
```

Standard Errors can be adjusted after estimation, "on-the-fly":

```python
# 		            Estimate	Std. Error	t value	    Pr(>|t|)	   ci_l	     ci_u
# fml	coefnames
# Y~X1|X2+X3	X1	0.041142	0.167284	0.245939	0.805755	-0.286927	0.36921
```