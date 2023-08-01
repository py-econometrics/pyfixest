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
# Model:  OLS
# Dep. var.:  Y
# Fixed effects:  X2+X3+X4
# Inference:  {'CRV1': 'X4'}
# Observations:  1000

 # Estimate  Std. Error  t value  Pr(>|t|)     ci_l     ci_u
 # 0.873691     0.03674 23.78022       0.0 0.801682 0.945701

```

## Quickstart

```python
import pyfixest as pf
import numpy as np
from pyfixest.utils import get_data

data = get_data()

fixest = pf.Fixest(data = data)
# OLS Estimation
fixest.feols("Y~X1 | csw0(X2, X3)", vcov = {'CRV1':'group_id'})
fixest.fetch_model(0).tidy()

# 	Estimate	Std. Error	t value	Pr(>|t|)	ci_l	ci_u
# coefnames
# Intercept	-3.941395	0.221365	-17.804976	2.442491e-15	-4.398269	-3.484520
# X1	-0.273096	0.165154	-1.653580	1.112355e-01	-0.613958	0.067766

fixest.fetch_model(1).tidy()
# 	Estimate	Std. Error	t value	Pr(>|t|)	ci_l	ci_u
# coefnames
# X1	-0.260472	0.163874	-1.589472	0.125042	-0.598691	0.077746
```

`PyFixest` also supports IV (Instrumental Variable) Estimation:

```python
fixest = pf.Fixest(data = data)
fixest.feols("Y~ 1 | X2 + X3 | X1 ~ Z1", vcov = {'CRV1':'group_id'})
fixest.fetch_model(0).tidy()

# Estimate	Std. Error	t value	Pr(>|t|)	ci_l	ci_u
# coefnames
# X1	0.041142	0.201983	0.203688	0.840315	-0.375731	0.458015
```

Standard Errors can be adjusted after estimation, "on-the-fly":

```python
# change SEs from CRV1 to HC1
fixest.vcov(vcov = "HC1").tidy()
# 		Estimate	Std. Error	t value	Pr(>|t|)	ci_l	ci_u
# fml	coefnames
# Y~X1|X2+X3	X1	0.041142	0.167284	0.245939	0.805755	-0.286927	0.36921

```