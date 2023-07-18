## PyFixest

[![PyPI - Version](https://img.shields.io/pypi/v/pyfixest.svg)](https://pypi.org/project/pyfixest/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyfixest.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyfixest)
[![image](https://codecov.io/gh/s3alfisc/pyfixest/branch/master/graph/badge.svg)](https://codecov.io/gh/s3alfisc/pyfixest)

This is a draft package (no longer highly experimental) for a Python clone of the excellent [fixest](https://github.com/lrberge/fixest) package.

The package aims to mimic `fixest` syntax and functionality as closely as possible.

Fixed effects are projected out via the [PyHDFE](https://github.com/jeffgortmaker/pyhdfe) package.

For a quick introduction, see the [tutorial](https://s3alfisc.github.io/pyfixest/tutorial/).

```python
import pyfixest as pf
import numpy as np
from pyfixest.utils import get_data

data = get_data()

fixest = pf.Fixest(data = data)
# OLS Estimation
fixest.feols("Y~X1 | csw0(X2, X3)", vcov = {'CRV1':'group_id'})
fixest.fetch_model(0).tidy()

#           Estimate  Std. Error  ...  confint_lower  confint_upper
#coefnames                        ...
#Intercept -3.941395    0.221365  ...       3.955276      -3.927514
#X1        -0.273096    0.165154  ...       0.283452      -0.262740

fixest.fetch_model(1).tidy()
#           Estimate  Std. Error  ...  confint_lower  confint_upper
#coefnames                        ...
#X1        -0.260472    0.163874  ...       0.270748      -0.250196
```

`PyFixest` also supports IV (Instrumental Variable) Estimation:

```python
fixest = pf.Fixest(data = data)
fixest.feols("Y~ 1 | X2 + X3 | X1 ~ Z1", vcov = {'CRV1':'group_id'})
fixest.fetch_model(0).tidy()
# Model:  Y~X1|X2+X3
#           Estimate  Std. Error  ...  confint_lower  confint_upper
# coefnames                        ...
# X1         0.041142    0.201983  ...      -0.028476       0.053807
```

Standard Errors can be adjusted after estimation, "on-the-fly":

```python
# change SEs from CRV1 to HC1
fixest.vcov(vcov = "HC1").tidy()
#                       Estimate  Std. Error  ...  confint_lower  confint_upper
# fml        coefnames                        ...
# Y~X1|X2+X3 X1         0.041142    0.167284  ...      -0.030652       0.051631

```