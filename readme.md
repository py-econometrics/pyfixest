## PyFixest

[![PyPI - Version](https://img.shields.io/pypi/v/pyfixest.svg)](https://pypi.org/project/pyfixest/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyfixest.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyfixest)
[![image](https://codecov.io/gh/s3alfisc/pyfixest/branch/master/graph/badge.svg)](https://codecov.io/gh/s3alfisc/pyfixest)

This is a draft package (highly experimental!) for a Python clone of the excellent [fixest](https://github.com/lrberge/fixest) package.

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
fixest.summary()
# ###
#
# Model:  OLS
# Dep. var.:  Y
# Inference:  {'CRV1': 'group_id'}
# Observations:  1998
#
#            Estimate  Std. Error    t value     Pr(>|t|)
# Intercept -3.941395    0.221365 -17.804976 2.442491e-15
#        X1 -0.273096    0.165154  -1.653580 1.112355e-01
# ---
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
# ###
#
# Model:  OLS
# Dep. var.:  Y
# Fixed effects:  X2+X3
# Inference:  {'CRV1': 'group_id'}
# Observations:  1998
#
#     Estimate  Std. Error  t value  Pr(>|t|)
# X1   0.03975    0.107003 0.371481  0.713538
# ---


# IV Estimation
fixest = pf.Fixest(data = data)
fixest.feols("Y~X1 | csw0(X2, X3) | X1 ~ Z1", vcov = {'CRV1':'group_id'})
fixest.summary()
# ###
#
# Model:  IV
# Dep. var.:  Y
# Inference:  {'CRV1': 'group_id'}
# Observations:  1998
#
#            Estimate  Std. Error    t value     Pr(>|t|)
# Intercept -3.941293    0.221354 -17.805377 2.442491e-15
#        X1 -0.265817    0.261940  -1.014803 3.203217e-01
# ---
# ###
#
# Model:  IV
# Dep. var.:  Y
# Fixed effects:  X2
# Inference:  {'CRV1': 'group_id'}
# Observations:  1998
#
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.259964    0.264817 -0.981674  0.336054
# ---
# ###
#
# Model:  IV
# Dep. var.:  Y
# Fixed effects:  X2+X3
# Inference:  {'CRV1': 'group_id'}
# Observations:  1998
#
#     Estimate  Std. Error  t value  Pr(>|t|)
# X1  0.041142    0.201983 0.203688  0.840315
# ---
```



