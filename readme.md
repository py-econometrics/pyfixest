## PyFixest

[![PyPI - Version](https://img.shields.io/pypi/v/pyfixest.svg)](https://pypi.org/project/pyfixest/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyfixest.svg)
![PyPI - Status](https://img.shields.io/pypi/status/pyfixest.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyfixest)

This is a draft package (highly experimental!) for a Python clone of the excellent [fixest](https://github.com/lrberge/fixest) package.

Fixed effects are projected out via the [PyHDFE](https://github.com/jeffgortmaker/pyhdfe) package.

For a quick introduction, see the [tutorial](https://s3alfisc.github.io/pyfixest/tutorial/).

```python
import pyfixest as pf
import numpy as np
from pyfixest.utils import get_data

data = get_data()

fixest = pf.Fixest(data = data)
fixest.feols("Y~X1 | csw0(X2, X3)", vcov = {'CRV1':'group_id'})
fixest.summary()
# ###
#
# ---
# ###
#
# Dep. var.:  Y
# Inference:  {'CRV1': 'group_id'}
# Observations:  998
#
#            Estimate  Std. Error   t value  Pr(>|t|)
# Intercept  6.648203    0.220649 30.130262   0.00000
#        X1 -0.141200    0.211081 -0.668937   0.50369
# ---
# ###
#
# Fixed effects:  X2
# Dep. var.:  Y
# Inference:  {'CRV1': 'group_id'}
# Observations:  998
#
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.142274    0.210556 -0.675707  0.499383
# ---
# ###
#
# Fixed effects:  X2+X3
# Dep. var.:  Y
# Inference:  {'CRV1': 'group_id'}
# Observations:  998
#
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.096317    0.204801 -0.470296  0.638247
```



