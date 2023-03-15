# Welcome to PyFixest

This is a draft package (highly experimental!) for a Python clone of the excellent [fixest](https://github.com/lrberge/fixest) package.

Fixed effects are projected out via the [PyHDFE](https://github.com/jeffgortmaker/pyhdfe) package.

```py
from pyfixest import Fixest
from pyfixest.utils import get_data

data = get_data()

fixest = Fixest(data = data)
fixest.feols("Y~X1 | X2", vcov = "HC1")
fixest.summary()
# ### Fixed-effects: X2
# Dep. var.: Y
#
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.103285    0.172956 -0.597172  0.550393
```