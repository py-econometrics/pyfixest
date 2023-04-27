## PyFixest


This is a draft package (highly experimental!) for a Python clone of the excellent [fixest](https://github.com/lrberge/fixest) package.

Fixed effects are projected out via the [PyHDFE](https://github.com/jeffgortmaker/pyhdfe) package.

For a quick introduction, see the [tutorial](https://s3alfisc.github.io/pyfixest/tutorial/).

```python
import pyfixest as pf
from pyfixest.utils import get_data

data = get_data()

fixest = pf.Fixest(data = data)
fixest.feols("Y~X1 | X2", vcov = "HC1")
fixest.summary()

# ###
#
# model: feols()
# fml: Y~X1 | X2
# ---
# ###
#
# Fixed effects:  X2
# Dep. var.:  Y
# Inference:  HC1
# Observations:  998
#
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.142274    0.210556 -0.675707  0.499383
```

