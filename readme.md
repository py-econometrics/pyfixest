## pyfixest

This is a draft package (highly experimental!) for a Python clone of the excellent [fixest](https://github.com/lrberge/fixest) package.

Fixed effects are projected out via the [PyHDFE](https://github.com/jeffgortmaker/pyhdfe) package.

For a quick introduction, see the [tutorial](https://s3alfisc.github.io/pyfixest/tutorial/).

```python
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

Support for more [fixest formula-sugar](https://cran.r-project.org/web/packages/fixest/vignettes/multiple_estimations.html) is work in progress.

