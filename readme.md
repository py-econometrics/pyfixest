## pyfixest

This is a draft package (highly experimental!) for a Python clone of the excellent [fixest](https://github.com/lrberge/fixest) package.

Fixed effects are projected out via the [PyHDFE](https://github.com/jeffgortmaker/pyhdfe) package.

```python
# create some data
import pandas as pd
import numpy as np
from pyfixest.api import feols
from pyfixest.utils import get_data

# create data
np.random.seed(123)
data = get_data()

feols('Y ~ X1 | X2 + X3 + X4', 'hetero', data)
#   colnames      coef        se    tstat    pvalue
# 0       X1  0.001469  0.003159  0.46505  0.641896
feols('Y ~ X1 + X2 + X3 + X4', 'iid', data)
#   colnames      coef        se    tstat    pvalue
# 2048         X1    0.001469  0.003159     0.465050  6.418959e-01
sm.ols('Y ~ X1 + X2 + X3 + X4', data).fit().summary()
#   colnames      coef        se    tstat    pvalue
# X1            0.0015      0.003      0.460      0.645

# cluster robust inference:
feols(fml = 'Y ~ X1', vcov = {'CRV1':'group_id'}, data = data)
#     colnames        coef        se       tstat    pvalue
# 0  Intercept -577.090042  1.072007 -538.326702  0.000000
# 1         X1    1.389563  1.002708    1.385810  0.165805
feols(fml = 'Y ~ X1', vcov = {'CRV3':'group_id'}, data = data)
#     colnames        coef        se       tstat    pvalue
# 0  Intercept -577.090042  1.139483 -506.449086  0.000000
# 1         X1    1.389563  1.066219    1.303261  0.192486
```

## Multiple Estimations

Currently supported: multiple dependent variables, `sw()`, `sw0()`, `csw()` and `csw0()`.

```python
# sw
# multiple estimation: multiple dependent variables
data.X2 = data.X2.astype(float)
feols(fml = 'Y + Y2 ~ sw(X1, X2) | csw0(X3, X4)', vcov = {'CRV1':'group_id'}, data = data)
# fml	       coefnames	coef	se	tstat	pvalue
# 0	Y~X1|0	   Intercept	7.386158	0.162152	45.550697	0.000000
# 1	Y~X1|0	         X1	-0.163744	0.166210	-0.985159	0.324546
# 0	Y~X2|0	   Intercept	20.069156	0.300012	66.894580	0.000000
# 1	Y~X2|0	         X2	-0.516267	0.009976	-51.751797	0.000000
# 0	Y2~X1|0	   Intercept	7.375426	0.162093	45.501283	0.000000
# 1	Y2~X1|0	         X1	-0.179765	0.164399	-1.093465	0.274190
# 0	Y2~X2|0	   Intercept	20.065154	0.302652	66.297875	0.000000
# 1	Y2~X2|0	         X2	-0.516542	0.010137	-50.956222	0.000000
# 0	Y~X1|X3          X1	-0.117885	0.154998	-0.760557	0.446922
# 0	Y~X2|X3	         X2	-0.515981	0.009493	-54.354837	0.000000
# 0	Y2~X1|X3	       X1	-0.134384	0.153802	-0.873746	0.382257
# 0	Y2~X2|X3	       X2	-0.516327	0.009654	-53.485357	0.000000
# 0	Y~X1|X3+X4	     X1	-0.063646	0.082916	-0.767600	0.442725
# 0	Y~X2|X3+X4	     X2	-0.509159	0.000583	-872.642351	0.000000
# 0	Y2~X1|X3+X4	     X1	-0.079504	0.082208	-0.967109	0.333489
# 0	Y2~X2|X3+X4	     X2	-0.509534	0.001034	-492.767382	0.000000


```

Support for more [fixest formula-sugar](https://cran.r-project.org/web/packages/fixest/vignettes/multiple_estimations.html) is work in progress.

