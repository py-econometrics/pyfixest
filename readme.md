## pyfixest

This is a draft package (highly experimental!) for a Python clone of the excellent [fixest](https://github.com/lrberge/fixest) package.

Fixed effects are projected out via the [PyHDFE](https://github.com/jeffgortmaker/pyhdfe) package.

```python
import pandas as pd
import numpy as np
from pyfixest.fixest import Fixest
from pyfixest.utils import get_data

data = get_data()

fixest = Fixest(data = data)
fixest.feols("Y~X1 | X2 + X3", vcov = "HC1")
fixest.summary()
#            coefnames      coef        se    tstat   pvalue
# fml                                                       
# Y~X1|X2+X3        X1 -0.063819  0.164427 -0.38813  0.69792

fixest.feols("Y~X1  | X2 + X3 + X4", vcov = "HC1")
fixest.summary()
#               coefnames      coef        se     tstat    pvalue
# fml                                                            
# Y~X1|X2+X3           X1 -0.063819  0.164427 -0.388130  0.697920
# Y~X1|X2+X3+X4        X1 -0.010369  0.010073 -1.029399  0.303292

fixest.feols("Y~X1 | csw0(X2, X3)", vcov = "HC1")
fixest.summary()
#                coefnames      coef        se      tstat    pvalue
# fml                                                              
# Y~X1|X2+X3            X1 -0.063819  0.164427  -0.388130  0.697920
# Y~X1|X2+X3+X4         X1 -0.010369  0.010073  -1.029399  0.303292
# Y~X1|0         Intercept  7.386158  0.187834  39.322750  0.000000
# Y~X1|0                X1 -0.163744  0.186504  -0.877964  0.379963
# Y~X1|X2               X1 -0.103285  0.172965  -0.597142  0.550412
 

```

Support for more [fixest formula-sugar](https://cran.r-project.org/web/packages/fixest/vignettes/multiple_estimations.html) is work in progress.

