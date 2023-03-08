## pyfixest

This is a draft package (highly experimental!) for a Python clone of the excellent [fixest](https://github.com/lrberge/fixest) package.

Fixed effects are projected out via the [PyHDFE](https://github.com/jeffgortmaker/pyhdfe) package.

```python
from pyfixest.fixest import Fixest
from pyfixest.utils import get_data

data = get_data()

fixest = Fixest(data = data)
fixest.feols("Y~X1 | X2", vcov = "HC1")
fixest.tidy()
#         coefnames      coef        se     tstat    pvalue
# fml
# Y~X1|X2        X1 -0.103285  0.172965 -0.597142  0.550412

fixest.feols("Y~X1  | X2 + X3 + X4", vcov = "HC1")
fixest.tidy()
#               coefnames      coef        se     tstat    pvalue
# fml
# Y~X1|X2              X1 -0.103285  0.172965 -0.597142  0.550412
# Y~X1|X2+X3+X4        X1 -0.010369  0.010073 -1.029399  0.303292

fixest.feols("Y~X1 | csw0(X3, X4)", vcov = "HC1")
fixest.tidy()
#                coefnames      coef        se      tstat    pvalue
# fml
# Y~X1|X2               X1 -0.103285  0.172965  -0.597142  0.550412
# Y~X1|X2+X3+X4         X1 -0.010369  0.010073  -1.029399  0.303292
# Y~X1|0         Intercept  7.386158  0.187834  39.322750  0.000000
# Y~X1|0                X1 -0.163744  0.186504  -0.877964  0.379963
# Y~X1|X3               X1 -0.117885  0.178658  -0.659834  0.509360
# Y~X1|X3+X4            X1 -0.063646  0.074755  -0.851397  0.394549

# change inference to HC3
 fixest.vcov("HC3").tidy()
#                coefnames      coef        se      tstat    pvalue
# fml
# Y~X1|X2               X1 -0.103285  0.172931  -0.597259  0.550334
# Y~X1|X2+X3+X4         X1 -0.010369  0.010071  -1.029600  0.303198
# Y~X1|0         Intercept  7.386158  0.187806  39.328639  0.000000
# Y~X1|0                X1 -0.163744  0.186467  -0.878136  0.379870
# Y~X1|X3               X1 -0.117885  0.178623  -0.659961  0.509279
# Y~X1|X3+X4            X1 -0.063646  0.074740  -0.851569  0.394454

```

Support for more [fixest formula-sugar](https://cran.r-project.org/web/packages/fixest/vignettes/multiple_estimations.html) is work in progress.

