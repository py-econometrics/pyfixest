## pyfixest

This is a draft package (highly experimental!) for a Python clone of the excellent [fixest](https://github.com/lrberge/fixest) package.

Fixed effects are projected out via the [PyHDFE](https://github.com/jeffgortmaker/pyhdfe) package.

```python
from pyfixest.fixest import Fixest
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

fixest.feols("Y~X1  | X2 + X3 + X4", vcov = "HC1")
fixest.summary()
# ### Fixed-effects: X2
# Dep. var.: Y
# 
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.103285    0.172956 -0.597172  0.550393
# ---
# 
# ### Fixed-effects: X2+X3+X4
# Dep. var.: Y
# 
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.010369    0.010073 -1.029451  0.303268

fixest.feols("Y~X1 | csw0(X3, X4)", vcov = "HC1")
fixest.summary()
# ### Fixed-effects: X2
# Dep. var.: Y
# 
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.103285    0.172956 -0.597172  0.550393
# ---
# 
# ### Fixed-effects: X2+X3+X4
# Dep. var.: Y
# 
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.010369    0.010073 -1.029451  0.303268
# ---
# 
# ### Fixed-effects: 0
# Dep. var.: Y
# 
#            Estimate  Std. Error   t value  Pr(>|t|)
# Intercept  7.386158    0.187825 39.324716  0.000000
#        X1 -0.163744    0.186494 -0.878008  0.379939
# ---
# 
# ### Fixed-effects: X3
# Dep. var.: Y
# 
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.117885    0.178649 -0.659867  0.509339
# ---
# 
# ### Fixed-effects: X3+X4
# Dep. var.: Y
# 
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.063646    0.074751 -0.851439  0.394525
# ---

# change inference to HC3
 fixest.vcov("HC3").summary()
# ### Fixed-effects: X2
# Dep. var.: Y
# 
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.103285    0.172931 -0.597259  0.550334
# ---
# 
# ### Fixed-effects: X2+X3+X4
# Dep. var.: Y
# 
#     Estimate  Std. Error  t value  Pr(>|t|)
# X1 -0.010369    0.010071  -1.0296  0.303198
# ---
# 
# ### Fixed-effects: 0
# Dep. var.: Y
# 
#            Estimate  Std. Error   t value  Pr(>|t|)
# Intercept  7.386158    0.187806 39.328639   0.00000
#        X1 -0.163744    0.186467 -0.878136   0.37987
# ---
# 
# ### Fixed-effects: X3
# Dep. var.: Y
# 
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.117885    0.178623 -0.659961  0.509279
# ---
# 
# ### Fixed-effects: X3+X4
# Dep. var.: Y
# 
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.063646     0.07474 -0.851569  0.394454
# ---

```

Support for more [fixest formula-sugar](https://cran.r-project.org/web/packages/fixest/vignettes/multiple_estimations.html) is work in progress.

