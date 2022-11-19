## pyfixest

This is a draft package (highly experimental!) for a Python clone of the excellent [fixest](https://github.com/lrberge/fixest) package. 

Fixed effects are projected out via the [PyHDFE](https://github.com/jeffgortmaker/pyhdfe) package. 

```python
import pandas as pd
import numpy as np
from pyfixest.api import feols
import statsmodels.api as sm

# create data
np.random.seed(123)
N = 100000
k = 4
G = 25
X = np.random.normal(0, 1, N * k).reshape((N,k))
X = pd.DataFrame(X)
X[1] = np.random.choice(list(range(0, 50)), N, True)
X[2] = np.random.choice(list(range(0, 1000)), N, True)
X[3] = np.random.choice(list(range(0, 1000)), N, True)

beta = np.random.normal(0,1,k)
beta[0] = 0.005
u = np.random.normal(0,1,N)
Y = 1 + X @ beta + u
cluster = np.random.choice(list(range(0,G)), N)

Y = pd.DataFrame(Y)
Y.rename(columns = {0:'Y'}, inplace = True)
X = pd.DataFrame(X)

data = pd.concat([Y, X], axis = 1)
data.rename(columns = {0:'X1', 1:'X2', 2:'X3', 3:'X4'}, inplace = True)
data['X4'] = data['X4'].astype('category')
data['X3'] = data['X3'].astype('category')
data['X2'] = data['X2'].astype('category')


feols('Y ~ X1 | X2 ', 'iid', data)
#   colnames      coef        se    tstat    pvalue
# 0       X1  1.380586  0.915999  1.50719  0.131765
feols('Y ~ X1 + X2 ', 'iid', data)
#   colnames      coef        se    tstat    pvalue
50         X1    1.380586  0.915999   1.507190  0.131765
sm.ols('Y ~ X1 + X2 ', data).fit().summary()
#   colnames      coef        se    tstat    pvalue
# X1             1.3806      0.916      1.507      0.132```
