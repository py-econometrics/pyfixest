## pyfixest

This is a draft package (highly experimental!) for a Python clone of the excellent [fixest](https://github.com/lrberge/fixest) package. 

Fixed effects are projected out via the [PyHDFE](https://github.com/jeffgortmaker/pyhdfe) package. 

```python
import pandas as pd
import numpy as np
from pyfixest.api import feols

# create data
np.random.seed(12312312)
N = 10000
k = 3
G = 25
X = np.random.normal(0, 1, N * k).reshape((N,k))
X = pd.DataFrame(X)
X[1] = np.random.choice(list(range(0, 50)), N, True)
X[2] = np.random.choice(list(range(0, 50)), N, True)
beta = np.random.normal(0,1,k)
beta[0] = 0.005
u = np.random.normal(0,1,N)
Y = 1 + X @ beta + u
cluster = np.random.choice(list(range(0,G)), N)

Y = pd.DataFrame(Y)
Y.rename(columns = {0:'Y'}, inplace = True)
X = pd.DataFrame(X)

data = pd.concat([Y, X], axis = 1)
data.rename(columns = {0:'X1'}, inplace = True)
data.rename(columns = {1:'X2'}, inplace = True)
data.rename(columns = {2:'X3'}, inplace = True)
data.rename(columns = {3:'X4'}, inplace = True)
data.columns
data['X3'] = data['X3'].astype('category')
data['X2'] = data['X2'].astype('category')

fml = 'Y ~ X1 + X2 + X3'
vcov = "hetero"

res_fe = feols('Y ~ X1 | X2 + X3', 'hetero', data) 
>>> res_fe['coef']
array([0.01228013])
>>> res_fe['se']
array([1.22801306])
>>> res_fe['pvalue']
array([0.99202149])
>>> res_fe['tstat']
array([0.01])


res = feols('Y ~ X1 + X2 + X3', 'hetero', data) 
>>> res['coef'][-1]
0.012280130646355025
>>> res['se'][-1]
1.2280130646353362
>>> res['pvalue'][-1]
0.9920214888642647
>>> res['tstat'][-1]
0.010000000000001355
```
