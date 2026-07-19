# Feols.wald_test

``` python
wald_test(R=None, q=None, distribution='F')
```

Conduct Wald test.

Compute a Wald test for a linear hypothesis of the form R \* beta = q. where R is m x k matrix, beta is a k x 1 vector of coefficients, and q is m x 1 vector. By default, tests the joint null hypothesis that all coefficients are zero.

This method producues the following attriutes

\_dfd : int degree of freedom in denominator \_dfn : int degree of freedom in numerator \_wald_statistic : scalar Wald-statistics computed for hypothesis testing \_f_statistic : scalar Wald-statistics(when R is an indentity matrix, and q being zero vector) computed for hypothesis testing \_p_value : scalar corresponding p-value for statistics

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| R | array - like | The matrix R of the linear hypothesis. If None, defaults to an identity matrix. | `None` |
| q | array - like | The vector q of the linear hypothesis. If None, defaults to a vector of zeros. | `None` |
| distribution | str | The distribution to use for the p-value. Can be either “F” or “chi2”. Defaults to “F”. | `'F'` |

## Returns

| Name | Type      | Description                                      |
|------|-----------|--------------------------------------------------|
|      | pd.Series | A pd.Series with the Wald statistic and p-value. |

## Examples

``` python
import numpy as np
import pandas as pd
import pyfixest as pf

data = pf.get_data()
fit = pf.feols("Y ~ X1 + X2| f1", data, vcov={"CRV1": "f1"}, ssc=pf.ssc(k_adj=False))

R = np.array([[1,-1]] )
q = np.array([0.0])

# Wald test
fit.wald_test(R=R, q=q, distribution = "chi2")
f_stat = fit._f_statistic
p_stat = fit._p_value

print(f"Python f_stat: {f_stat}")
print(f"Python p_stat: {p_stat}")
```

    Python f_stat: 126.40650474043508
    Python p_stat: 2.505309282813844e-29
