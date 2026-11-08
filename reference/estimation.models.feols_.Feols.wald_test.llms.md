# Feols.wald_test

``` python
wald_test(R=None, q=None, distribution='F')
```

Conduct Wald test.

Compute a Wald test for a linear hypothesis of the form R \* beta = q. where R is m x k matrix, beta is a k x 1 vector of coefficients, and q is m x 1 vector. By default, tests the joint null hypothesis that all coefficients are zero.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| R | array - like | The matrix R of the linear hypothesis. If None, defaults to an identity matrix. | `None` |
| q | array - like | The vector q of the linear hypothesis. If None, defaults to a vector of zeros. | `None` |
| distribution | str | The distribution to use for the p-value. Can be either “F” or “chi2”. Defaults to “F”. The F distribution is only used for the joint null that all coefficients are zero; any other restriction falls back to “chi2” with a warning. | `'F'` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | WaldTest | The test statistic of the reference distribution used, its p-value, both scalings of the Wald quadratic form, and the degrees of freedom. See [WaldTest](../reference/estimation.state.WaldTest.llms.md). |

## Examples

``` python
import numpy as np
import pyfixest as pf

data = pf.get_data()
fit = pf.feols("Y ~ X1 + X2| f1", data, vcov={"CRV1": "f1"}, ssc=pf.ssc(k_adj=False))

R = np.array([[1,-1]] )
q = np.array([0.0])

# Wald test
wald = fit.wald_test(R=R, q=q, distribution = "chi2")

print(f"Python statistic: {wald.stat}")
print(f"Python p-value: {wald.pvalue}")
```

    Python statistic: 126.40650474043474
    Python p-value: 2.5053092828142597e-29
