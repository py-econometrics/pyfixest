# VarianceCovariance

``` python
VarianceCovariance(vcov, meat, ssc, df_k, df_t, spec, G)
```

Variance-covariance estimate of the coefficients and its building blocks.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| vcov | NDArray\[np.float64\] | Small-sample-adjusted covariance matrix, shape (n_coefficients, n_coefficients). | *required* |
| meat | NDArray\[np.float64\] or None | Adjusted meat of the sandwich, shape (n_coefficients, n_coefficients), so that `vcov == bread @ meat @ bread` with the bread of `fit.sandwich`. For multiway clustering the per-dimension meats enter with their signs and adjustment factors. `None` where no sandwich exists: `"iid"`, `"CRV3"`, and quantile regression. | *required* |
| ssc | NDArray\[np.float64\] | Small-sample adjustment factors. Length one, or one entry per cluster combination for CRV inference: `2**n_clusters - 1` entries. | *required* |
| df_k | int | Number of parameters counted by the `k_adj` adjustment. | *required* |
| df_t | int or float | Degrees of freedom of the t reference distribution. | *required* |
| spec | VcovSpec | The requested estimator: family, detail, cluster variables, and HAC arguments. | *required* |
| G | tuple\[int, …\] | Cluster counts per dimension after the `G_df` rule; empty unless clustered. | *required* |

## Examples

``` python
import numpy as np
import pyfixest as pf

fit = pf.feols("Y ~ X1 | f1", pf.get_data(), vcov={"CRV1": "f1"})
cov = fit.variance_covariance
cov.spec.vcov_type_detail, cov.G, cov.df_t, cov.ssc
```

    ('CRV1', (30,), 29, array([1.03552244]))

``` python
bread = fit.sandwich.bread
np.allclose(cov.vcov, bread @ cov.meat @ bread)
```

    True
