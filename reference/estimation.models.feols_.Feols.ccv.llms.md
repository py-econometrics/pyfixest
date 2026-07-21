# Feols.ccv

``` python
ccv(treatment, cluster=None, seed=None, n_splits=8, pk=1, qk=1)
```

Compute the Causal Cluster Variance following Abadie et al (QJE 2023).

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| treatment |  | The name of the treatment variable. | *required* |
| cluster | str | The name of the cluster variable. None by default. If None, uses the cluster variable from the model fit. | `None` |
| seed | int | An integer to set the random seed. Defaults to None. | `None` |
| n_splits | int | The number of splits to use in the cross-fitting procedure. Defaults to 8. | `8` |
| pk | float | The proportion of sampled clusters. Defaults to 1, which corresponds to all clusters of the population being sampled. | `1` |
| qk | float | The proportion of sampled observations within each cluster. Defaults to 1, which corresponds to all observations within each cluster being sampled. | `1` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | pd.DataFrame | A DataFrame with inference based on the “Causal Cluster Variance” and “regular” CRV1 inference. |

## Examples

``` python
import pyfixest as pf
import numpy as np

data = pf.get_data()
data["D"] = np.random.choice([0, 1], size=data.shape[0])

fit = pf.feols("Y ~ D", data=data, vcov={"CRV1": "group_id"})
fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=8, seed=123).head()
```

|      | Estimate            | Std. Error | t value  | Pr(\>\|t\|) | 2.5%      | 97.5%    |
|------|---------------------|------------|----------|-------------|-----------|----------|
| CCV  | 0.09327953935461972 | 0.242687   | 0.384361 | 0.705214    | -0.416587 | 0.603146 |
| CRV1 | 0.09328             | 0.177456   | 0.52565  | 0.605547    | -0.279541 | 0.4661   |
