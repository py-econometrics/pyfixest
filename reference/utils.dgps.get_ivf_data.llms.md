# get_ivf_data

``` python
get_ivf_data(N=2000, seed=1234)
```

Synthetic data for the motherhood penalty IV application (IVF instrument).

## DGP

Unobserved confounder: career_ambition ~ N(0, 1)

First stage (num_children on ivf_success): num_children = 1.2 - 0.4*career_ambition + 0.8*ivf_success + N(0, 0.5) → ivf_success is relevant (first-stage coefficient ≈ 0.8, F \>\> 10)

Outcome (structural equation): earnings = 10 + 0.6*career_ambition + TRUE_EFFECT*num_children + N(0, 1) TRUE_EFFECT = -0.15

OVB formula for naive OLS (earnings ~ num_children): bias ≈ gamma_ambition \* Cov(num_children, ambition) / Var(num_children) ≈ 0.6 \* (-0.4) / 0.57 ≈ -0.42 beta_OLS ≈ -0.15 + (-0.42) ≈ -0.57 (overstates the penalty) beta_IV ≈ -0.15 (recovers the true effect)

## Parameters

| Name | Type | Description                              | Default |
|------|------|------------------------------------------|---------|
| N    | int  | Number of observations. Default is 2000. | `2000`  |
| seed | int  | Random seed. Default is 1234.            | `1234`  |

## Returns

| Name | Type             | Description                                         |
|------|------------------|-----------------------------------------------------|
|      | pandas.DataFrame | Columns: `earnings`, `num_children`, `ivf_success`. |

## Examples

``` python
import pyfixest as pf

data = pf.get_ivf_data()

# OLS is biased by unobserved career ambition, IV recovers the true -0.15
pf.etable(
    [
        pf.feols("earnings ~ num_children", data),
        pf.feols("earnings ~ 1 | num_children ~ ivf_success", data),
    ]
)
```

[TABLE]
