# get_twin_data

``` python
get_twin_data(N_pairs=500, seed=42)
```

Generate twin study data for returns to education.

Inspired by Ashenfelter & Krueger (1994). Each twin pair shares an unobserved `ability` component. The true return to education is 0.08 log-points per year; naive OLS is biased upward because ability is correlated with both education and wages.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| N_pairs | int | Number of twin pairs. Total observations = 2 \* N_pairs. | `500` |
| seed | int | Random seed for reproducibility. | `42` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | pd.DataFrame | Columns: twin_pair_id, twin_id, ability, educ, age, experience, log_wage. |

## Examples

``` python
import pyfixest as pf

data = pf.get_twin_data()

# OLS is biased upward by ability, twin-pair FE recover the true 0.08
pf.etable(
    [
        pf.feols("log_wage ~ educ", data),
        pf.feols("log_wage ~ educ | twin_pair_id", data),
    ]
)
```

[TABLE]
