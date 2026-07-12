<!-- Generated from docs/reference/utils.dgps.get_bartik_data.qmd; do not edit. -->

# utils.dgps.get_bartik_data

```python
utils.dgps.get_bartik_data(N=300, seed=1234)
```

Synthetic data for a Bartik (shift-share) IV application on immigration and wages.

## DGP

Unobserved confounder: local_demand ~ N(0, 1)

First stage (immigration on bartik_instrument, conditional on log_population):
    immigration = 0.5 + 0.7*bartik_instrument + 0.9*local_demand + N(0, 0.5)
    → bartik_instrument is relevant; bartik ⊥ local_demand (exogenous)

Outcome (structural equation):
    wages = 8 + 0.5*local_demand + TRUE_EFFECT*immigration + 0.2*log_population + N(0, 1)
    TRUE_EFFECT = -0.3

OVB for naive OLS (wages ~ immigration + log_population):
    Partial bias from local_demand ≈ 0.5 * 0.9/Var(immigration|log_pop) > 0
    β_OLS on immigration ≈ -0.3 + positive_bias → attenuated (less negative or positive)
    β_IV  on immigration ≈ -0.3  (recovers the true effect)

## Parameters

| Name   | Type   | Description                                       | Default   |
|--------|--------|---------------------------------------------------|-----------|
| N      | int    | Number of observations (regions). Default is 300. | `300`     |
| seed   | int    | Random seed. Default is 1234.                     | `1234`    |

## Returns

| Name   | Type             | Description                                                                     |
|--------|------------------|---------------------------------------------------------------------------------|
|        | pandas.DataFrame | Columns: ``wages``, ``immigration``, ``log_population``, ``bartik_instrument``. |
