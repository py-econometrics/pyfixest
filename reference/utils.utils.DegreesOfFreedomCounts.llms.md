# DegreesOfFreedomCounts

``` python
DegreesOfFreedomCounts(
    N,
    k,
    k_fe,
    n_fe,
    k_fe_nested=0,
    n_fe_fully_nested=0,
    k_slopes=0,
    G,
)
```

Model counts that enter the small-sample correction.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| N | int or float | Number of observations (the frequency-weight sum under fweights). | *required* |
| k | int | Number of estimated coefficients, excluding fixed effects. | *required* |
| k_fe | int | Number of levels across the fixed effects with intercepts. | *required* |
| n_fe | int | Number of fixed effects with intercepts; `Y ~ X \| f1 + f2` has two. | *required* |
| k_fe_nested | int | Fixed-effect levels nested within the cluster variables. | `0` |
| n_fe_fully_nested | int | Fixed effects fully nested within the cluster variables. | `0` |
| k_slopes | int | Number of varying-slope coefficients. As in fixest, they count under every `k_fixef` option, including `"none"`. | `0` |
| G | int or float | Number of clusters; the number of time periods for HAC and `N` for heteroskedasticity-robust inference. | *required* |
