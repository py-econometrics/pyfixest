# optimal_mixture_precision

``` python
optimal_mixture_precision(nobs, number_of_coefficients, alpha)
```

Compute the mixture precision that minimizes SAVI sequence width at a specified sample size `nobs`.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| nobs | float | Target number of observations. | *required* |
| number_of_coefficients | float | Number of estimated coefficients, including the intercept. | *required* |
| alpha | float | Significance level between zero and one. | *required* |

## Returns

| Name | Type  | Description                                             |
|------|-------|---------------------------------------------------------|
|      | float | Mixture precision optimized for the target sample size. |

## Examples

``` python
import pyfixest as pf

pf.optimal_mixture_precision(
    nobs=100,
    number_of_coefficients=3,
    alpha=0.05,
)
```

    11.61158319474894
