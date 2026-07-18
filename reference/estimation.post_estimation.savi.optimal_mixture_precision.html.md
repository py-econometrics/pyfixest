# estimation.post_estimation.savi.optimal_mixture_precision { #pyfixest.estimation.post_estimation.savi.optimal_mixture_precision }

```python
estimation.post_estimation.savi.optimal_mixture_precision(
    nobs,
    number_of_coefficients,
    alpha,
)
```

Compute the mixture precision that minimizes SAVI sequence width
at a specified sample size `nobs`.

## Parameters {.doc-section .doc-section-parameters}

| Name                   | Type   | Description                                                | Default    |
|------------------------|--------|------------------------------------------------------------|------------|
| nobs                   | float  | Target number of observations.                             | _required_ |
| number_of_coefficients | float  | Number of estimated coefficients, including the intercept. | _required_ |
| alpha                  | float  | Significance level between zero and one.                   | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                             |
|--------|--------|---------------------------------------------------------|
|        | float  | Mixture precision optimized for the target sample size. |

## Examples {.doc-section .doc-section-examples}

```{python}
import pyfixest as pf

pf.optimal_mixture_precision(
    nobs=100,
    number_of_coefficients=3,
    alpha=0.05,
)
```