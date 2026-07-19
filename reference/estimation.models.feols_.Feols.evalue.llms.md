# Feols.evalue

``` python
evalue(mixture_precision=1.0)
```

Compute coefficient-wise SAVI e-values.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| mixture_precision | float | Positive mixture precision fixed before sequential monitoring. Defaults to 1. Use `pyfixest.optimal_mixture_precision()` to minimize confidence-sequence width at a target sample size. | `1.0` |

## Returns

| Name | Type      | Description                  |
|------|-----------|------------------------------|
|      | pd.Series | One e-value per coefficient. |

## Notes

SAVI currently supports unweighted, non-IV `feols` models without absorbed fixed effects. The covariance estimator must be iid or heteroskedasticity robust (`hetero`, `HC1`, `HC2`, or `HC3`). Note that for `HC2`/`HC3`, pyfixest’s default small-sample correction scales the variance by `n / (n - k)` while the R implementation in `avlm` does not. Inference is pointwise / by coefficient.

## Examples

``` python
import pyfixest as pf

data = pf.get_data()
fit = pf.feols("Y ~ X1 + X2", data=data, vcov="hetero")
fit.evalue()
```

    Intercept    6.193976e+12
    X1           5.321643e+29
    X2           2.465287e+12
    Name: e_value, dtype: float64
