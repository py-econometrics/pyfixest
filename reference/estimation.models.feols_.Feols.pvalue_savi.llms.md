# Feols.pvalue_savi

``` python
pvalue_savi(mixture_precision=1.0)
```

Compute coefficient-wise SAVI sequential p-values.

The sequential-p-value analogue of `evalue`. See `evalue` for the `mixture_precision` argument and the supported-model restrictions.

## Returns

| Name | Type      | Description                             |
|------|-----------|-----------------------------------------|
|      | pd.Series | One sequential p-value per coefficient. |

## Examples

``` python
import pyfixest as pf

data = pf.get_data()
fit = pf.feols("Y ~ X1 + X2", data=data, vcov="HC1")
fit.pvalue_savi()
```

    Intercept    1.614472e-13
    X1           1.879119e-30
    X2           4.056323e-13
    Name: Pr(>|t|), dtype: float64
