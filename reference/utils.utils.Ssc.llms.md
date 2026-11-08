# Ssc

``` python
Ssc(k_adj=True, k_fixef='nonnested', G_adj=True, G_df='min')
```

Small-sample correction options.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| k_adj | bool | Apply the `(N - 1) / (N - k)` adjustment (`N / (N - k)` for heteroskedasticity-robust estimators). | `True` |
| k_fixef | ('none', 'full', 'nonnested') | Which fixed-effect parameters count toward `k`. | `"none"` |
| G_adj | bool | Apply the `G / (G - 1)` cluster adjustment. | `True` |
| G_df | ('min', 'conventional') | Whether multiway clustering adjusts every dimension with the smallest cluster count or with its own. | `"min"` |

## Examples

``` python
import pyfixest as pf

pf.ssc(k_adj=False)
```

    Ssc(k_adj=False, k_fixef='nonnested', G_adj=True, G_df='min')

## Methods

| Name | Description |
|----|----|
| [Ssc.from_mapping](#pyfixest.utils.utils.Ssc.from_mapping) | Build from a legacy `{"k_adj": ..., "k_fixef": ..., ...}` dict. |

### Ssc.from_mapping

``` python
from_mapping(mapping)
```

Build from a legacy `{"k_adj": ..., "k_fixef": ..., ...}` dict.
