# VcovSpec

``` python
VcovSpec(
    vcov_type,
    vcov_type_detail,
    clustervar=(),
    lag=None,
    time_id=None,
    panel_id=None,
)
```

The requested covariance estimator, parsed from the `vcov` argument.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| vcov_type | str | Estimator family: `"iid"`, `"hetero"`, `"HAC"`, `"CRV"`, or `"nid"`. | *required* |
| vcov_type_detail | str | Requested estimator, for example `"HC1"`, `"NW"`, or `"CRV1"`. | *required* |
| clustervar | tuple\[str, …\] | Cluster variables; empty unless clustered. | `()` |
| lag | int or None | Newey-West or Driscoll-Kraay lag; `None` unless HAC. | `None` |
| time_id | str or None | Time variable of HAC inference; `None` unless HAC. | `None` |
| panel_id | str or None | Panel variable of HAC inference; `None` unless HAC. | `None` |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 | f1", pf.get_data(), vcov={"CRV1": "f1+f2"})
fit.variance_covariance.spec
```

    VcovSpec(vcov_type='CRV', vcov_type_detail='CRV1', clustervar=('f1', 'f2'), lag=None, time_id=None, panel_id=None)

## Attributes

| Name | Description |
|----|----|
| [VcovSpec.is_clustered](#pyfixest.estimation.state.VcovSpec.is_clustered) | Whether the estimator clusters on at least one variable. |

## Methods

| Name | Description |
|----|----|
| [VcovSpec.from_user_input](#pyfixest.estimation.state.VcovSpec.from_user_input) | Parse and validate the `vcov` and `vcov_kwargs` arguments. |

### VcovSpec.from_user_input

``` python
from_user_input(vcov, vcov_kwargs=None)
```

Parse and validate the `vcov` and `vcov_kwargs` arguments.

The check is syntactic, so the estimation functions run it before fitting and `vcov()` runs it on each call. Whether a fitted model supports the parsed estimator is checked by the model.

Raises `TypeError` for input of the wrong type and `ValueError` for unknown or incomplete values. `vcov_kwargs` is only accepted with `vcov="NW"` or `vcov="DK"`; other estimators reject it.
