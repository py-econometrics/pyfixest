# CollinearityCheck

``` python
CollinearityCheck(dropped_coef_names, mask, coefnames)
```

Outcome of one rank check that removes collinear design columns.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| dropped_coef_names | tuple\[str, …\] | Names of the columns removed, in input order. Empty when the design had full rank. | *required* |
| mask | tuple\[bool, …\] | One entry per checked column, `True` where the column was dropped. | *required* |
| coefnames | tuple\[str, …\] | Names of the retained columns, in input order. | *required* |

## Examples

``` python
import pyfixest as pf

data = pf.get_data()
fit = pf.feols("Y ~ X1 + f1 | f1", data)
fit.collinearity.dropped_coef_names, fit.collinearity.coefnames
```

    (('f1',), ('X1',))

## Attributes

| Name | Description |
|----|----|
| [CollinearityCheck.any_dropped](#pyfixest.estimation.state.CollinearityCheck.any_dropped) | Whether the check removed at least one column. |

## Methods

| Name | Description |
|----|----|
| [CollinearityCheck.select](#pyfixest.estimation.state.CollinearityCheck.select) | Return `columns` without the columns this check dropped. |

### CollinearityCheck.select

``` python
select(columns)
```

Return `columns` without the columns this check dropped.

The argument must have one column per entry of `mask`, in the same order as the checked matrix. Returns the input unchanged when the design had full rank.
