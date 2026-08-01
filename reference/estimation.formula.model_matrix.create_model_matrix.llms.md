# create_model_matrix

``` python
create_model_matrix(
    formula,
    data,
    weights=None,
    offset=None,
    drop_singletons=False,
    drop_intercept=False,
    ensure_full_rank=True,
    context=0,
)
```

Create a ModelMatrix from a formula and data.

This function constructs model matrices for econometric estimation by parsing formulas and extracting the necessary components (dependent/independent variables, fixed effects, instruments, weights) from the provided data.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| formula | Formula | A Formula object specifying the model structure, including dependent and independent variables, fixed effects, and instrumental variables. | *required* |
| data | pd.DataFrame | The input data containing all variables referenced in the formula. The index will be reset during processing. | *required* |
| weights | str or None | Column name in data to use as observation weights. Weights must be non-negative numeric values. If None, no weighting is applied. | `None` |
| offset | str or None | Column name in data to use as an offset (added to the linear predictor with a fixed coefficient of 1). Rows with NaN in the offset column are dropped together with NaN rows in the rest of the formula. | `None` |
| drop_singletons | bool | If True, observations that are singletons in any fixed effect category are dropped from the model. | `False` |
| drop_intercept | bool | If True, the intercept column is removed from the independent variables and instruments matrices. The intercept is always removed when fixed effects are present, regardless of this parameter. | `False` |
| ensure_full_rank | bool | If True, formulaic will ensure the design matrix is full rank by dropping collinear columns. | `True` |
| context | int or Mapping\[str, Any\] | Additional context variables for formulaic during model matrix creation. Can be an integer (stack frame depth) or a dictionary of variables to make available in the formula environment (e.g., custom transformations). | `0` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | ModelMatrix | A ModelMatrix object containing the processed dependent and independent variables, fixed effects, instruments, weights, and metadata about dropped observations. |

## Examples

``` python
import pyfixest as pf
from pyfixest.estimation.formula.model_matrix import create_model_matrix
from pyfixest.estimation.formula.parse import Formula

data = pf.get_data()
formula = Formula.parse("Y ~ X1 + f1 + f2")[0]
model_matrix = create_model_matrix(formula=formula, data=data)
```
