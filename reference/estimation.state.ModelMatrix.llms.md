# ModelMatrix

``` python
ModelMatrix(
    model_matrix,
    drop_rows,
    drop_singletons=True,
    drop_intercept=False,
    fixed_effect_specifications=(),
)
```

A wrapper around formulaic.ModelMatrix for the specification of PyFixest models.

This class organizes and processes model matrices for econometric estimation, extracting dependent and independent variables, fixed effects, instrumental variables, and weights. It handles missing data, singleton observations, and ensures proper formatting for estimation procedures.

Obtain this component from `fit.model_matrix`. Its properties expose the model frames as pandas DataFrames for inspection and internal calculations. See the [formula syntax tutorial](../tutorials/formula-syntax.llms.md).

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| model_matrix | formulaic.model_matrix.ModelMatrices | Model frames produced by formulaic, containing the response, regressors, and any fixed effects, instruments, weights, or offsets. | *required* |
| drop_rows | frozenset\[int\] | Row positions already removed when constructing the model frames. | *required* |
| drop_singletons | bool | Whether to remove singleton fixed-effect groups. | `True` |
| drop_intercept | bool | Whether to remove the structural intercept. | `False` |
| fixed_effect_specifications | tuple\[FixedEffectSpecification, …\] | Parsed fixed effects, which say whether each effect has an intercept and which varying slopes it has. | `()` |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 | f1", pf.get_data())
fit.model_matrix.dependent.head()
```

|     | Y         |
|-----|-----------|
| 3   | 3.319513  |
| 4   | 0.134420  |
| 5   | -0.278350 |
| 6   | -1.519790 |
| 7   | -2.072451 |

This instance retains the model frames after construction. Estimator-level row filters such as GLM separation call `without_rows`, which returns a filtered copy instead of mutating the instance.

## Attributes

| Name | Type | Description |
|----|----|----|
| dependent | pd.DataFrame | The dependent variable(s) (left-hand side of the main equation). |
| independent | pd.DataFrame | The independent variable(s) (right-hand side of the main equation). |
| fixed_effects | pd.DataFrame or None | Fixed effects variables, encoded as integers. |
| effects | list\[within.Effect\] or None | Complete fixed effects with group codes, intercept flags, and slope covariates, or None when no fixed effects are specified. |
| endogenous | pd.DataFrame or None | Endogenous variables in instrumental variable specifications. |
| instruments | pd.DataFrame or None | Instrumental variables for IV estimation. |
| weights | pd.DataFrame or None | Observation weights for weighted estimation. |
| model_spec | Mapping\[str, formulaic.ModelSpec\] | The underlying formulaic model specifications keyed by role. |
| n_rows | int | Number of rows that survived every filtering stage. |
| dropped_row_index | frozenset\[int\] | Positions of the dropped rows in the frame the matrix was built from. |
| dropped_by_stage | DroppedRowCounts | Dropped rows counted by the filtering stage that removed them. |

## Methods

| Name | Description |
|----|----|
| [ModelMatrix.column_keys](#pyfixest.estimation.state.ModelMatrix.column_keys) | Key the columns of one frame by their rendered term and name. |
| [ModelMatrix.without_rows](#pyfixest.estimation.state.ModelMatrix.without_rows) | Return a shallow copy without `rows`, counted under `stage`. |

### ModelMatrix.column_keys

``` python
column_keys(role)
```

Key the columns of one frame by their rendered term and name.

Distinct terms can encode columns with equal names, e.g. `i(f3, X2)` and `i(f3, np.exp(X2 / 5))`, so names alone cannot share cached demeaned columns across models.

### ModelMatrix.without_rows

``` python
without_rows(rows, *, stage)
```

Return a shallow copy without `rows`, counted under `stage`.

The copied object receives a new filtered data frame and dropped-row bookkeeping that counts `rows` under `stage`; its unchanged formula metadata remains shared with the original object. An empty `rows` sequence returns this instance unchanged.
