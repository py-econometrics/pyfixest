# estimation.formula.model_matrix.ModelMatrix { #pyfixest.estimation.formula.model_matrix.ModelMatrix }

```python
estimation.formula.model_matrix.ModelMatrix(
    model_matrix,
    drop_rows,
    drop_singletons=True,
    drop_intercept=False,
)
```

A wrapper around formulaic.ModelMatrix for the specification of PyFixest models.

This class organizes and processes model matrices for econometric estimation,
extracting dependent and independent variables, fixed effects, instrumental
variables, and weights. It handles missing data, singleton observations,
and ensures proper formatting for estimation procedures.

## Attributes {.doc-section .doc-section-attributes}

| Name          | Type                 | Description                                                         |
|---------------|----------------------|---------------------------------------------------------------------|
| dependent     | pd.DataFrame         | The dependent variable(s) (left-hand side of the main equation).    |
| independent   | pd.DataFrame         | The independent variable(s) (right-hand side of the main equation). |
| fixed_effects | pd.DataFrame or None | Fixed effects variables, encoded as integers.                       |
| endogenous    | pd.DataFrame or None | Endogenous variables in instrumental variable specifications.       |
| instruments   | pd.DataFrame or None | Instrumental variables for IV estimation.                           |
| weights       | pd.DataFrame or None | Observation weights for weighted estimation.                        |
| model_spec    | formulaic.ModelSpec  | The underlying formulaic model specification.                       |
| na_index      | frozenset\[int\]     | Indices of rows that were dropped.                                  |