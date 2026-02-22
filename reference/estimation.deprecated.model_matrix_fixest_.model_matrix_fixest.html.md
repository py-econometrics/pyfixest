# estimation.deprecated.model_matrix_fixest_.model_matrix_fixest { #pyfixest.estimation.deprecated.model_matrix_fixest_.model_matrix_fixest }

```python
estimation.deprecated.model_matrix_fixest_.model_matrix_fixest(
    FixestFormula,
    data,
    drop_singletons=False,
    weights=None,
    drop_intercept=False,
    context=0,
)
```

Create model matrices for fixed effects estimation.

This function processes the data and then calls
`formulaic.Formula.get_model_matrix()` to create the model matrices.

## Parameters {.doc-section .doc-section-parameters}

| Name            | Type                                                     | Description                                                                                                                                                                                                                                                          | Default    |
|-----------------|----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| FixestFormula   | A pyfixest.estimation.FormulaParser.FixestFormula object | that contains information on the model formula, the formula of the first and second stage, dependent variable, covariates, fixed effects, endogenous variables (if any), and instruments (if any).                                                                   | _required_ |
| data            | pd.DataFrame                                             | The input DataFrame containing the data.                                                                                                                                                                                                                             | _required_ |
| drop_singletons | bool                                                     | Whether to drop singleton fixed effects. Default is False.                                                                                                                                                                                                           | `False`    |
| weights         | str or None                                              | A string specifying the name of the weights column in `data`. Default is None.                                                                                                                                                                                       | `None`     |
| data            | pd.DataFrame                                             | The input DataFrame containing the data.                                                                                                                                                                                                                             | _required_ |
| drop_intercept  | bool                                                     | Whether to drop the intercept from the model matrix. Default is False. If True, the intercept is dropped ex post from the model matrix created by formulaic.                                                                                                         | `False`    |
| context         | int or Mapping\[str, Any\]                               | A dictionary containing additional context variables to be used by formulaic during the creation of the model matrix. This can include custom factorization functions, transformations, or any other variables that need to be available in the formula environment. | `0`        |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|--------|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | dict   | A dictionary with the following keys and value types: - 'Y' : pd.DataFrame     The dependent variable. - 'X' : pd.DataFrame     The Design Matrix. - 'fe' : Optional[pd.DataFrame]     The model's fixed effects. None if not applicable. - 'endogvar' : Optional[pd.DataFrame]     The model's endogenous variable(s), None if not applicable. - 'Z' : np.ndarray     The model's set of instruments (exogenous covariates plus instruments).     None if not applicable. - 'weights_df' : Optional[pd.DataFrame]     DataFrame containing weights, None if weights are not used. - 'na_index' : np.ndarray     Array indicating rows droppled beause of NA values or singleton     fixed effects. - 'na_index_str' : str     String representation of 'na_index'. - '_icovars' : Optional[list[str]]     List of variables interacted with i() syntax, None if not applicable. - 'X_is_empty' : bool     Flag indicating whether X is empty. - 'model_spec' : formulaic ModelSpec     The model specification used to create the model matrices. |

## Examples {.doc-section .doc-section-examples}

```{python}
import pyfixest as pf
from pyfixest.estimation.deprecated.model_matrix_fixest_ import model_matrix_fixest

data = pf.get_data()
fit = pf.feols("Y ~ X1 + f1 + f2", data=data)
FixestFormula = fit.FixestFormula

mm = model_matrix_fixest(FixestFormula, data)
mm
```

.. deprecated::
    This function will be deprecated in a future version.
    Use `pyfixest.estimation.formula.model_matrix.create_model_matrix()` with a `Formula` object instead.
    See https://pyfixest.org/reference/estimation.formula.model_matrix.ModelMatrix.html