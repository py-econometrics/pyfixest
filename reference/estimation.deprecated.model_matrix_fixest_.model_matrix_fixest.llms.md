# model_matrix_fixest

``` python
model_matrix_fixest(
    FixestFormula,
    data,
    drop_singletons=False,
    weights=None,
    drop_intercept=False,
    context=0,
)
```

Create model matrices for fixed effects estimation.

This function processes the data and then calls `formulaic.Formula.get_model_matrix()` to create the model matrices.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| FixestFormula | A pyfixest.estimation.FormulaParser.FixestFormula object | that contains information on the model formula, the formula of the first and second stage, dependent variable, covariates, fixed effects, endogenous variables (if any), and instruments (if any). | *required* |
| data | pd.DataFrame | The input DataFrame containing the data. | *required* |
| drop_singletons | bool | Whether to drop singleton fixed effects. Default is False. | `False` |
| weights | str or None | A string specifying the name of the weights column in `data`. Default is None. | `None` |
| data | pd.DataFrame | The input DataFrame containing the data. | *required* |
| drop_intercept | bool | Whether to drop the intercept from the model matrix. Default is False. If True, the intercept is dropped ex post from the model matrix created by formulaic. | `False` |
| context | int or Mapping\[str, Any\] | A dictionary containing additional context variables to be used by formulaic during the creation of the model matrix. This can include custom factorization functions, transformations, or any other variables that need to be available in the formula environment. | `0` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | dict | A dictionary with the following keys and value types: - ‘Y’ : pd.DataFrame The dependent variable. - ‘X’ : pd.DataFrame The Design Matrix. - ‘fe’ : Optional\[pd.DataFrame\] The model’s fixed effects. None if not applicable. - ‘endogvar’ : Optional\[pd.DataFrame\] The model’s endogenous variable(s), None if not applicable. - ‘Z’ : np.ndarray The model’s set of instruments (exogenous covariates plus instruments). None if not applicable. - ‘weights_df’ : Optional\[pd.DataFrame\] DataFrame containing weights, None if weights are not used. - ‘na_index’ : np.ndarray Array indicating rows droppled beause of NA values or singleton fixed effects. - ‘na_index_str’ : str String representation of ‘na_index’. - ’\_icovars’ : Optional\[list\[str\]\] List of variables interacted with i() syntax, None if not applicable. - ‘X_is_empty’ : bool Flag indicating whether X is empty. - ‘model_spec’ : formulaic ModelSpec The model specification used to create the model matrices. |

## Examples

``` python
import pyfixest as pf
from pyfixest.estimation.deprecated.model_matrix_fixest_ import model_matrix_fixest

data = pf.get_data()
fit = pf.feols("Y ~ X1 + f1 + f2", data=data)
FixestFormula = fit.FixestFormula

mm = model_matrix_fixest(FixestFormula, data)
mm
```

    {'Y':             Y
     3    3.319513
     4    0.134420
     5   -0.278350
     6   -1.519790
     7   -2.072451
     ..        ...
     995 -2.876714
     996  1.430674
     997 -0.494217
     998 -1.047594
     999  0.105551
     
     [997 rows x 1 columns],
     'X':       X1    f1    f2
     3    1.0   1.0  10.0
     4    2.0  19.0  20.0
     5    2.0  13.0   3.0
     6    1.0   2.0  16.0
     7    0.0   2.0  23.0
     ..   ...   ...   ...
     995  2.0  14.0  23.0
     996  0.0  19.0  17.0
     997  1.0   3.0   5.0
     998  0.0  18.0  20.0
     999  2.0   4.0  19.0
     
     [997 rows x 3 columns],
     'fe': None,
     'endogvar': None,
     'Z': None,
     'weights_df': None,
     'na_index': array([0, 1, 2]),
     'na_index_str': '0,1,2',
     'icovars': None,
     'X_is_empty': False,
     'model_spec': .fml_second_stage:
         .lhs:
             ModelSpec(formula=Y, materializer='pandas', materializer_params={}, ensure_full_rank=True, na_action=<NAAction.DROP: 'drop'>, output='pandas', cluster_by=<ClusterBy.NONE: 'none'>, structure=[EncodedTermStructure(term=Y, scoped_terms=[Y], columns=['Y'])], transform_state={}, encoder_state={'Y': (<Kind.NUMERICAL: 'numerical'>, {})})
         .rhs:
             ModelSpec(formula=X1 + f1 + f2, materializer='pandas', materializer_params={}, ensure_full_rank=True, na_action=<NAAction.DROP: 'drop'>, output='pandas', cluster_by=<ClusterBy.NONE: 'none'>, structure=[EncodedTermStructure(term=X1, scoped_terms=[X1], columns=['X1']), EncodedTermStructure(term=f1, scoped_terms=[f1], columns=['f1']), EncodedTermStructure(term=f2, scoped_terms=[f2], columns=['f2'])], transform_state={}, encoder_state={'X1': (<Kind.NUMERICAL: 'numerical'>, {}), 'f1': (<Kind.NUMERICAL: 'numerical'>, {}), 'f2': (<Kind.NUMERICAL: 'numerical'>, {})})}

.. deprecated:: This function will be deprecated in a future version. Use `pyfixest.estimation.formula.model_matrix.create_model_matrix()` with a `Formula` object instead. See https://pyfixest.org/reference/estimation.formula.model_matrix.ModelMatrix.html
