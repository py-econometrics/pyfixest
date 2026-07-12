<!-- Generated from docs/reference/estimation.models.feiv_.Feiv.qmd; do not edit. -->

# estimation.models.feiv_.Feiv

```python
estimation.models.feiv_.Feiv(
    FixestFormula,
    data,
    ssc_dict,
    drop_singletons,
    drop_intercept,
    weights,
    weights_type,
    collin_tol,
    lookup_demeaned_data,
    solver='scipy.linalg.solve',
    demeaner=None,
    lookup_preconditioner=None,
    store_data=True,
    copy_data=True,
    lean=False,
    context=0,
    sample_split_var=None,
    sample_split_value=None,
)
```

Fitted instrumental-variable model estimated by two-stage least squares.

This class inherits from `Feols`. Construct it with a three-part formula in
[feols()](estimation.api.feols.feols.md), not by calling the
constructor. The planner and runner prepare the model, and this class performs
its own fixed-effects demeaning using the shared cache supplied by the runner.

## Parameters

| Name         | Type                                                                                                      | Description                                                                                               | Default                |
|--------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|------------------------|
| Y            | np.ndarray                                                                                                | Dependent variable, a two-dimensional np.array.                                                           | _required_             |
| X            | np.ndarray                                                                                                | Independent variables, a two-dimensional np.array.                                                        | _required_             |
| endgvar      | np.ndarray                                                                                                | Endogenous Indenpendent variables, a two-dimensional np.array.                                            | _required_             |
| Z            | np.ndarray                                                                                                | Instruments, a two-dimensional np.array.                                                                  | _required_             |
| weights      | np.ndarray                                                                                                | Weights, a one-dimensional np.array.                                                                      | _required_             |
| coefnames_x  | list                                                                                                      | Names of the coefficients of X.                                                                           | _required_             |
| coefnames_z  | list                                                                                                      | Names of the coefficients of Z.                                                                           | _required_             |
| collin_tol   | float                                                                                                     | Tolerance for the collinearity check. Public estimators default to `1e-9`.                                | _required_             |
| solver       | Literal\[\'np.linalg.lstsq\', \'np.linalg.solve\', \'scipy.linalg.solve\', \'scipy.sparse.linalg.lsqr\'\] | "scipy.sparse.linalg.lsqr"], default is "scipy.linalg.solve". Solver to use for the estimation.           | `'scipy.linalg.solve'` |
| demeaner     | Optional\[AnyDemeaner\]                                                                                   | Resolved typed demeaner configuration.                                                                    | `None`                 |
| weights_name | Optional\[str\]                                                                                           | Name of the weights variable.                                                                             | _required_             |
| weights_type | Optional\[str\]                                                                                           | Type of the weights variable. Either "aweights" for analytic weights or "fweights" for frequency weights. | _required_             |

## Attributes

| Name                    | Type         | Description                                                                                                                                                                                                                                                                                                                                                                                                   |
|-------------------------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| first_stage_model       | Feols        | Fitted first-stage regression.                                                                                                                                                                                                                                                                                                                                                                                |
| first_stage_f_statistic | float        | F statistic for joint significance of the excluded instruments.                                                                                                                                                                                                                                                                                                                                               |
| effective_f_statistic   | float        | Olea-Pflueger effective F statistic, available after `IV_Diag()`.                                                                                                                                                                                                                                                                                                                                             |
| _Z                      | np.ndarray   | Processed instruments after handling multicollinearity.                                                                                                                                                                                                                                                                                                                                                       |
| _weights_type_feiv      | str          | Type of the weights variable defined in Feiv class. Either "aweights" for analytic weights or "fweights" for frequency weights.                                                                                                                                                                                                                                                                               |
| _coefnames_z            | list         | Names of coefficients for Z after handling multicollinearity.                                                                                                                                                                                                                                                                                                                                                 |
| _collin_vars_z          | list         | Variables identified as collinear in Z.                                                                                                                                                                                                                                                                                                                                                                       |
| _collin_index_z         | list         | Indices of collinear variables in Z.                                                                                                                                                                                                                                                                                                                                                                          |
| _is_iv                  | bool         | Indicator if instrumental variables are used.                                                                                                                                                                                                                                                                                                                                                                 |
| _support_crv3_inference | bool         | Indicator for supporting CRV3 inference.                                                                                                                                                                                                                                                                                                                                                                      |
| _support_iid_inference  | bool         | Indicator for supporting IID inference.                                                                                                                                                                                                                                                                                                                                                                       |
| _tZX                    | np.ndarray   | Transpose of Z times X.                                                                                                                                                                                                                                                                                                                                                                                       |
| _tXZ                    | np.ndarray   | Transpose of X times Z.                                                                                                                                                                                                                                                                                                                                                                                       |
| _tZy                    | np.ndarray   | Transpose of Z times Y.                                                                                                                                                                                                                                                                                                                                                                                       |
| _tZZinv                 | np.ndarray   | Inverse of transpose of Z times Z.                                                                                                                                                                                                                                                                                                                                                                            |
| _beta_hat               | np.ndarray   | Estimated regression coefficients.                                                                                                                                                                                                                                                                                                                                                                            |
| _Y_hat_link             | np.ndarray   | Predicted values of the regression model.                                                                                                                                                                                                                                                                                                                                                                     |
| _u_hat                  | np.ndarray   | Residuals of the regression model.                                                                                                                                                                                                                                                                                                                                                                            |
| _scores                 | np.ndarray   | Scores used in the regression.                                                                                                                                                                                                                                                                                                                                                                                |
| _hessian                | np.ndarray   | Hessian matrix used in the regression.                                                                                                                                                                                                                                                                                                                                                                        |
| _bread                  | np.ndarray   | Bread matrix used in the regression.                                                                                                                                                                                                                                                                                                                                                                          |
| _pi_hat                 | np.ndarray   | Estimated coefficients from 1st stage regression                                                                                                                                                                                                                                                                                                                                                              |
| _X_hat                  | np.ndarray   | Predicted values of the 1st stage regression                                                                                                                                                                                                                                                                                                                                                                  |
| _v_hat                  | np.ndarray   | Residuals of the 1st stage regression                                                                                                                                                                                                                                                                                                                                                                         |
| _model_1st_stage        | Any          | feols object of 1st stage regression. It contains various results and diagnostics from the fixed effects OLS regression.                                                                                                                                                                                                                                                                                      |
| _endogvar_1st_stage     | np.ndarray   | Unweihgted Endogenous independent variable vector                                                                                                                                                                                                                                                                                                                                                             |
| _Z_1st_stage            | np.ndarray   | Unweighted instruments vector to be used for 1st stage                                                                                                                                                                                                                                                                                                                                                        |
| _non_exo_instruments    | list         | List of instruments name excluding exogenous independent vars.                                                                                                                                                                                                                                                                                                                                                |
| __p_iv                  | scalar       | Number of instruments listed in _non_exo_instruments                                                                                                                                                                                                                                                                                                                                                          |
| _f_stat_1st_stage       | scalar       | F-statistics of First Stage regression for evaluation of IV weakness. The computed F-statistics test the following null hypothesis : # H0 : beta_{z_1} = 0 & ... & beta_{z_{p_iv}} = 0 where z_1, ..., z_{p_iv} # are the instrument variables # H1 : H0 does not hold Note that this F-statistics is adjusted to heteroskedasticity / clusters if users set specification of variance-covariance matrix type |
| _eff_F                  | scalar       | Effective F-statistics of first stage regression as in Olea and Pflueger 2013                                                                                                                                                                                                                                                                                                                                 |
| _data                   | pd.DataFrame | The data frame used in the estimation. None if arguments `lean = True` or `store_data = False`.                                                                                                                                                                                                                                                                                                               |

## Raises

| Name   | Type       | Description                          |
|--------|------------|--------------------------------------|
|        | ValueError | If Z is not a two-dimensional array. |

## Methods

| Name | Description |
| --- | --- |
| [IV_Diag](#pyfixest.estimation.models.feiv_.Feiv.IV_Diag) | Implement IV diagnostic tests. |
| [IV_weakness_test](#pyfixest.estimation.models.feiv_.Feiv.IV_weakness_test) | Implement IV weakness test (F-test). |
| [demean](#pyfixest.estimation.models.feiv_.Feiv.demean) | Demean instruments and endogeneous variable. |
| [drop_multicol_vars](#pyfixest.estimation.models.feiv_.Feiv.drop_multicol_vars) | Drop multicollinear variables in matrix of instruments Z. |
| [eff_F](#pyfixest.estimation.models.feiv_.Feiv.eff_F) | Compute Effective F stat (Olea and Pflueger 2013). |
| [first_stage](#pyfixest.estimation.models.feiv_.Feiv.first_stage) | Implement First stage regression. |
| [get_fit](#pyfixest.estimation.models.feiv_.Feiv.get_fit) | Fit a IV model using a 2SLS estimator. |
| [to_array](#pyfixest.estimation.models.feiv_.Feiv.to_array) | Transform estimation DataFrames to arrays. |
| [wls_transform](#pyfixest.estimation.models.feiv_.Feiv.wls_transform) | Transform variables for WLS estimation. |

### IV_Diag

```python
estimation.models.feiv_.Feiv.IV_Diag(statistics=None)
```

Implement IV diagnostic tests.

#### Notes

This method covers diagnostic tests related with IV regression.
We currently have IV weak tests only. More test will be updated
in future updates!

#### Parameters

| Name       | Type        | Description                      | Default   |
|------------|-------------|----------------------------------|-----------|
| statistics | list\[str\] | List of IV diagnostic statistics | `None`    |

#### Example

The following is an example usage of this method:

    ```python

    import numpy as np
    import pandas as pd
    from pyfixest.estimation import feols

    # Use an isolated random generator for reproducibility
    rng = np.random.default_rng(1)

    # Number of observations
    n = 1000

    # Simulate the data
    # Instrumental variable
    z = rng.binomial(1, 0.5, size=n)
    z2 = rng.binomial(1, 0.5, size=n)

    # Endogenous variable
    d = 0.5 * z + 1.5 * z2 + rng.normal(size=n)

    # Control variables
    c1 = rng.normal(size=n)
    c2 = rng.normal(size=n)

    # Outcome variable
    y = 1.0 + 1.5 * d + 0.8 * c1 + 0.5 * c2 + rng.normal(size=n)

    # Cluster variable
    cluster = rng.integers(1, 50, size=n)
    weights = rng.uniform(1, 3, size=n)

    # Create a DataFrame
    data = pd.DataFrame({
        'd': d,
        'y': y,
        'z': z,
        'z2': z2,
        'c1': c1,
        'c2': c2,
        'cluster': cluster,
        'weights': weights
    })

    vcov_detail = "iid"

    # Fit OLS model
    fit_ols = feols("y ~ 1 + d + c1 + c2", data=data, vcov=vcov_detail)

    # Fit IV model
    fit_iv = feols("y ~ 1 + c1 + c2 | d ~ z", data=data,
             vcov=vcov_detail,
             weights="weights")
    fit_iv.first_stage()
    F_stat_pf = fit_iv.first_stage_f_statistic
    fit_iv.IV_Diag()
    F_stat_eff_pf = fit_iv.effective_f_statistic

    print("(Unadjusted) F stat :", F_stat_pf)
    print("Effective F stat :", F_stat_eff_pf)

    ```

### IV_weakness_test

```python
estimation.models.feiv_.Feiv.IV_weakness_test(iv_diag_statistics=None)
```

Implement IV weakness test (F-test).

This method covers hetero-robust and clustered-robust F statistics.
It produces two statistics:

- `first_stage_f_statistic`: F statistic of the first-stage regression.
- `effective_f_statistic`: Effective F statistic (Olea and Pflueger
  2013) of the first-stage regression.

#### Notes

`first_stage_f_statistic` is adjusted to the specification of vcov.
If vcov_detail = "iid", F statistics is not adjusted,
otherwise it is always adjusted.

#### Parameters

| Name               | Type   | Description                    | Default   |
|--------------------|--------|--------------------------------|-----------|
| iv_diag_statistics | list   | List of IV weakness statistics | `None`    |

### demean

```python
estimation.models.feiv_.Feiv.demean()
```

Demean instruments and endogeneous variable.

### drop_multicol_vars

```python
estimation.models.feiv_.Feiv.drop_multicol_vars()
```

Drop multicollinear variables in matrix of instruments Z.

### eff_F

```python
estimation.models.feiv_.Feiv.eff_F()
```

Compute Effective F stat (Olea and Pflueger 2013).

### first_stage

```python
estimation.models.feiv_.Feiv.first_stage()
```

Implement First stage regression.

### get_fit

```python
estimation.models.feiv_.Feiv.get_fit()
```

Fit a IV model using a 2SLS estimator.

### to_array

```python
estimation.models.feiv_.Feiv.to_array()
```

Transform estimation DataFrames to arrays.

### wls_transform

```python
estimation.models.feiv_.Feiv.wls_transform()
```

Transform variables for WLS estimation.
