# estimation.quantreg.quantreg_.Quantreg { #pyfixest.estimation.quantreg.quantreg_.Quantreg }

```python
estimation.quantreg.quantreg_.Quantreg(
    FixestFormula,
    data,
    ssc_dict,
    drop_singletons,
    drop_intercept,
    weights,
    weights_type,
    collin_tol,
    fixef_tol,
    fixef_maxiter,
    lookup_demeaned_data,
    solver='np.linalg.solve',
    demeaner_backend='numba',
    store_data=True,
    copy_data=True,
    lean=False,
    context=0,
    sample_split_var=None,
    sample_split_value=None,
    quantile=0.5,
    method='fn',
    quantile_tol=1e-06,
    quantile_maxiter=None,
    seed=None,
)
```

Quantile regression model.

## Attributes

| Name | Description |
| --- | --- |
| [objective_value](#pyfixest.estimation.quantreg.quantreg_.Quantreg.objective_value) | Compute the total loss of the quantile regression model. |

## Methods

| Name | Description |
| --- | --- |
| [fit_qreg_fn](#pyfixest.estimation.quantreg.quantreg_.Quantreg.fit_qreg_fn) | Fit a quantile regression model using the Frisch-Newton Interior Point Solver. |
| [fit_qreg_pfn](#pyfixest.estimation.quantreg.quantreg_.Quantreg.fit_qreg_pfn) | Fit a quantile regression model using the Frisch-Newton Interior Point Solver with pre-processing. |
| [get_fit](#pyfixest.estimation.quantreg.quantreg_.Quantreg.get_fit) | Fit a quantile regression model using the interior point method. |
| [get_performance](#pyfixest.estimation.quantreg.quantreg_.Quantreg.get_performance) | Compute performance metrics for the quantile regression model. |
| [prepare_model_matrix](#pyfixest.estimation.quantreg.quantreg_.Quantreg.prepare_model_matrix) | Prepare model inputs for estimation. |
| [to_array](#pyfixest.estimation.quantreg.quantreg_.Quantreg.to_array) | Turn estimation DataFrames to np arrays. |

### fit_qreg_fn { #pyfixest.estimation.quantreg.quantreg_.Quantreg.fit_qreg_fn }

```python
estimation.quantreg.quantreg_.Quantreg.fit_qreg_fn(
    X,
    Y,
    q,
    tol=None,
    maxiter=None,
    beta_init=None,
)
```

Fit a quantile regression model using the Frisch-Newton Interior Point Solver.

### fit_qreg_pfn { #pyfixest.estimation.quantreg.quantreg_.Quantreg.fit_qreg_pfn }

```python
estimation.quantreg.quantreg_.Quantreg.fit_qreg_pfn(
    X,
    Y,
    q,
    m=None,
    tol=None,
    maxiter=None,
    beta_init=None,
    rng=None,
    eta=None,
)
```

Fit a quantile regression model using the Frisch-Newton Interior Point Solver with pre-processing.

### get_fit { #pyfixest.estimation.quantreg.quantreg_.Quantreg.get_fit }

```python
estimation.quantreg.quantreg_.Quantreg.get_fit()
```

Fit a quantile regression model using the interior point method.

### get_performance { #pyfixest.estimation.quantreg.quantreg_.Quantreg.get_performance }

```python
estimation.quantreg.quantreg_.Quantreg.get_performance()
```

Compute performance metrics for the quantile regression model.

### prepare_model_matrix { #pyfixest.estimation.quantreg.quantreg_.Quantreg.prepare_model_matrix }

```python
estimation.quantreg.quantreg_.Quantreg.prepare_model_matrix()
```

Prepare model inputs for estimation.

### to_array { #pyfixest.estimation.quantreg.quantreg_.Quantreg.to_array }

```python
estimation.quantreg.quantreg_.Quantreg.to_array()
```

Turn estimation DataFrames to np arrays.