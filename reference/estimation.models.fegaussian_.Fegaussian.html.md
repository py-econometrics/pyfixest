# estimation.models.fegaussian_.Fegaussian { #pyfixest.estimation.models.fegaussian_.Fegaussian }

```python
estimation.models.fegaussian_.Fegaussian(
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
    tol,
    maxiter,
    solver,
    store_data=True,
    copy_data=True,
    lean=False,
    sample_split_var=None,
    sample_split_value=None,
    separation_check=None,
    context=0,
    demeaner_backend='numba',
    accelerate=True,
)
```

Class for the estimation of a fixed-effects GLM with normal errors.