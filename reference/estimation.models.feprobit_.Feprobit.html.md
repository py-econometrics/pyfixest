# estimation.models.feprobit_.Feprobit { #pyfixest.estimation.models.feprobit_.Feprobit }

```python
estimation.models.feprobit_.Feprobit(
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
    demeaner_backend='numba',
    store_data=True,
    copy_data=True,
    lean=False,
    sample_split_var=None,
    sample_split_value=None,
    separation_check=None,
    context=0,
    accelerate=True,
)
```

Class for the estimation of a fixed-effects probit model.