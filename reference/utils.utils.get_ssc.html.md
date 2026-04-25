# utils.utils.get_ssc { #pyfixest.utils.utils.get_ssc }

```python
utils.utils.get_ssc(
    ssc_dict,
    N,
    k,
    k_fe,
    k_fe_nested,
    n_fe,
    n_fe_fully_nested,
    G,
    vcov_sign,
    vcov_type,
)
```

Compute small sample adjustment factors.

## Parameters {.doc-section .doc-section-parameters}

| Name              | Type         | Description                                                                                   | Default    |
|-------------------|--------------|-----------------------------------------------------------------------------------------------|------------|
| ssc_dict          | dict         | A dictionary created via the ssc() function.                                                  | _required_ |
| N                 | int          | The number of observations.                                                                   | _required_ |
| k                 | int          | The number of estimated parameters (as in the first part of the model formula)                | _required_ |
| k_fe              | int          | The number of estimated fixed effects (as specified in the second part of the model formula). | _required_ |
| k_fe_nested       | int          | The number of estimated fixed effects nested within clusters.                                 | _required_ |
| n_fe              | int          | The number of fixed effects in the model. I.e. 'Y ~ X1  \| f1 + f2' has 2 fixed effects.      | _required_ |
| n_fe_fully_nested | int          | The number of fixed effects that are fully nested within clusters.                            | _required_ |
| G                 | int          | The number of clusters.                                                                       | _required_ |
| vcov_sign         | array - like | A vector that helps create the covariance matrix.                                             | _required_ |
| vcov_type         | str          | The type of covariance matrix. Must be one of "iid", "hetero", "HAC", or "CRV".               | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type                        | Description                                                                                         |
|--------|-----------------------------|-----------------------------------------------------------------------------------------------------|
|        | tuple of np.ndarray and int | A small sample adjustment factor and the effective number of coefficients k used in the adjustment. |

## Raises {.doc-section .doc-section-raises}

| Name   | Type       | Description                                                                                    |
|--------|------------|------------------------------------------------------------------------------------------------|
|        | ValueError | If vcov_type is not "iid", "hetero", or "CRV", or if G_df is neither "conventional" nor "min". |