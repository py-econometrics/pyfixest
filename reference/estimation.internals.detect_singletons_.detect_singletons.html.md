# estimation.internals.detect_singletons_.detect_singletons { #pyfixest.estimation.internals.detect_singletons_.detect_singletons }

```python
estimation.internals.detect_singletons_.detect_singletons(ids)
```

Detect singleton fixed effects in a dataset.

This function iterates over the columns of a 2D numpy array representing
fixed effects to identify singleton fixed effects.
An observation is considered a singleton if it is the only one in its group
(fixed effect identifier).

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type       | Description                                                                                                                                                           | Default    |
|--------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| ids    | np.ndarray | A 2D numpy array representing fixed effects, with a shape of (n_samples, n_features). Elements should be non-negative integers representing fixed effect identifiers. | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type          | Description                                                                                         |
|--------|---------------|-----------------------------------------------------------------------------------------------------|
|        | numpy.ndarray | A boolean array of shape (n_samples,), indicating which observations have a singleton fixed effect. |

## Notes {.doc-section .doc-section-notes}

The algorithm iterates over columns to identify fixed effects. After each
column is processed, it updates the record of non-singleton rows. This approach
accounts for the possibility that removing an observation in one column can
lead to the emergence of new singletons in subsequent columns.

For performance reasons, the input array should be in column-major order.
Operating on a row-major array can lead to significant performance losses.