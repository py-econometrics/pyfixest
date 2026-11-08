# EstimationSample

``` python
EstimationSample(dropped_row_index, n_rows, n_obs, dropped_by_stage)
```

Summary of the sample used to fit a model.

`dropped_row_index` holds zero-based row positions within the estimator input after sample splitting.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| dropped_row_index | frozenset\[int\] | Positions of the rows dropped by any filtering stage. | *required* |
| n_rows | int | Number of physical fitted rows. | *required* |
| n_obs | int or float | `n_rows`, or the weight sum for frequency weights. | *required* |
| dropped_by_stage | DroppedRowCounts | Dropped rows by filtering stage; their total equals `len(dropped_row_index)`. | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 | f1", pf.get_data())
fit.sample_info.n_rows, fit.sample_info.n_obs, fit.sample_info.dropped_by_stage.missing
```

    (997, 997, 3)
