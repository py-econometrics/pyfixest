# DroppedRowCounts

``` python
DroppedRowCounts(
    missing=0,
    infinite=0,
    zero_weight=0,
    singleton=0,
    separation=0,
)
```

Counts of rows removed at each sample-filtering stage.

Counts are mutually exclusive and stages run in field order, matching `DropStageOptions`. Sample splitting is not a filtering stage.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| missing | int | Rows with a missing value in any formula variable. | `0` |
| infinite | int | Rows with an infinite value in a materialized column. | `0` |
| zero_weight | int | Rows with an observation weight of zero. | `0` |
| singleton | int | Rows removed as singleton fixed-effect levels (`fixef_rm="singleton"`). | `0` |
| separation | int | Rows removed by the GLM separation check. | `0` |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 | f1", pf.get_data())
fit.sample_info.dropped_by_stage
```

    DroppedRowCounts(missing=3, infinite=0, zero_weight=0, singleton=0, separation=0)

## Attributes

| Name | Description |
|----|----|
| [DroppedRowCounts.total](#pyfixest.estimation.state.DroppedRowCounts.total) | Number of dropped rows over all stages. |
