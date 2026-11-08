# ObservationWeights

``` python
ObservationWeights(values, weights_type)
```

Canonical observation weights retained by a fitted model.

`values` are always user-scale weights; `None` means no weights. Observation counts, including the frequency-weight sum, live in `EstimationSample`.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| values | NDArray\[np.float64\] or None | Flat, user-scale observation weights. `None` for an unweighted fit. | *required* |
| weights_type | ('aweights', 'fweights') | Weight type. `None` for an unweighted fit. | `"aweights"` |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1", pf.get_data(), weights="weights")
fit.observation_weights.values[:3]
```

    array([0.99092854, 0.0211227 , 0.79081451])

## Attributes

| Name | Description |
|----|----|
| [ObservationWeights.is_weighted](#pyfixest.estimation.state.ObservationWeights.is_weighted) | Whether this state contains user-supplied observation weights. |

## Methods

| Name | Description |
|----|----|
| [ObservationWeights.from_values](#pyfixest.estimation.state.ObservationWeights.from_values) | Construct canonical weighted state from user-scale weights. |
| [ObservationWeights.unweighted](#pyfixest.estimation.state.ObservationWeights.unweighted) | Construct the representation of an unweighted fit. |

### ObservationWeights.from_values

``` python
from_values(weights, *, weights_type)
```

Construct canonical weighted state from user-scale weights.

### ObservationWeights.unweighted

``` python
unweighted()
```

Construct the representation of an unweighted fit.
