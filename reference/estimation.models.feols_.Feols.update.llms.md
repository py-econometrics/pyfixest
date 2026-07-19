# Feols.update

``` python
update(X_new, y_new, inplace=False)
```

Update coefficients for new observations using Sherman-Morrison formula.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| X_new | np.ndarray | Covariates for new data points. Users expected to ensure conformability with existing data. | *required* |
| y_new | np.ndarray | Outcome values for new data points. | *required* |
| inplace | bool | Whether to update the model object in place. Defaults to False. | `False` |

## Returns

| Name | Type       | Description           |
|------|------------|-----------------------|
|      | np.ndarray | Updated coefficients. |

## Notes

Updates the coefficients in closed form via the Sherman-Morrison identity instead of refitting on the full sample. `X_new` has to include the intercept column. Models with fixed effects are not supported.

## Examples

Fit on all but the last observation, then add it:

``` python
import numpy as np
import pyfixest as pf

data = pf.get_data().dropna()
fit = pf.feols("Y ~ X1 + X2", data.iloc[:-1])

last = data.iloc[[-1]]
X_new = np.column_stack(
    [np.ones(1), last["X1"].to_numpy(), last["X2"].to_numpy()]
)
y_new = last["Y"].to_numpy()

fit.update(X_new, y_new)
```

    array([ 0.88955689, -0.99519687, -0.17661729])
