# GlmWorkingState

``` python
GlmWorkingState(
    working_response_within,
    design_within,
    working_weights,
    eta,
    mu,
    response_residuals,
    working_residuals,
)
```

Final GLM IRLS state in within scale.

`working_weights` are the IRLS weights of the last iteration, `W_i = w_i / (g'(mu_i)^2 V(mu_i))`, where `w_i` is the user-supplied observation weight (one when unweighted). They therefore already contain the observation weights, which live separately and unchanged in `ObservationWeights`; nothing downstream multiplies by `w` again. For the Gaussian family `W` equals `w`. Square-root weighted arrays are solver-local temporaries and deliberately absent.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| working_response_within | NDArray\[np.float64\] | Final within-scale working response, shape (n_rows,). | *required* |
| design_within | NDArray\[np.float64\] | Demeaned working regressors after removing collinear columns, shape (n_rows, n_coefficients). | *required* |
| working_weights | NDArray\[np.float64\] | Final IRLS weights including observation weights, shape (n_rows,). | *required* |
| eta | NDArray\[np.float64\] | Linear predictor including any offset, shape (n_rows,). | *required* |
| mu | NDArray\[np.float64\] | Response mean, shape (n_rows,). | *required* |
| response_residuals | NDArray\[np.float64\] | Observed response minus its mean, shape (n_rows,). | *required* |
| working_residuals | NDArray\[np.float64\] | Within working response minus design times coefficients, shape (n_rows,). | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.feglm("Y ~ X1", pf.get_data(), family="gaussian")
fit.working_state.mu[:3]
```

    array([-1.08165392, -0.08156808, -1.08165392])
