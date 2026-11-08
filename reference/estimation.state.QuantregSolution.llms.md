# QuantregSolution

``` python
QuantregSolution(beta, has_converged, iterations, x, s, z, w, y)
```

Solution of the Frisch-Newton interior point solver.

Fields follow the notation of Portnoy and Koenker (1997), [Statistical Science](https://doi.org/10.1214/ss/1030037960).

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| beta | NDArray\[np.float64\] | Coefficient estimates, shape (n_coefficients,); the negated `y`. | *required* |
| has_converged | bool | Whether the duality gap fell below the tolerance within the iteration budget. | *required* |
| iterations | int | Index of the final interior point iteration, counted from zero. | *required* |
| x | NDArray\[np.float64\] | Primal variable, shape (n_solver_rows,). | *required* |
| s | NDArray\[np.float64\] | Primal slack of the upper bound, `u - x`, shape (n_solver_rows,). | *required* |
| z | NDArray\[np.float64\] | Dual variable of `x >= 0`, shape (n_solver_rows,). | *required* |
| w | NDArray\[np.float64\] | Dual variable of `x <= u`, shape (n_solver_rows,). | *required* |
| y | NDArray\[np.float64\] | Dual variable of `A x = b`, shape (n_coefficients,). | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.quantreg("Y ~ X1", pf.get_data(), quantile=0.5)
fit.solution.has_converged, fit.solution.iterations
```

    (True, 10)
