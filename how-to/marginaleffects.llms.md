# Marginal Effects and Hypothesis Tests via `marginaleffects`

`PyFixest` integrates with the excellent [marginaleffects](https://github.com/vincentarelbundock/pymarginaleffects) package. Among many other things, `marginaleffects` allows you to compute:

- average marginal effects,
- subgroup marginal effects,
- linear and non-linear hypothesis tests.

``` python
import numpy as np
from marginaleffects import avg_slopes, hypotheses

import pyfixest as pf

data = pf.get_data()
data["Y_bin"] = np.where(data["Y"] > 0, 1, 0)

fit = pf.feglm("Y_bin ~ X1 + X2 | f1", data=data, family="logit")
fit.tidy()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|)  | 2.5%      | 97.5%     |
|-------------|-----------|------------|-----------|--------------|-----------|-----------|
| Coefficient |           |            |           |              |           |           |
| X1          | -1.016344 | 0.109323   | -9.296712 | 0.000000e+00 | -1.230613 | -0.802075 |
| X2          | -0.165899 | 0.028449   | -5.831409 | 5.496138e-09 | -0.221659 | -0.110140 |

## Average marginal effects

To compute the average marginal effect of `X1` on the response scale:

``` python
avg_slopes(fit, variables="X1")
```

shape: (1, 3)

| term | contrast | estimate  |
|------|----------|-----------|
| str  | str      | f64       |
| "X1" | "dY/dX"  | -1.016344 |

You can also report group-specific average marginal effects:

``` python
avg_slopes(fit, variables="X1", by="f1")
```

shape: (30, 4)

| f1   | term | contrast | estimate  |
|------|------|----------|-----------|
| f64  | str  | str      | f64       |
| 0.0  | "X1" | "dY/dX"  | -1.016344 |
| 1.0  | "X1" | "dY/dX"  | -1.016344 |
| 2.0  | "X1" | "dY/dX"  | -1.016344 |
| 3.0  | "X1" | "dY/dX"  | -1.016344 |
| 4.0  | "X1" | "dY/dX"  | -1.016344 |
| …    | …    | …        | …         |
| 25.0 | "X1" | "dY/dX"  | -1.016344 |
| 26.0 | "X1" | "dY/dX"  | -1.016344 |
| 27.0 | "X1" | "dY/dX"  | -1.016344 |
| 28.0 | "X1" | "dY/dX"  | -1.016344 |
| 29.0 | "X1" | "dY/dX"  | -1.016344 |

## Linear hypothesis tests

Suppose we want to test the linear restriction \\X_1 = X_2\\. The `hypotheses()` function uses the model’s estimated covariance matrix, so it works naturally with `PyFixest` objects.

``` python
hypotheses(fit, "X1 - X2 = 0")
```

shape: (1, 8)

| term | estimate | std_error | statistic | p_value | s_value | conf_low | conf_high |
|----|----|----|----|----|----|----|----|
| str | f64 | f64 | f64 | f64 | f64 | f64 | f64 |
| "X1-X2=0" | -0.850445 | 0.1086 | -7.830952 | 4.8850e-15 | 47.540568 | -1.063297 | -0.637592 |

## Non-linear hypothesis tests

For non-linear transformations, `marginaleffects` applies the delta method automatically. This is useful for ratio-style summaries or uplift calculations.

Here is a simple OLS example:

``` python
fit_ols = pf.feols("Y ~ X1 + X2", data=data)
```

``` python
hypotheses(fit_ols, "(X1 / Intercept - 1) * 100 = 0")
```

shape: (1, 8)

| term | estimate | std_error | statistic | p_value | s_value | conf_low | conf_high |
|----|----|----|----|----|----|----|----|
| str | f64 | f64 | f64 | f64 | f64 | f64 | f64 |
| "(X1/Intercept-1)\*100=0" | -211.719067 | 8.478682 | -24.970752 | 0.0 | inf | -228.336978 | -195.101155 |

## Notes

For broader functionality / the full capability of `marginaleffects`, see the [marginaleffects book](https://marginaleffects.com/index.html).
