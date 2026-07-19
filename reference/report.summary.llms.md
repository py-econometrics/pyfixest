# summary

``` python
summary(models, digits=3, inference_type='regular')
```

Print a summary of estimation results for each estimated model.

For each model, this method prints a header indicating the fixed-effects and the dependent variable, followed by a table of coefficient estimates with standard errors, t-values, and p-values.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| models | Feols, Fepois, Feiv, FixestMulti, or list | A fitted model object, or a list of Feols, Fepois, and Feiv models. | *required* |
| digits | int | The number of decimal places to round the summary statistics to. Default is 3. | `3` |
| inference_type | regular | Type of coefficient-wise inference to report, handled the same way as in `tidy()`. Only `"regular"` is currently available. Defaults to `"regular"`. | `"regular"` |

## Returns

| Name | Type | Description |
|------|------|-------------|
|      | None |             |

## Examples

``` python
import pyfixest as pf

# load data
df = pf.get_data()
fit1 = pf.feols("Y~X1 + X2 | f1", df)
fit2 = pf.feols("Y~X1 + X2 | f1 + f2", df)
fit3 = pf.feols("Y~X1 + X2 | f1 + f2 + f3", df)

pf.summary([fit1, fit2, fit3])
```

    ###

    Estimation:  OLS
    Dep. var.: Y, Fixed effects: f1
    sample: None = all
    Inference:  iid
    Observations:  997

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | X1            |     -0.950 |        0.066 |   -14.306 |      0.000 | -1.080 |  -0.819 |
    | X2            |     -0.174 |        0.018 |    -9.902 |      0.000 | -0.209 |  -0.140 |
    ---
    RMSE: 1.648 R2: 0.489 R2 Within: 0.239 
    ###

    Estimation:  OLS
    Dep. var.: Y, Fixed effects: f1 + f2
    sample: None = all
    Inference:  iid
    Observations:  997

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | X1            |     -0.924 |        0.056 |   -16.483 |      0.000 | -1.034 |  -0.814 |
    | X2            |     -0.174 |        0.015 |   -11.717 |      0.000 | -0.203 |  -0.145 |
    ---
    RMSE: 1.346 R2: 0.659 R2 Within: 0.303 
    ###

    Estimation:  OLS
    Dep. var.: Y, Fixed effects: f1 + f2 + f3
    sample: None = all
    Inference:  iid
    Observations:  997

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | X1            |     -0.957 |        0.041 |   -23.321 |      0.000 | -1.038 |  -0.877 |
    | X2            |     -0.194 |        0.011 |   -17.895 |      0.000 | -0.215 |  -0.173 |
    ---
    RMSE: 0.97 R2: 0.823 R2 Within: 0.481 
