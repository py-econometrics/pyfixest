# get_worker_panel

``` python
get_worker_panel(N_workers=500, N_firms=50, N_years=11, seed=42)
```

Generate a worker-firm panel dataset with two-way fixed effects.

Inspired by Abowd, Kramarz & Margolis (1999). Workers switch firms with ~20 % probability each year. Both worker and firm fixed effects contribute to wages.

## Parameters

| Name      | Type | Description                                        | Default |
|-----------|------|----------------------------------------------------|---------|
| N_workers | int  | Number of workers.                                 | `500`   |
| N_firms   | int  | Number of firms.                                   | `50`    |
| N_years   | int  | Number of years in the panel (starting from 2000). | `11`    |
| seed      | int  | Random seed for reproducibility.                   | `42`    |

## Returns

| Name | Type | Description |
|----|----|----|
|  | pd.DataFrame | Columns: worker_id, firm_id, year, female, experience, tenure, log_wage, worker_fe, firm_fe. |

## Examples

``` python
import pyfixest as pf

data = pf.get_worker_panel()

# two-way worker and firm fixed effects, as in AKM
fit = pf.feols("log_wage ~ experience + tenure | worker_id + firm_id", data)
fit.summary()
```

    ###

    Estimation:  OLS
    Dep. var.: log_wage, Fixed effects: worker_id + firm_id
    sample: None = all
    Inference:  iid
    Observations:  5500

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | experience    |      0.021 |        0.001 |    22.547 |      0.000 |  0.019 |   0.023 |
    | tenure        |      0.011 |        0.002 |     6.830 |      0.000 |  0.008 |   0.014 |
    ---
    RMSE: 0.192 R2: 0.908 R2 Within: 0.159 

The true effects are returned as the columns `worker_fe` and `firm_fe`, so estimates can be compared against them.
