<!-- Generated from docs/reference/utils.dgps.get_worker_panel.qmd; do not edit. -->

# utils.dgps.get_worker_panel

```python
utils.dgps.get_worker_panel(N_workers=500, N_firms=50, N_years=11, seed=42)
```

Generate a worker-firm panel dataset with two-way fixed effects.

Inspired by Abowd, Kramarz & Margolis (1999).  Workers switch firms
with ~20 % probability each year.  Both worker and firm fixed effects
contribute to wages.

## Parameters

| Name      | Type   | Description                                        | Default   |
|-----------|--------|----------------------------------------------------|-----------|
| N_workers | int    | Number of workers.                                 | `500`     |
| N_firms   | int    | Number of firms.                                   | `50`      |
| N_years   | int    | Number of years in the panel (starting from 2000). | `11`      |
| seed      | int    | Random seed for reproducibility.                   | `42`      |

## Returns

| Name   | Type         | Description                                                                                  |
|--------|--------------|----------------------------------------------------------------------------------------------|
|        | pd.DataFrame | Columns: worker_id, firm_id, year, female, experience, tenure, log_wage, worker_fe, firm_fe. |
