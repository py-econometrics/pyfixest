<!-- Generated from docs/reference/utils.dgps.get_motherhood_event_study_data.qmd; do not edit. -->

# utils.dgps.get_motherhood_event_study_data

```python
utils.dgps.get_motherhood_event_study_data(
    n_per_country=280,
    start_year=2000,
    end_year=2020,
    seed=2026,
)
```

Generate a fertility-timing panel for motherhood-penalty event studies.

The DGP encodes:
- stronger post-birth penalties in DACH than in Scandinavia
- endogenous fertility timing: slower career trajectories lead to earlier births
- nontrivial share of never-treated units

## Parameters

| Name          | Type   | Description                         | Default   |
|---------------|--------|-------------------------------------|-----------|
| n_per_country | int    | Number of units per country.        | `280`     |
| start_year    | int    | First year in the panel.            | `2000`    |
| end_year      | int    | Last year in the panel (inclusive). | `2020`    |
| seed          | int    | Random seed for reproducibility.    | `2026`    |

## Returns

| Name   | Type         | Description                                                            |
|--------|--------------|------------------------------------------------------------------------|
|        | pd.DataFrame | Columns: unit, country, region, year, g, treat, rel_year, log_earnings |
