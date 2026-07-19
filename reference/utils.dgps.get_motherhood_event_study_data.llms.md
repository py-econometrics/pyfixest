# get_motherhood_event_study_data

``` python
get_motherhood_event_study_data(
    n_per_country=280,
    start_year=2000,
    end_year=2020,
    seed=2026,
)
```

Generate a fertility-timing panel for motherhood-penalty event studies.

The DGP encodes: - stronger post-birth penalties in DACH than in Scandinavia - endogenous fertility timing: slower career trajectories lead to earlier births - nontrivial share of never-treated units

## Parameters

| Name          | Type | Description                         | Default |
|---------------|------|-------------------------------------|---------|
| n_per_country | int  | Number of units per country.        | `280`   |
| start_year    | int  | First year in the panel.            | `2000`  |
| end_year      | int  | Last year in the panel (inclusive). | `2020`  |
| seed          | int  | Random seed for reproducibility.    | `2026`  |

## Returns

| Name | Type | Description |
|----|----|----|
|  | pd.DataFrame | Columns: unit, country, region, year, g, treat, rel_year, log_earnings |

## Examples

``` python
import pyfixest as pf

data = pf.get_motherhood_event_study_data()
data.head()
```

|     | unit | country | region | year | g    | treat | rel_year | log_earnings |
|-----|------|---------|--------|------|------|-------|----------|--------------|
| 0   | 1    | DE      | DACH   | 2000 | 2006 | 0     | -6       | 9.871932     |
| 1   | 1    | DE      | DACH   | 2001 | 2006 | 0     | -5       | 9.799586     |
| 2   | 1    | DE      | DACH   | 2002 | 2006 | 0     | -4       | 9.826801     |
| 3   | 1    | DE      | DACH   | 2003 | 2006 | 0     | -3       | 9.897912     |
| 4   | 1    | DE      | DACH   | 2004 | 2006 | 0     | -2       | 9.837200     |

`g` is the year of first birth and `rel_year` the event time.
