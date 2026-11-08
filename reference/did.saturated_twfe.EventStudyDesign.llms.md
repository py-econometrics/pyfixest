# EventStudyDesign

``` python
EventStudyDesign(yname, idname, tname, gname, att, cohort_event_times)
```

Event-study design a fitted event study was estimated on.

Published as `fit.event_study_design` by [event_study()](../reference/did.estimation.event_study.llms.md) with `estimator="saturated"`. It is the only record of the design the fitted model retains; the generic fitted-model classes carry no DiD state.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| yname | str | Name of the outcome variable. | *required* |
| idname | str | Name of the unit identifier variable. | *required* |
| tname | str | Name of the calendar-period variable. | *required* |
| gname | str | Name of the variable holding the unit-specific period of initial treatment. | *required* |
| att | bool | Whether the average treatment effect on the treated was estimated instead of the canonical event study with all leads and lags. | *required* |
| cohort_event_times | Mapping\[str, Mapping\[str, Any\]\] | Cohort-specific event-study curves, keyed by treatment cohort. Each value holds the tidy coefficient table of that cohort (`"est"`) and its event times (`"time"`). | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.event_study(
    pf.get_motherhood_event_study_data(),
    yname="log_earnings",
    idname="unit",
    tname="year",
    gname="g",
    estimator="saturated",
)
design = fit.event_study_design
design.gname, sorted(design.cohort_event_times)
```

    ('g',
     ['2003',
      '2004',
      '2005',
      '2006',
      '2007',
      '2008',
      '2009',
      '2010',
      '2011',
      '2012',
      '2013',
      '2014',
      '2015',
      '2016',
      '2017'])
