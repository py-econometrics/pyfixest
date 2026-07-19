# get_encouragement_data

``` python
get_encouragement_data(N=4000, seed=1234)
```

Synthetic data for an A/B encouragement design IV application.

## DGP

Instrument: assigned_treatment ~ Bernoulli(0.5) \[randomized, exogenous\] Fixed effect: user_type ∈ {0, 1, 2}

First stage (compliance): P(adopt \| encouraged) = 0.70 (compliers + always-takers) P(adopt \| not encouraged) = 0.15 (always-takers only) First-stage coefficient = 0.70 - 0.15 = 0.55

Outcome (structural equation): revenue = 5 + user_type_FE + TRUE_LATE\*adopted_feature + N(0, 1) TRUE_LATE = 2.0 (effect on compliers)

Wald identity (exact by construction): ITT = E\[Y\|Z=1\] - E\[Y\|Z=0\] = 2.0 \* 0.55 = 1.10 LATE = ITT / first_stage = 1.10 / 0.55 = 2.0 ✓

## Parameters

| Name | Type | Description                                      | Default |
|------|------|--------------------------------------------------|---------|
| N    | int  | Number of observations (users). Default is 4000. | `4000`  |
| seed | int  | Random seed. Default is 1234.                    | `1234`  |

## Returns

| Name | Type | Description |
|----|----|----|
|  | pandas.DataFrame | Columns: `revenue`, `assigned_treatment`, `adopted_feature`, `user_type`. |

## Examples

``` python
import pyfixest as pf

data = pf.get_encouragement_data()

# instrument take-up with the randomized encouragement, LATE is 2.0
fit = pf.feols(
    "revenue ~ 1 | user_type | adopted_feature ~ assigned_treatment", data
)
fit.summary()
```

    ###

    Estimation:  IV
    Dep. var.: revenue, Fixed effects: user_type
    sample: None = all
    Inference:  iid
    Observations:  4000

    | Coefficient     |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:----------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | adopted_feature |      2.004 |        0.057 |    34.977 |      0.000 |  1.891 |   2.116 |
    ---
