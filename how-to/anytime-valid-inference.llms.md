# When to Use Anytime-Valid Inference

Inference

AB Testing

A practical guide to anytime-valid inference in randomized experiments, early stopping, and decisions based on interim results.

## Decisions based on interim A/B-test results

Suppose that your team has finished a new feature and wants to test it with an A/B test. Before launch, you choose the primary KPI, the effect size used for the power calculation, and a 5% false-positive rate. Given the expected traffic and baseline conversion rate, the calculation implies a runtime of twenty days.

While the decision on acceptance or rejection of the feature is scheduled for day twenty, it is often the case that interim results are visible the whole time on the experimentation platform that your company uses, be it built in-house, or supplied by GrowthBook, Eppo, Statsig, or any other platform vendor. It is common that the platform recomputes the treatment effect estimate and p-value each morning, and that everyone on your team has access to the dashboard and can monitor test results before the end of the scheduled runtime of 20 days.

On day one, you peek for the first time: The p-value is 0.15. On day two, it falls to 0.08. On day three, it falls below 0.05 for the first time, and stays there throughout day eight. By now, your team begins discussing whether to ship before the scheduled decision on day 20.

The p-value on the dashboard, however, was not calibrated for a discretionary day-three or day-eight decision to accept or reject the feature. Its 5% false-positive rate is a property of one prespecified analysis: to fit the model once, on day twenty, and reject if \\p \leq 0.05\\. Being open to already taking a decision on day three/eight replaces this plan with a different procedure, one that checks the p-value every day and acts the first time it falls below 0.05. The day-twenty rule rejects when \\p\_{20} \leq 0.05\\; the daily rule rejects when \\\min\_{t \leq 20} p_t \leq 0.05\\.

If the twenty daily analyses were independent, at least one of them would fall below 0.05 with probability \\1 - (1 - 0.05)^{20} \approx 0.64\\ when the true treatment effect is zero. The cumulative analyses are dependent because each one reuses the data from earlier days. The 64% calculation is an independence benchmark, not an upper bound for the cumulative analysis. The 5% guarantee applies to the prespecified day-twenty analysis, but not to a rule that rejects after the first daily p-value at or below 0.05. The simulation below estimates the false-positive rate of that daily stopping rule.

> **NOTE:**
>
> Opening the experimentation platform and “peeking” at the experiment’s interim results in itself is not a problem and does not change the pointwise calibration of that day’s p-value. The pointwise 5% guarantee **does not apply** to a decision rule that uses any interim threshold crossing to stop, ship, roll back, or extend the experiment.

**Anytime-valid inference** calibrates inference for repeated looks at accumulating data when any review may trigger a decision.

## What SAVI changes

Safe anytime-valid inference (SAVI) replaces the fixed-time p-value and confidence interval with SAVI counterparts calibrated for repeated monitoring. In the large randomized experiments considered here, a 5% SAVI test satisfies, under the method’s assumptions, an asymptotic bound of 5% on the probability that the SAVI p-value is at or below 0.05 at any review. The review schedule therefore does not need to be chosen before launch. A prespecified rule that rejects the zero-effect null the first time the SAVI p-value is at or below 0.05 has asymptotic type-I error of at most 5%. The SAVI p-value has an interval companion, the **confidence sequence**: a sequence of intervals that contains the true effect at every review simultaneously, with asymptotic probability of at least 95%.

At the sample sizes used here, the SAVI confidence sequences are wider than fixed-time confidence intervals, and the SAVI p-values are typically larger. The SAVI statistics account for repeated monitoring rather than one prespecified analysis. In the simulation design below, a 4 pp lift often crosses the SAVI threshold before day twenty, while lifts of 1 or 2 pp usually take longer. These stopping times depend on the sample size, outcome variance, monitoring schedule, and mixture precision.

To gain a better intuition about when SAVI is useful and how it behaves, we simulate several scenarios that aim to illustrate the usefulness of SAVI inference for real-world use cases often encountered in tech companies.

The first set of simulations uses fixed-time and SAVI inference to measure false positives under daily stopping. For the second set of simulations, the **confidence sequences are the product**: the simulations illustrate how to use confidence sequences to qualify that a Java-to-Kotlin rewrite does no harm, to report correct inferences on weekly holdout results,[^1] and make hourly rollback decisions for a canary release.[^2] A final simulation studies regression adjustment with SAVI inference.

All simulations use `PyFixest`’s implementation of the linear-model method of [Lindon et al. (2026)](https://doi.org/10.1080/01621459.2026.2692052), which combines anytime-valid inference with regression adjustment for variance reduction.[^3]

``` python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pyfixest as pf
```

Show the simulation helpers

``` python
ALPHA = 0.05
N_DAYS = 20
USERS_PER_ARM_PER_DAY = 200
N_SIMULATIONS = 500
N_REPRESENTATIVE_PATHS = 100
PLANNED_MDE = 0.02
MAIN_TARGET_N = 2 * USERS_PER_ARM_PER_DAY * N_DAYS
MAIN_MIXTURE_PRECISION = pf.optimal_mixture_precision(
    nobs=MAIN_TARGET_N,
    number_of_coefficients=2,
    alpha=ALPHA,
)

SAVI_BLUE = "#2563EB"
ORDINARY_ORANGE = "#EA580C"
ESTIMATE_TEAL = "#0D9488"
NEUTRAL_GREY = "#64748B"


def _rgba(hex_color, opacity):
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {opacity})"


def _fit_at_day(data, with_intervals, mixture_precision):
    fit = pf.feols("converted ~ treated", data=data, vcov="hetero")
    row = {
        "estimate": float(fit.coef().loc["treated"]),
        "regular_pvalue": float(fit.pvalue().loc["treated"]),
        "savi_pvalue": float(
            fit.pvalue_savi(
                mixture_precision=mixture_precision,
            ).loc["treated"]
        ),
    }
    if with_intervals:
        ci = fit.confint(alpha=ALPHA).loc["treated"]
        cs = fit.confint(
            alpha=ALPHA,
            inference_type="savi",
            mixture_precision=mixture_precision,
        ).loc["treated"]
        row |= {
            "ci_lower": float(ci.iloc[0]),
            "ci_upper": float(ci.iloc[1]),
            "cs_lower": float(cs.iloc[0]),
            "cs_upper": float(cs.iloc[1]),
        }
    return row


def _experiment_path(
    p_control,
    p_treatment,
    seed,
    with_intervals=False,
    n_days=N_DAYS,
    users_per_day=USERS_PER_ARM_PER_DAY,
    mixture_precision=MAIN_MIXTURE_PRECISION,
):
    """One experiment: fit the model after each batch of users arrives."""
    rng = np.random.default_rng(seed)
    batches = []
    path = []
    for day in range(1, n_days + 1):
        treated = np.repeat([0, 1], users_per_day)
        converted = np.r_[
            rng.binomial(1, p_control, users_per_day),
            rng.binomial(1, p_treatment, users_per_day),
        ]
        order = rng.permutation(2 * users_per_day)
        batches.append(
            pd.DataFrame({"treated": treated[order], "converted": converted[order]})
        )
        row = {"day": day, "users": 2 * users_per_day * day}
        row |= _fit_at_day(
            pd.concat(batches, ignore_index=True),
            with_intervals,
            mixture_precision,
        )
        path.append(row)
    return pd.DataFrame(path)


def _simulate_paths(
    p_control,
    p_treatment,
    seed=42,
    n_simulations=N_SIMULATIONS,
    n_days=N_DAYS,
    mixture_precision=MAIN_MIXTURE_PRECISION,
):
    """Run many experiments; keep the full day-by-day estimate and p-value paths."""
    estimate = np.full((n_simulations, n_days), np.nan)
    regular_p = np.ones((n_simulations, n_days))
    savi_p = np.ones((n_simulations, n_days))
    for s in range(n_simulations):
        path = _experiment_path(
            p_control,
            p_treatment,
            seed + s,
            n_days=n_days,
            mixture_precision=mixture_precision,
        )
        estimate[s] = path["estimate"].to_numpy()
        regular_p[s] = path["regular_pvalue"].to_numpy()
        savi_p[s] = path["savi_pvalue"].to_numpy()
    return {"estimate": estimate, "regular_p": regular_p, "savi_p": savi_p}


def _stop_day(pmatrix):
    """First day (1-indexed) each row crosses ALPHA; 0 if it never does."""
    crossed = pmatrix <= ALPHA
    return np.where(crossed.any(axis=1), crossed.argmax(axis=1) + 1, 0)


def _representative_seed(stop_days, first_seed):
    """Choose a detected path whose stopping day is closest to the median."""
    detected = np.flatnonzero(stop_days > 0)
    median_day = np.median(stop_days[detected])
    index = detected[np.argmin(np.abs(stop_days[detected] - median_day))]
    return first_seed + int(index)


PLAY_PAUSE = [
    {
        "type": "buttons",
        "direction": "left",
        "x": 0,
        "y": -0.15,
        "xanchor": "left",
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 500, "redraw": True},
                        "transition": {"duration": 150},
                        "fromcurrent": False,
                    },
                ],
            },
            {
                "label": "❚❚ Pause",
                "method": "animate",
                "args": [
                    [None],
                    {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                ],
            },
        ],
    }
]


def _slider(days, active):
    first_day, last_day = days[0], days[-1]
    return [
        {
            "active": active,
            "currentvalue": {"prefix": "Day "},
            "pad": {"t": 50},
            "steps": [
                {
                    "label": (
                        str(day)
                        if day in {first_day, last_day} or day % 5 == 0
                        else ""
                    ),
                    "method": "animate",
                    "args": [
                        [str(day)],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                        },
                    ],
                }
                for day in days
            ],
        }
    ]
```

## Scenario 1: false positives from daily stopping

We start with a simulation that aims to study the false positive detection rate of different decision rules. To do so, we simulate 500 A/A experiments with a treatment effect of zero and applies three decision rules to each experiment:

1.  The simulation uses the fixed-time p-value for the prespecified decision on day twenty.
2.  The simulation recomputes the fixed-time p-value every day and stops the first time it is at or below 0.05.
3.  The simulation applies the same daily stopping rule with the SAVI p-value.

Each experiment runs for twenty days, with 200 new users per arm each day. Conversion is 10% in both arms, so the true treatment effect is zero and every rejection is a false positive.

The figure below shows the results of the first 100 experiments. Each grey line traces one lift estimate, in percentage points, that is updated each day with incoming data. Estimates vary most during the first few days but stabilize as more data accumulates.

The markers show the decisions made by the three rules. An orange cross marks the first day the fixed-time daily p-value is at or below 0.05. A blue cross marks the first day the SAVI p-value is at or below 0.05. An orange diamond marks a rejection by the fixed-time test on its prespecified analysis day, day twenty. Note that an experiment’s path can have more than one marker because all three policies are applied to every experiment.

Rejections occur when an estimate is large relative to its standard error. Stopping at the first fixed-time p-value below 0.05 therefore selects large early deviations, including paths that later move back toward zero. The SAVI p-value is calibrated for the whole sequence of daily checks rather than for any single one, which is why the figure contains far fewer blue crosses.

Show the simulation code

``` python
aa_paths = _simulate_paths(p_control=0.10, p_treatment=0.10, seed=42)

fpr_planned_analysis = (aa_paths["regular_p"][:, -1] <= ALPHA).mean()
fpr_daily_ordinary = (aa_paths["regular_p"] <= ALPHA).any(axis=1).mean()
fpr_daily_savi = (aa_paths["savi_p"] <= ALPHA).any(axis=1).mean()
```

Show the figure code

``` python
N_SHOWN = 100
EVIDENCE_BAR = -np.log10(ALPHA)
EVIDENCE_CAP = 4.0

show_est = aa_paths["estimate"][:N_SHOWN] * 100
reg_stop = _stop_day(aa_paths["regular_p"][:N_SHOWN])
savi_stop = _stop_day(aa_paths["savi_p"][:N_SHOWN])
final_reject = aa_paths["regular_p"][:N_SHOWN, -1] <= ALPHA

aa_days = np.arange(1, N_DAYS + 1)
fig = go.Figure()
for experiment in range(N_SHOWN):
    fig.add_trace(
        go.Scatter(
            x=aa_days,
            y=show_est[experiment],
            mode="lines",
            line={"color": _rgba(NEUTRAL_GREY, 0.3), "width": 1},
            name="One A/A path",
            showlegend=experiment == 0,
            hoverinfo="skip",
        )
    )

ordinary_daily_rejected = np.flatnonzero(reg_stop > 0)
savi_rejected = np.flatnonzero(savi_stop > 0)
planned_rejected = np.flatnonzero(final_reject)
fig.add_trace(
    go.Scatter(
        x=reg_stop[ordinary_daily_rejected],
        y=show_est[
            ordinary_daily_rejected,
            reg_stop[ordinary_daily_rejected] - 1,
        ],
        mode="markers",
        marker={"color": ORDINARY_ORANGE, "size": 9, "symbol": "x"},
        name=(
            f"Fixed-time daily stop "
            f"({ordinary_daily_rejected.size}/{N_SHOWN})"
        ),
        hovertemplate=(
            "Stopped on day %{x}<br>Reported effect: %{y:.1f} pp<extra></extra>"
        ),
    )
)
fig.add_trace(
    go.Scatter(
        x=savi_stop[savi_rejected],
        y=show_est[savi_rejected, savi_stop[savi_rejected] - 1],
        mode="markers",
        marker={"color": SAVI_BLUE, "size": 11, "symbol": "x"},
        name=f"SAVI stop ({savi_rejected.size}/{N_SHOWN})",
        hovertemplate=(
            "Stopped on day %{x}<br>Reported effect: %{y:.1f} pp<extra></extra>"
        ),
    )
)
fig.add_trace(
    go.Scatter(
        x=np.full(planned_rejected.size, N_DAYS),
        y=show_est[planned_rejected, -1],
        mode="markers",
        marker={
            "color": ORDINARY_ORANGE,
            "size": 10,
            "symbol": "diamond-open",
            "line": {"width": 2},
        },
        name=f"Planned day-20 rejection ({planned_rejected.size}/{N_SHOWN})",
        hovertemplate="Rejected on day 20<br>Estimate: %{y:.1f} pp<extra></extra>",
    )
)
fig.add_hline(y=0, line_color="black", opacity=0.4)
fig.add_vline(
    x=N_DAYS,
    line_dash="dash",
    line_color=NEUTRAL_GREY,
    opacity=0.6,
    annotation_text="Planned end",
    annotation_position="top left",
    annotation_font_color=NEUTRAL_GREY,
)
fig.update_layout(
    title=(
        "100 A/A experiments, three decision rules"
        "<br>True effect is zero: every marked stop is a false positive"
    ),
    title_font_size=16,
    template="plotly_white",
    height=500,
    xaxis_title="Day of experiment",
    yaxis_title="Estimated effect (pp)",
    legend={"orientation": "h", "y": -0.22},
    margin={"l": 65, "r": 30, "t": 95, "b": 120},
)
fig.update_xaxes(dtick=2, range=[0.5, N_DAYS + 0.5])
fig
```

Effect paths for 100 simulated A/A experiments. With a true effect of zero, every marked rejection is a false positive. Orange crosses show the first day the fixed-time p-value falls to 0.05 or below under the daily stopping rule. Orange diamonds show rejections from the fixed-sample analysis after 20 days, and blue crosses show the first SAVI rejection.

Across all 500 experiments:

Show the table code

``` python
pd.DataFrame(
    {
        "Decision rule": [
            "Fixed-time test, prespecified day-20 analysis",
            "Fixed-time p-value, stop at first p ≤ 0.05",
            "SAVI p-value, stop at first p ≤ 0.05",
        ],
        "False-positive rate": [
            f"{fpr_planned_analysis:.1%}",
            f"{fpr_daily_ordinary:.1%}",
            f"{fpr_daily_savi:.1%}",
        ],
    }
)
```

|     | Decision rule                                 | False-positive rate |
|-----|-----------------------------------------------|---------------------|
| 0   | Fixed-time test, prespecified day-20 analysis | 5.2%                |
| 1   | Fixed-time p-value, stop at first p ≤ 0.05    | 25.0%               |
| 2   | SAVI p-value, stop at first p ≤ 0.05          | 1.0%                |

The fixed-time analyses performed only on day twenty reject 5.2% of the time, close to the intended 5%. Checking the fixed-time p-value daily and stopping at the first p-value below 0.05 inflates the rate to 25%. This is below the 64% independence benchmark; the twenty analyses reuse the same accumulating data and their p-values are strongly dependent. It is nevertheless several times the specified false-positive rate of 5%. SAVI rejects 1.0% of the same null experiments. This rate does not need to equal 5%: under the method’s assumptions, the probability of a false rejection at any time is asymptotically at most 5%, so the probability of rejecting by day twenty is also at most 5% and can be lower.

The simulation also shows that discretionary early stopping further inflates the absolute effect reported at rejection. Estimates are noisiest in the first days, and conditioning on a first p-value below 0.05 selects unusually large estimates. For each experiment that rejects, the second plot records the estimate on its first rejection day; experiments that never reject are omitted.

Show the simulation code

``` python
ordinary_daily_stop = _stop_day(aa_paths["regular_p"])
savi_aa_stop = _stop_day(aa_paths["savi_p"])


def _estimate_at_stop(stop):
    idx = np.where(stop > 0)[0]
    return np.array([aa_paths["estimate"][i, stop[i] - 1] for i in idx]) * 100


ordinary_daily_at_stop = _estimate_at_stop(ordinary_daily_stop)
savi_at_stop = _estimate_at_stop(savi_aa_stop)

ordinary_daily_rejection_rate = (ordinary_daily_stop > 0).mean()
savi_rejection_rate = (savi_aa_stop > 0).mean()
ordinary_daily_mean_abs = np.abs(ordinary_daily_at_stop).mean()
```

Show the figure code

``` python
jitter = np.random.default_rng(1).uniform(-0.16, 0.16, size=N_SIMULATIONS)

fig = go.Figure()
for est, base, color, label in [
    (
        ordinary_daily_at_stop,
        1,
        ORDINARY_ORANGE,
        "Fixed-time p-value stopping",
    ),
    (savi_at_stop, 0, SAVI_BLUE, "SAVI"),
]:
    fig.add_trace(
        go.Scatter(
            x=est,
            y=base + jitter[: len(est)],
            mode="markers",
            marker={"color": _rgba(color, 0.5), "size": 7, "line": {"width": 0}},
            name=f"{label} ({len(est)}/{N_SIMULATIONS} rejected)",
            hovertemplate="Reported effect: %{x:.2f} pp<extra></extra>",
        )
    )
fig.add_vline(
    x=0,
    line_dash="dot",
    line_color=NEUTRAL_GREY,
    annotation_text="True effect: 0",
    annotation_position="top",
)
fig.update_layout(
    title="Estimates selected by each stopping rule<br>(true effect: 0)",
    template="plotly_white",
    height=380,
    xaxis_title="Estimated effect at stop (pp)",
    yaxis={
        "tickmode": "array",
        "tickvals": [0, 1],
        "ticktext": ["SAVI", "Fixed-time p-value<br>stopping"],
        "range": [-0.5, 1.5],
    },
    legend={"orientation": "h", "y": -0.3},
    margin={"l": 80, "r": 30, "t": 90, "b": 105},
)
fig
```

Estimated effects at the first rejection, among experiments that reject within twenty days. Non-rejecting experiments are omitted.

The daily fixed-time p-value stopping rule rejects in 25% of the simulated experiments with null effects. Among the experiments that reject, the estimated effect at the stopping time averages 3.9 pp in absolute value even though the true effect is zero. These large magnitudes arise from selection at the stopping time.

The SAVI rule rejects 1.0% of these null experiments because its threshold is calibrated across all daily checks. Conditioning on a SAVI rejection still selects unusually large absolute point estimates.

## Scenario 2: effect size and stopping time

The twenty-day design targets 80% power for a lift from 10% to 12%. Note that this 2 pp lift is the MDE used to choose the sample size, not a threshold for shipping the feature. We first simulate a treatment conversion rate of 14%, twice the planning MDE, and compare the SAVI stopping day with the prespecified day-twenty decision. In this design, the larger lift often reaches the SAVI boundary before day twenty.

The top panel in the figure below tracks the estimated lift together with a fixed-time 95% confidence interval and a 95% SAVI confidence sequence; while the fixed-time interval has pointwise coverage at a prespecified analysis time, the confidence sequence has time-uniform asymptotic coverage. The bottom panel shows the fixed-time and SAVI p-values on a \\-\log\_{10}(p)\\ scale, since the raw p-values soon become too small to plot next to the 0.05 line; the horizontal line corresponds to \\p=0.05\\.

Show the simulation code

``` python
TRUE_LIFT = 2 * PLANNED_MDE
LARGE_EFFECT_FIRST_SEED = 20_000

large_effect_candidates = _simulate_paths(
    p_control=0.10,
    p_treatment=0.10 + TRUE_LIFT,
    seed=LARGE_EFFECT_FIRST_SEED,
    n_simulations=N_REPRESENTATIVE_PATHS,
)
large_effect_positive_p = np.where(
    large_effect_candidates["estimate"] > 0,
    large_effect_candidates["savi_p"],
    1.0,
)
large_effect_stop_days = _stop_day(large_effect_positive_p)
representative_effect_seed = _representative_seed(
    large_effect_stop_days,
    LARGE_EFFECT_FIRST_SEED,
)

representative_effect_path = _experiment_path(
    p_control=0.10,
    p_treatment=0.10 + TRUE_LIFT,
    seed=representative_effect_seed,
    with_intervals=True,
).set_index("day", drop=False)

ordinary_positive_rejection = (
    representative_effect_path["regular_pvalue"] <= ALPHA
) & (representative_effect_path["estimate"] > 0)
savi_positive_rejection = (
    representative_effect_path["savi_pvalue"] <= ALPHA
) & (representative_effect_path["estimate"] > 0)
peek_day = int(
    representative_effect_path.loc[ordinary_positive_rejection, "day"].iloc[0]
)
savi_day = int(
    representative_effect_path.loc[savi_positive_rejection, "day"].iloc[0]
)
savi_stop_row = representative_effect_path.loc[savi_day]
days_saved = N_DAYS - savi_day

peek_lift = f"{representative_effect_path.loc[peek_day, 'estimate']:.1%}"
savi_users = f"{int(savi_stop_row['users']):,}"
savi_cs = (
    f"[{savi_stop_row['cs_lower']:.1%}, "
    f"{savi_stop_row['cs_upper']:.1%}]"
)
```

Show the animation code

``` python
effect_path_days = list(range(1, N_DAYS + 1))
displayed_effect_path = representative_effect_path.loc[effect_path_days]
estimate_range = [
    float(displayed_effect_path["cs_lower"].min()) - 0.01,
    float(displayed_effect_path["cs_upper"].max()) + 0.01,
]


def _effect_path_traces(day):
    seen = representative_effect_path.loc[
        representative_effect_path["day"] <= day
    ]
    current = representative_effect_path.loc[day]
    band = {"mode": "lines", "line": {"width": 0}, "hoverinfo": "skip"}
    ordinary_crossed = day >= peek_day
    savi_crossed = day >= savi_day
    return [
        go.Scatter(x=seen["day"], y=seen["cs_upper"], showlegend=False, **band),
        go.Scatter(
            x=seen["day"],
            y=seen["cs_lower"],
            fill="tonexty",
            fillcolor=_rgba(SAVI_BLUE, 0.15),
            name="95% SAVI confidence sequence",
            **band,
        ),
        go.Scatter(x=seen["day"], y=seen["ci_upper"], showlegend=False, **band),
        go.Scatter(
            x=seen["day"],
            y=seen["ci_lower"],
            fill="tonexty",
            fillcolor=_rgba(ORDINARY_ORANGE, 0.18),
            name="95% fixed-time confidence interval",
            **band,
        ),
        go.Scatter(
            x=[day, day],
            y=[current["cs_lower"], current["cs_upper"]],
            mode="lines+markers",
            line={"color": SAVI_BLUE, "width": 3},
            marker={"color": SAVI_BLUE, "size": 5},
            showlegend=False,
            hovertemplate=(
                f"Day {day}<br>SAVI confidence sequence: "
                f"[{current['cs_lower']:.1%}, {current['cs_upper']:.1%}]"
                "<extra></extra>"
            ),
        ),
        go.Scatter(
            x=[day, day],
            y=[current["ci_lower"], current["ci_upper"]],
            mode="lines+markers",
            line={"color": ORDINARY_ORANGE, "width": 2},
            marker={"color": ORDINARY_ORANGE, "size": 4},
            showlegend=False,
            hovertemplate=(
                f"Day {day}<br>Fixed-time interval: "
                f"[{current['ci_lower']:.1%}, {current['ci_upper']:.1%}]"
                "<extra></extra>"
            ),
        ),
        go.Scatter(
            x=seen["day"],
            y=seen["estimate"],
            mode="lines+markers",
            line_color=ESTIMATE_TEAL,
            name="Estimated lift",
            hovertemplate="Day %{x}<br>Lift: %{y:.1%}<extra></extra>",
        ),
        go.Scatter(
            x=seen["day"],
            y=np.minimum(-np.log10(seen["regular_pvalue"]), EVIDENCE_CAP),
            mode="lines+markers",
            line={"color": ORDINARY_ORANGE, "dash": "dot"},
            name="Fixed-time p-value",
            customdata=seen["regular_pvalue"],
            hovertemplate="Day %{x}<br>p = %{customdata:.4f}<extra></extra>",
        ),
        go.Scatter(
            x=seen["day"],
            y=np.minimum(-np.log10(seen["savi_pvalue"]), EVIDENCE_CAP),
            mode="lines+markers",
            line_color=SAVI_BLUE,
            name="SAVI p-value",
            customdata=seen["savi_pvalue"],
            hovertemplate="Day %{x}<br>p = %{customdata:.4f}<extra></extra>",
        ),
        go.Scatter(
            x=[peek_day] if ordinary_crossed else [None],
            y=[
                min(
                    -np.log10(
                        representative_effect_path.loc[
                            peek_day,
                            "regular_pvalue",
                        ]
                    ),
                    EVIDENCE_CAP,
                )
            ]
            if ordinary_crossed
            else [None],
            mode="markers",
            marker={"color": ORDINARY_ORANGE, "size": 10, "symbol": "diamond"},
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[savi_day] if savi_crossed else [None],
            y=[
                min(
                    -np.log10(
                        representative_effect_path.loc[
                            savi_day,
                            "savi_pvalue",
                        ]
                    ),
                    EVIDENCE_CAP,
                )
            ]
            if savi_crossed
            else [None],
            mode="markers",
            marker={"color": SAVI_BLUE, "size": 10, "symbol": "diamond"},
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[peek_day, peek_day] if ordinary_crossed else [None, None],
            y=[0, EVIDENCE_CAP],
            mode="lines",
            line={"color": ORDINARY_ORANGE, "dash": "dot", "width": 1.5},
            showlegend=False,
            hovertemplate=(
                f"Day {peek_day}: fixed-time p-value below 0.05;"
                " daily first-crossing rule is not calibrated at 5%"
                "<extra></extra>"
            ),
        ),
        go.Scatter(
            x=[savi_day, savi_day] if savi_crossed else [None, None],
            y=[0, EVIDENCE_CAP],
            mode="lines",
            line={"color": SAVI_BLUE, "dash": "dot", "width": 1.5},
            showlegend=False,
            hovertemplate=(
                f"Day {savi_day}: SAVI p-value below 0.05;"
                " reject the null at the anytime-valid 5% level"
                "<extra></extra>"
            ),
        ),
    ]


def _effect_path_title(day):
    users = int(representative_effect_path.loc[day, "users"])
    if day < peek_day:
        decision = "Continue"
    elif day < savi_day:
        decision = "Fixed-time p < 0.05; daily rule is not calibrated at 5%"
    else:
        decision = "SAVI boundary reached; reject at the anytime-valid 5% level"
    return f"Day {day} of {N_DAYS} · {users:,} users<br>{decision}"


fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.12,
    subplot_titles=("Estimated lift", "Evidence against 'no effect'"),
)
for i, trace in enumerate(_effect_path_traces(effect_path_days[0])):
    fig.add_trace(trace, row=1 if i < 7 else 2, col=1)
fig.frames = [
    go.Frame(
        data=_effect_path_traces(day),
        traces=list(range(13)),
        name=str(day),
        layout={"title": {"text": _effect_path_title(day)}},
    )
    for day in effect_path_days
]
fig.add_hline(
    y=TRUE_LIFT,
    line_dash="dot",
    line_color=NEUTRAL_GREY,
    annotation_text=f"True lift: {TRUE_LIFT:.0%}",
    row=1,
    col=1,
)
fig.add_hline(
    y=EVIDENCE_BAR,
    line_dash="dash",
    line_color="black",
    annotation_text="Decision bar (p = 0.05)",
    row=2,
    col=1,
)
fig.add_vline(
    x=N_DAYS,
    line_dash="dash",
    line_color=NEUTRAL_GREY,
    opacity=0.7,
    annotation_text="Prespecified fixed-time analysis: day 20",
    annotation_position="top left",
    annotation_font_color=NEUTRAL_GREY,
    row=2,
    col=1,
)
fig.update_layout(
    title=_effect_path_title(effect_path_days[0]),
    title_font_size=16,
    template="plotly_white",
    height=740,
    xaxis_range=[0.5, N_DAYS + 0.5],
    xaxis2_range=[0.5, N_DAYS + 0.5],
    yaxis_range=estimate_range,
    yaxis2_range=[0, EVIDENCE_CAP],
    yaxis_tickformat=".0%",
    yaxis2_title="−log10(p-value), capped at 4",
    xaxis2_title="Day of experiment",
    legend={"orientation": "h", "y": -0.48},
    updatemenus=PLAY_PAUSE,
    sliders=_slider(effect_path_days, active=0),
    margin={"l": 70, "r": 35, "t": 125, "b": 270},
)
fig.update_xaxes(dtick=2)
fig
```

One simulated experiment with a true lift of 4 pp and a planning MDE of 2 pp. The orange and blue vertical lines mark the first fixed-time and SAVI p-values at or below 0.05. The evidence axis is capped at 4 so that the \\p = 0.05\\ line remains visible.

The fixed-time p-value falls below 0.05 on day 3, when the estimated lift is 4.5%. But stopping at this point in time uses the event \\\min\_{t \leq 20} p_t \leq 0.05\\, for which the fixed-time p-values do not provide 5% type-I error control. Using fixed-time inference, you should not stop here, but only after 20 days.

With SAVI, the p-value falls below 0.05 on day 7, after 2,800 users. The shipping rule in this example requires the SAVI confidence sequence to be positive and exclude zero. Because the lower end of the confidence sequence excludes zero before day twenty, the SAVI decision rule ships 13 days early. The confidence sequence at that day is \[0.3%, 8.0%\]. This establishes a positive effect, not an effect of at least 2 pp. If 2 pp were the minimum worthwhile lift rather than the planning MDE, shipping would require the lower endpoint to exceed 2 pp.

Earlier stopping is specific to the design and effect size. To show how the stopping-time distribution changes by the true effect size, we simulate true lifts of 1, 2, and 4 percentage points for forty days while keeping the planning MDE at 2 pp and runtime at 20 days.

Show the simulation code

``` python
MDE_MONITORING_DAYS = 40

mde_effect_paths = _simulate_paths(
    p_control=0.10,
    p_treatment=0.10 + PLANNED_MDE,
    seed=10_000,
    n_days=MDE_MONITORING_DAYS,
)
smaller_effect_paths = _simulate_paths(
    p_control=0.10,
    p_treatment=0.11,
    seed=11_000,
    n_days=MDE_MONITORING_DAYS,
)
larger_effect_paths = _simulate_paths(
    p_control=0.10,
    p_treatment=0.14,
    seed=12_000,
    n_days=MDE_MONITORING_DAYS,
)

savi_detected = np.maximum.accumulate(
    (mde_effect_paths["savi_p"] <= ALPHA)
    & (mde_effect_paths["estimate"] > 0),
    axis=1,
)
savi_detection = savi_detected.mean(axis=0)
smaller_savi_detection = np.maximum.accumulate(
    (smaller_effect_paths["savi_p"] <= ALPHA)
    & (smaller_effect_paths["estimate"] > 0),
    axis=1,
).mean(axis=0)
larger_savi_detection = np.maximum.accumulate(
    (larger_effect_paths["savi_p"] <= ALPHA)
    & (larger_effect_paths["estimate"] > 0),
    axis=1,
).mean(axis=0)
oneshot_power = (
    (mde_effect_paths["regular_p"][:, N_DAYS - 1] <= ALPHA)
    & (mde_effect_paths["estimate"][:, N_DAYS - 1] > 0)
).mean()
```

Show the figure code

``` python
days = np.arange(1, MDE_MONITORING_DAYS + 1)
oneshot_curve = np.where(days >= N_DAYS, oneshot_power, 0.0)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=days,
        y=oneshot_curve,
        mode="lines+markers",
        line={"shape": "hv", "dash": "dash", "color": ORDINARY_ORANGE},
        name="Fixed-time test: 2 pp, day 20",
        hovertemplate="Day %{x}<br>Detected: %{y:.0%}<extra></extra>",
    )
)
for detection, dash, marker, label in [
    (smaller_savi_detection, "dot", "circle-open", "SAVI: 1 pp lift"),
    (savi_detection, "dash", "diamond-open", "SAVI: 2 pp lift (MDE)"),
    (larger_savi_detection, "solid", "circle", "SAVI: 4 pp lift"),
]:
    fig.add_trace(
        go.Scatter(
            x=days,
            y=detection,
            mode="lines+markers",
            line={"color": SAVI_BLUE, "dash": dash},
            marker={"symbol": marker},
            name=label,
            hovertemplate=(
                "Day %{x}<br>Positive SAVI rejection by this day: "
                "%{y:.0%}<extra></extra>"
            ),
        )
    )
fig.add_vline(
    x=N_DAYS,
    line_dash="dash",
    line_color=NEUTRAL_GREY,
    opacity=0.7,
    annotation_text="Planned end: day 20",
    annotation_position="bottom right",
    annotation_font_color=NEUTRAL_GREY,
)
fig.update_layout(
    title="SAVI stopping time by true lift",
    xaxis_title="Day of experiment",
    yaxis_title="Share with a positive SAVI rejection",
    yaxis_tickformat=".0%",
    yaxis_range=[0, 1],
    hovermode="x unified",
    template="plotly_white",
    legend={"orientation": "h", "y": -0.3},
    height=500,
    margin={"l": 75, "r": 30, "t": 90, "b": 145},
)
fig.update_xaxes(dtick=5, range=[1, MDE_MONITORING_DAYS])
fig
```

Share of SAVI experiments with a positive rejection by each day, for true lifts of 1, 2, and 4 pp. The fixed-time curve reports power for a true 2 pp lift at its prespecified decision time, day twenty.

At day twenty, the fixed-time test detects the effect in 80% of experiments when the true lift is the planning MDE of 2 pp. By then, SAVI has produced a positive rejection of the hypothesis of “no effect” in 99% of experiments with a 4 pp lift, 49% with a 2 pp lift, and 8% with a 1 pp lift. The same prespecified SAVI rule remains calibrated when monitoring continues past day twenty. By day forty, those shares are 100%, 86%, and 23%, respectively. In this simulated design, SAVI has lower power than the fixed-time test at day twenty when the true lift equals the planning MDE of 2 pp. With a true 4 pp lift in the same design, SAVI often reaches its rejection boundary before day twenty. This is a stopping-time advantage, not a power advantage over the fixed-time test at the same effect and horizon.

## Scenario 3: From Java to Kotlin - certify a backend migration

Your backend team has rewritten a backend service from Java to Kotlin.[^4] Before the test is launched, the team defines the migration as acceptable when the true effect on conversion lies strictly between -1 and +1 pp. Certification therefore depends on the whole interval, not only the point estimate.

Your team checks the interval once per day and agrees to accept the new Kotlin backend as soon as the whole interval is within the acceptance range of ±1 pp. Once again, your team is faced with a discretionary decision rule: “peek every day and stop if the interval is inside the range.” As before, it holds that the usual 95% fixed-time confidence interval is only valid for one prespecified analysis day; it is not calibrated for checking every day and stopping when the interval first falls inside the range. A SAVI confidence sequence is instead designed for such a workflow: it provides simultaneous asymptotic coverage across all daily reviews.

The Kotlin backend works just as smooth as Java, and the simulation once again assumes a true treatment effect of 0 and sends 2,000 users per arm per day to the Java and Kotlin implementations.

Show the animation code

``` python
DELTA = 0.01
BACKEND_USERS = 2_000
BACKEND_MAX_DAYS = 30
BACKEND_FIRST_SEED = 50_000
BACKEND_MIXTURE_PRECISION = pf.optimal_mixture_precision(
    nobs=2 * BACKEND_USERS * BACKEND_MAX_DAYS,
    number_of_coefficients=2,
    alpha=ALPHA,
)

backend_cert_days = np.zeros(N_REPRESENTATIVE_PATHS, dtype=int)
backend_ci_cert_days = np.zeros(N_REPRESENTATIVE_PATHS, dtype=int)
for simulation in range(N_REPRESENTATIVE_PATHS):
    candidate = _experiment_path(
        p_control=0.10,
        p_treatment=0.10,
        seed=BACKEND_FIRST_SEED + simulation,
        with_intervals=True,
        n_days=BACKEND_MAX_DAYS,
        users_per_day=BACKEND_USERS,
        mixture_precision=BACKEND_MIXTURE_PRECISION,
    )
    candidate_certified = (
        (candidate["cs_lower"] > -DELTA)
        & (candidate["cs_upper"] < DELTA)
    )
    if candidate_certified.any():
        backend_cert_days[simulation] = int(
            candidate.loc[candidate_certified, "day"].iloc[0]
        )
    candidate_ci_certified = (
        (candidate["ci_lower"] > -DELTA)
        & (candidate["ci_upper"] < DELTA)
    )
    if candidate_ci_certified.any():
        backend_ci_cert_days[simulation] = int(
            candidate.loc[candidate_ci_certified, "day"].iloc[0]
        )

median_cs_cert = int(np.median(backend_cert_days[backend_cert_days > 0]))
median_ci_cert = int(np.median(backend_ci_cert_days[backend_ci_cert_days > 0]))
cs_cert_rate = (backend_cert_days > 0).mean()
ci_cert_rate = (backend_ci_cert_days > 0).mean()

backend_seed = _representative_seed(
    backend_cert_days,
    BACKEND_FIRST_SEED,
)

backend_full = _experiment_path(
    p_control=0.10,
    p_treatment=0.10,
    seed=backend_seed,
    with_intervals=True,
    n_days=BACKEND_MAX_DAYS,
    users_per_day=BACKEND_USERS,
    mixture_precision=BACKEND_MIXTURE_PRECISION,
).set_index("day", drop=False)

certified = (backend_full["cs_lower"] > -DELTA) & (backend_full["cs_upper"] < DELTA)
cert_day = int(backend_full.loc[certified, "day"].iloc[0])
cert_cs = (
    f"[{backend_full.loc[cert_day, 'cs_lower']:.2%}, "
    f"{backend_full.loc[cert_day, 'cs_upper']:.2%}]"
)
ci_certified = (backend_full["ci_lower"] > -DELTA) & (backend_full["ci_upper"] < DELTA)
ci_cert_day = int(backend_full.loc[ci_certified, "day"].iloc[0])

backend_path = backend_full.loc[backend_full["day"] <= cert_day]
b_range = [
    float(backend_path["cs_lower"].min()) - 0.005,
    float(backend_path["cs_upper"].max()) + 0.005,
]


def _backend_traces(day):
    seen = backend_path.loc[backend_path["day"] <= day]
    current = backend_path.loc[day]
    band = {"mode": "lines", "line": {"width": 0}, "hoverinfo": "skip"}
    tolerance_met = day >= cert_day
    return [
        go.Scatter(x=seen["day"], y=seen["cs_upper"], showlegend=False, **band),
        go.Scatter(
            x=seen["day"],
            y=seen["cs_lower"],
            fill="tonexty",
            fillcolor=_rgba(SAVI_BLUE, 0.15),
            name="95% SAVI confidence sequence",
            **band,
        ),
        go.Scatter(x=seen["day"], y=seen["ci_upper"], showlegend=False, **band),
        go.Scatter(
            x=seen["day"],
            y=seen["ci_lower"],
            fill="tonexty",
            fillcolor=_rgba(ORDINARY_ORANGE, 0.18),
            name="95% fixed-time confidence interval",
            **band,
        ),
        go.Scatter(
            x=[day, day],
            y=[current["cs_lower"], current["cs_upper"]],
            mode="lines+markers",
            line={"color": SAVI_BLUE, "width": 3},
            marker={"color": SAVI_BLUE, "size": 5},
            showlegend=False,
            hovertemplate=(
                f"Day {day}<br>SAVI confidence sequence: "
                f"[{current['cs_lower']:.2%}, {current['cs_upper']:.2%}]"
                "<extra></extra>"
            ),
        ),
        go.Scatter(
            x=[day, day],
            y=[current["ci_lower"], current["ci_upper"]],
            mode="lines+markers",
            line={"color": ORDINARY_ORANGE, "width": 2},
            marker={"color": ORDINARY_ORANGE, "size": 4},
            showlegend=False,
            hovertemplate=(
                f"Day {day}<br>Fixed-time interval: "
                f"[{current['ci_lower']:.2%}, {current['ci_upper']:.2%}]"
                "<extra></extra>"
            ),
        ),
        go.Scatter(
            x=seen["day"],
            y=seen["estimate"],
            mode="lines+markers",
            line_color=ESTIMATE_TEAL,
            name="Estimated effect",
            hovertemplate="Day %{x}<br>Effect: %{y:.1%}<extra></extra>",
        ),
        go.Scatter(
            x=[cert_day] if tolerance_met else [None],
            y=[backend_path.loc[cert_day, "estimate"]] if tolerance_met else [None],
            mode="markers+text",
            marker={"color": SAVI_BLUE, "size": 11, "symbol": "diamond"},
            text=["Tolerance met"] if tolerance_met else [None],
            textposition="bottom left",
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[ci_cert_day, ci_cert_day] if day >= ci_cert_day else [None, None],
            y=b_range,
            mode="lines",
            line={"color": ORDINARY_ORANGE, "dash": "dot", "width": 2},
            showlegend=False,
            hovertemplate=(
                f"Day {ci_cert_day}: fixed-time interval first enters range;"
                " first-entry rule lacks time-uniform 95% coverage"
                "<extra></extra>"
            ),
        ),
        go.Scatter(
            x=[cert_day, cert_day] if tolerance_met else [None, None],
            y=b_range,
            mode="lines",
            line={"color": SAVI_BLUE, "dash": "dot", "width": 2},
            showlegend=False,
            hovertemplate=(
                f"Day {cert_day}: SAVI confidence sequence first enters range;"
                " SAVI certification rule triggered<extra></extra>"
            ),
        ),
    ]


def _backend_title(day):
    users = int(backend_path.loc[day, "users"])
    if day == cert_day:
        decision = "SAVI confidence sequence inside ±1 pp: certification rule met"
    elif day >= ci_cert_day:
        decision = "Fixed-time interval inside tolerance; continue"
    else:
        decision = "Tolerance not met; keep collecting"
    return f"Day {day} · {users:,} users<br>{decision}"


backend_days = list(range(1, cert_day + 1))
fig = go.Figure(_backend_traces(1))
fig.frames = [
    go.Frame(
        data=_backend_traces(day),
        name=str(day),
        layout={"title": {"text": _backend_title(day)}},
    )
    for day in backend_days
]
fig.add_hrect(
    y0=-DELTA,
    y1=DELTA,
    fillcolor=_rgba(SAVI_BLUE, 0.07),
    line_width=0,
)
fig.add_hline(
    y=DELTA,
    line_dash="dash",
    line_color=NEUTRAL_GREY,
    opacity=0.7,
    annotation_text="Certification tube: ±1 pp",
    annotation_position="top right",
    annotation_font_color=NEUTRAL_GREY,
)
fig.add_hline(y=-DELTA, line_dash="dash", line_color=NEUTRAL_GREY, opacity=0.7)
fig.add_hline(y=0, line_color="black", opacity=0.4)
fig.update_layout(
    title=_backend_title(1),
    title_font_size=16,
    template="plotly_white",
    height=600,
    xaxis_title="Day of A/A test",
    xaxis_range=[0.5, cert_day + 0.5],
    yaxis_title="Estimated effect",
    yaxis_range=b_range,
    yaxis_tickformat=".0%",
    legend={"orientation": "h", "y": -0.48},
    updatemenus=PLAY_PAUSE,
    sliders=_slider(backend_days, active=0),
    margin={"l": 70, "r": 35, "t": 105, "b": 245},
)
fig.update_xaxes(dtick=2)
fig
```

The blue vertical line marks the first day the SAVI confidence sequence lies entirely inside the -1 to +1 pp acceptance range. The orange line marks the first day the fixed-time confidence interval does the same.

On day 13, the SAVI confidence sequence \[-0.96%, 0.68%\] lies inside the ±1 pp tolerance. Its time-uniform asymptotic coverage allows your team to stop and rule out effects larger than the tolerance.

For the selected path, the fixed-time interval enters the tolerance on day 5, earlier than the SAVI confidence sequence on day 13. A fixed-time interval provides pointwise 95% coverage when the analysis time is fixed independently of the results. Here, however, the your team selects the first day on which the interval fits inside the acceptance range. Now comes the same old story - repeated checks make it more likely that sampling variation produces a favorable interval on at least one day. As a result, the discretionary stopping rule does not have a 95% simultaneous coverage guarantee.

## Scenario 4: monitor a long-term holdout

In many tech companies, it is not uncommon to keep a holdout set (see e.g. [here](https://www.growthbook.io/blog/what-does-a-holdout-measure) and [here](https://medium.com/disney-streaming/universal-holdout-groups-at-disney-streaming-2043360def4f)). Often, a holdout is used for weekly reporting rather than for ship/no ship decisions via a stopping rule. For example, a product area may [keep a small randomized group of users from receiving new features](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4877025) for an entire year. Comparing the holdout with exposed users estimates the effect of assignment to the year-long release policy. The reporting target then is simultaneous coverage across all 52 weekly business reviews.

Each fixed-time confidence interval has 95% pointwise coverage at a fixed review time, but the collection of weekly intervals does not have 95% simultaneous coverage. As the number of reviews grows, so does the probability that at least one interval misses the true effect. A 95% SAVI confidence sequence is constructed to contain the true effect at every review with asymptotic probability of at least 95%, under the method’s assumptions and regardless of the monitoring schedule.

Below, we simulate such a holdout and assume that the shipped features lift conversion from 10% to 11%. Each week adds 500 users per arm, and our team reviews the result weekly for a year.

Show the simulation code

``` python
HOLDOUT_WEEKS = 52
HOLDOUT_USERS_PER_WEEK = 500
HOLDOUT_LIFT = 0.01
HOLDOUT_SIMULATIONS = 100
HOLDOUT_FIRST_SEED = 60_000
HOLDOUT_MIXTURE_PRECISION = pf.optimal_mixture_precision(
    nobs=2 * HOLDOUT_USERS_PER_WEEK * HOLDOUT_WEEKS,
    number_of_coefficients=2,
    alpha=ALPHA,
)

holdout_bounds = {
    key: np.empty((HOLDOUT_SIMULATIONS, HOLDOUT_WEEKS))
    for key in ("estimate", "ci_lower", "ci_upper", "cs_lower", "cs_upper")
}
for simulation in range(HOLDOUT_SIMULATIONS):
    path = _experiment_path(
        p_control=0.10,
        p_treatment=0.10 + HOLDOUT_LIFT,
        seed=HOLDOUT_FIRST_SEED + simulation,
        with_intervals=True,
        n_days=HOLDOUT_WEEKS,
        users_per_day=HOLDOUT_USERS_PER_WEEK,
        mixture_precision=HOLDOUT_MIXTURE_PRECISION,
    )
    for key in holdout_bounds:
        holdout_bounds[key][simulation] = path[key].to_numpy()

ci_covers = (holdout_bounds["ci_lower"] <= HOLDOUT_LIFT) & (
    holdout_bounds["ci_upper"] >= HOLDOUT_LIFT
)
cs_covers = (holdout_bounds["cs_lower"] <= HOLDOUT_LIFT) & (
    holdout_bounds["cs_upper"] >= HOLDOUT_LIFT
)
ci_always_covered = np.logical_and.accumulate(ci_covers, axis=1).mean(axis=0)
cs_always_covered = np.logical_and.accumulate(cs_covers, axis=1).mean(axis=0)
ci_ever_missed = 1 - ci_always_covered
cs_ever_missed = 1 - cs_always_covered

holdout_weeks_axis = np.arange(1, HOLDOUT_WEEKS + 1)
holdout_final_estimates = holdout_bounds["estimate"][:, -1]
holdout_index = int(
    np.argmin(
        np.abs(holdout_final_estimates - np.median(holdout_final_estimates))
    )
)
holdout_path = pd.DataFrame(
    {key: values[holdout_index] for key, values in holdout_bounds.items()}
).assign(week=holdout_weeks_axis)
holdout_final = holdout_path.iloc[-1]
holdout_cs = f"[{holdout_final['cs_lower']:.2%}, {holdout_final['cs_upper']:.2%}]"
holdout_ci = f"[{holdout_final['ci_lower']:.2%}, {holdout_final['ci_upper']:.2%}]"
```

Show the figure code

``` python
band = {"mode": "lines", "line": {"width": 0}, "hoverinfo": "skip"}
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=holdout_path["week"],
        y=holdout_path["cs_upper"],
        showlegend=False,
        **band,
    )
)
fig.add_trace(
    go.Scatter(
        x=holdout_path["week"],
        y=holdout_path["cs_lower"],
        fill="tonexty",
        fillcolor=_rgba(SAVI_BLUE, 0.15),
        name="95% SAVI confidence sequence",
        **band,
    )
)
fig.add_trace(
    go.Scatter(
        x=holdout_path["week"],
        y=holdout_path["ci_upper"],
        showlegend=False,
        **band,
    )
)
fig.add_trace(
    go.Scatter(
        x=holdout_path["week"],
        y=holdout_path["ci_lower"],
        fill="tonexty",
        fillcolor=_rgba(ORDINARY_ORANGE, 0.18),
        name="95% fixed-time confidence interval",
        **band,
    )
)
fig.add_trace(
    go.Scatter(
        x=holdout_path["week"],
        y=holdout_path["estimate"],
        mode="lines+markers",
        line_color=ESTIMATE_TEAL,
        name="Estimated lift",
        hovertemplate="Week %{x}<br>Lift: %{y:.2%}<extra></extra>",
    )
)
fig.add_hline(y=0, line_color="black", opacity=0.4)
fig.add_hline(
    y=HOLDOUT_LIFT,
    line_dash="dot",
    line_color=NEUTRAL_GREY,
    annotation_text="True lift: 1 pp",
    annotation_position="top right",
)
fig.add_annotation(
    x=3,
    y=0.029,
    text="Early intervals extend beyond this range",
    showarrow=False,
    xanchor="left",
    font={"color": NEUTRAL_GREY},
)
fig.update_layout(
    title="A long-term holdout, reviewed weekly for one year",
    template="plotly_white",
    height=430,
    xaxis_title="Week of holdout",
    yaxis_title="Estimated lift",
    yaxis_tickformat=".1%",
    yaxis_range=[-0.02, 0.032],
    legend={"orientation": "h", "y": -0.3},
    margin={"l": 70, "r": 35, "t": 70, "b": 120},
)
fig
```

One simulated holdout, reviewed weekly for a year; the y-axis clips the very wide intervals of the first weeks. The SAVI confidence sequence is constructed for simultaneous coverage across all reviews, while the fixed-time interval is calibrated for one prespecified review.

At week 52, the displayed path has a SAVI confidence sequence of \[0.18%, 1.82%\] and a fixed-time interval of \[0.47%, 1.53%\]. To estimate simultaneous coverage, the analysis uses all 100 simulated holdouts and records whether each interval has contained the true lift at every review so far:

Show the figure code

``` python
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=holdout_weeks_axis,
        y=ci_ever_missed,
        mode="lines+markers",
        line={"color": ORDINARY_ORANGE, "dash": "dot"},
        name="Fixed-time 95% confidence interval",
        hovertemplate="Week %{x}<br>Missed at least once: %{y:.0%}<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=holdout_weeks_axis,
        y=cs_ever_missed,
        mode="lines+markers",
        line_color=SAVI_BLUE,
        name="95% SAVI confidence sequence",
        hovertemplate="Week %{x}<br>Missed at least once: %{y:.0%}<extra></extra>",
    )
)
fig.add_hline(
    y=ALPHA,
    line_dash="dash",
    line_color="black",
    annotation_text="5% simultaneous miss-rate boundary",
    annotation_position="top right",
)
fig.update_layout(
    title="Cumulative interval miss rate across weekly reviews",
    template="plotly_white",
    height=430,
    xaxis_title="Week of holdout",
    yaxis_title="Missed at least once",
    yaxis_tickformat=".0%",
    yaxis_range=[0, 0.35],
    hovermode="x unified",
    legend={"orientation": "h", "y": -0.3},
    margin={"l": 75, "r": 35, "t": 70, "b": 120},
)
fig
```

The figure plots the share of simulated holdouts whose interval has missed the true lift at least once by each weekly review.

By week 52, the fixed-time interval has missed the true lift at least once in 29% of the simulated holdouts. The SAVI confidence sequence did not miss in any of these 100 runs.

## Scenario 5: decide whether to roll back a canary deployment

To steer the risk of shipping new features and thereby breaking production, [canary deployments](https://docs.cloud.google.com/deploy/docs/deployment-strategies/canary) are a very common practice in software engineering.

A canary sends a small share of traffic to a new service version before a full rollout. If the version reduces conversion, hourly monitoring can limit the number of users exposed before rollback. In this simulation, the planned canary horizon is 48 hours. A fixed-time confidence interval provides pointwise coverage for one prespecified look at the end of that horizon.

The team does not want to wait 48 hours to roll back a harmful release. It reviews results hourly, so it needs SAVI confidence sequences that remain valid when any of those reviews can trigger a decision.

Before the launch, the team agrees that it will tolerate a conversion loss of no more than 1 pp, so -1 pp is the decision boundary. With 500 new users per arm each hour, the team decides to:

- **Roll back** if the upper endpoint is below -1 pp; every effect compatible with the data implies a loss greater than 1 pp.
- **Promote** if the lower endpoint is above -1 pp; a loss of 1 pp or more has been ruled out.
- **Continue** while the SAVI confidence sequence contains -1 pp.

In the simulation, we artificially reduce conversion from 10% to 7%, a true effect of -3 pp which the team would optimally want to roll back as quickly as possible. The simulation records the first hour at which the SAVI confidence-sequence rule calls for a rollback.

Show the simulation code

``` python
CANARY_USERS_PER_HOUR = 500
CANARY_MAX_HOURS = 48
CANARY_HARM_TOLERANCE = 0.01
CANARY_MIXTURE_PRECISION = pf.optimal_mixture_precision(
    nobs=2 * CANARY_USERS_PER_HOUR * CANARY_MAX_HOURS,
    number_of_coefficients=2,
    alpha=ALPHA,
)

canary_full = _experiment_path(
    p_control=0.10,
    p_treatment=0.07,
    seed=8,
    with_intervals=True,
    n_days=CANARY_MAX_HOURS,
    users_per_day=CANARY_USERS_PER_HOUR,
    mixture_precision=CANARY_MIXTURE_PRECISION,
).set_index("day", drop=False)

rollback_hour = int(
    canary_full.loc[
        canary_full["cs_upper"] < -CANARY_HARM_TOLERANCE,
        "day",
    ].iloc[0]
)
rollback_cs = (
    f"[{canary_full.loc[rollback_hour, 'cs_lower']:.2%}, "
    f"{canary_full.loc[rollback_hour, 'cs_upper']:.2%}]"
)
hours_saved = CANARY_MAX_HOURS - rollback_hour
canary_path = canary_full.loc[canary_full["day"] <= rollback_hour]
```

Show the figure code

``` python
band = {"mode": "lines", "line": {"width": 0}, "hoverinfo": "skip"}
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=canary_path["day"], y=canary_path["cs_upper"], showlegend=False, **band)
)
fig.add_trace(
    go.Scatter(
        x=canary_path["day"],
        y=canary_path["cs_lower"],
        fill="tonexty",
        fillcolor=_rgba(SAVI_BLUE, 0.15),
        name="95% SAVI confidence sequence",
        **band,
    )
)
fig.add_trace(
    go.Scatter(
        x=canary_path["day"],
        y=canary_path["estimate"],
        mode="lines+markers",
        line_color=ESTIMATE_TEAL,
        name="Estimated effect",
        hovertemplate="Hour %{x}<br>Effect: %{y:.1%}<extra></extra>",
    )
)
fig.add_hline(y=0, line_color="black", opacity=0.4)
fig.add_hline(
    y=-CANARY_HARM_TOLERANCE,
    line_dash="dash",
    line_color=ORDINARY_ORANGE,
    annotation_text="Decision boundary: −1 pp",
    annotation_position="top left",
)
fig.add_hline(
    y=-0.03,
    line_dash="dot",
    line_color=NEUTRAL_GREY,
    annotation_text="True effect: −3 pp",
    annotation_position="bottom left",
)
fig.add_vline(
    x=rollback_hour,
    line_dash="dash",
    line_color=ORDINARY_ORANGE,
    opacity=0.7,
    annotation_text=f"Hour {rollback_hour}: roll back",
    annotation_position="top right",
    annotation_font_color=ORDINARY_ORANGE,
)
fig.add_vline(
    x=CANARY_MAX_HOURS,
    line_dash="dash",
    line_color=NEUTRAL_GREY,
    opacity=0.7,
    annotation_text="Planned 2-day runtime",
    annotation_position="bottom left",
    annotation_font_color=NEUTRAL_GREY,
)
fig.update_layout(
    title="A harmful canary release",
    template="plotly_white",
    height=430,
    xaxis_title="Hour since release",
    xaxis_range=[0.5, CANARY_MAX_HOURS + 0.5],
    yaxis_title="Estimated effect on conversion",
    yaxis_tickformat=".0%",
    yaxis_range=[-0.085, 0.025],
    legend={"orientation": "h", "y": -0.3},
    margin={"l": 70, "r": 35, "t": 70, "b": 120},
)
fig
```

A canary with a true effect of -3 pp. Rollback occurs when the SAVI confidence sequence’s upper endpoint falls below the -1 pp boundary.

Initially, the SAVI confidence sequence is wide because one hour of data provides little information. After 15 hours, the SAVI confidence sequence is \[-3.88%, -1.03%\]; its upper endpoint is below -1 pp, so the rollback rule fires. A fixed-time test planned for hour 48 would instead have to wait another 33 hours before making the same decision.

> **NOTE:**
>
> It seems tempting to compute the usual fixed-time interval every hour and roll back whenever its upper endpoint drops below -1 pp. While it is true that such a rule catches bad releases earlier, it also rolls back too often when the new version is not actually harmful, because it gives itself 48 chances to find one “bad-looking” interval.

## Regression adjustment can narrow the SAVI confidence sequence

In many A/B tests, users differ before treatment: some were already more likely to convert, spend, or return. A pre-treatment metric that predicts the outcome can absorb part of this baseline variation, and we can pick up these differences in a regression model by including pre-experimental measurements of the metric. While this does not change the treatment effect being estimated, as long as the metric was measured before treatment and is unaffected by treatment assignment, it can make the SAVI confidence sequence narrower. This procedure is known as variance reduction via regression adjustment.

Lindon et al.’s method fits naturally because it works with regression fits and not only with raw differences in means. We can add the pre-treatment metric to the regression and still use the same SAVI confidence sequence for the treatment coefficient.

In our last simulation, we estimate treatment effects with SAVI inference with and without regression adjustment. We simulate a true average treatment effect of 0.18. The data-generating process is nonlinear, heteroskedastic, and heavy-tailed, with a treatment effect that varies with the pre-experiment metric. The adjusted regression follows [Lin (2013)](https://doi.org/10.1214/12-AOAS583): it includes the centered pre-experiment metric and its interaction with treatment. Both regressions use robust standard errors to account for heteroskedasticity. The adjusted conditional-mean model omits the quadratic term. The unadjusted regression, `outcome ~ treated`, correctly specifies the marginal conditional mean \\E\[Y \mid T\]\\, but leaves variation explained by the pre-experiment metric in the residual. For each day, the simulation records the median half-width of the SAVI confidence sequence (the distance from the point estimate to either endpoint) and the share of experiments that meet the stopping rule.

Show the regression-adjustment simulation

``` python
ADJUSTMENT_SIMULATIONS = 200
ADJUSTMENT_NULL_SIMULATIONS = 200
ADJUSTMENT_USERS_PER_DAY = 200
ADJUSTMENT_EFFECT = 0.18
ADJUSTMENT_TARGET_N = N_DAYS * ADJUSTMENT_USERS_PER_DAY
ADJUSTMENT_MIXTURE_PRECISION = pf.optimal_mixture_precision(
    nobs=ADJUSTMENT_TARGET_N,
    number_of_coefficients=4,
    alpha=ALPHA,
)


def _adjustment_path(
    seed,
    average_effect=ADJUSTMENT_EFFECT,
    labels=("unadjusted", "adjusted"),
):
    rng = np.random.default_rng(seed)
    batches = []
    path = []
    for day in range(1, N_DAYS + 1):
        pre_metric = rng.normal(size=ADJUSTMENT_USERS_PER_DAY)
        treated = rng.binomial(1, 0.5, size=ADJUSTMENT_USERS_PER_DAY)
        residual = rng.standard_t(df=3, size=ADJUSTMENT_USERS_PER_DAY) / np.sqrt(3)
        outcome = (
            1.5 * pre_metric
            + 0.5 * (pre_metric**2 - 1)
            + treated * (average_effect + 0.12 * pre_metric)
            + (0.8 + 0.25 * pre_metric**2) * residual
        )
        batches.append(
            pd.DataFrame(
                {
                    "outcome": outcome,
                    "treated": treated,
                    "pre_metric": pre_metric,
                }
            )
        )
        data = pd.concat(batches, ignore_index=True)
        # The population mean is zero and remains fixed across all looks.
        data["pre_metric_centered"] = data["pre_metric"]
        row = {"day": day}
        for label, formula in [
            ("unadjusted", "outcome ~ treated"),
            ("adjusted", "outcome ~ treated * pre_metric_centered"),
        ]:
            if label not in labels:
                continue
            fit = pf.feols(formula, data=data, vcov="hetero")
            cs = fit.confint(
                alpha=ALPHA,
                inference_type="savi",
                mixture_precision=ADJUSTMENT_MIXTURE_PRECISION,
            ).loc["treated"]
            row[f"{label}_half_width"] = float((cs.iloc[1] - cs.iloc[0]) / 2)
            row[f"{label}_pvalue"] = float(
                fit.pvalue_savi(
                    mixture_precision=ADJUSTMENT_MIXTURE_PRECISION
                ).loc["treated"]
            )
            row[f"{label}_estimate"] = float(fit.coef().loc["treated"])
        path.append(row)
    return pd.DataFrame(path)


adjustment_width = {
    "unadjusted": np.empty((ADJUSTMENT_SIMULATIONS, N_DAYS)),
    "adjusted": np.empty((ADJUSTMENT_SIMULATIONS, N_DAYS)),
}
adjustment_pvalue = {
    "unadjusted": np.empty((ADJUSTMENT_SIMULATIONS, N_DAYS)),
    "adjusted": np.empty((ADJUSTMENT_SIMULATIONS, N_DAYS)),
}
adjustment_estimate = {
    "unadjusted": np.empty((ADJUSTMENT_SIMULATIONS, N_DAYS)),
    "adjusted": np.empty((ADJUSTMENT_SIMULATIONS, N_DAYS)),
}

for simulation in range(ADJUSTMENT_SIMULATIONS):
    path = _adjustment_path(seed=30_000 + simulation)
    for label in ("unadjusted", "adjusted"):
        adjustment_width[label][simulation] = path[
            f"{label}_half_width"
        ].to_numpy()
        adjustment_pvalue[label][simulation] = path[f"{label}_pvalue"].to_numpy()
        adjustment_estimate[label][simulation] = path[
            f"{label}_estimate"
        ].to_numpy()

adjustment_null_pvalue = np.empty((ADJUSTMENT_NULL_SIMULATIONS, N_DAYS))
for simulation in range(ADJUSTMENT_NULL_SIMULATIONS):
    path = _adjustment_path(
        seed=40_000 + simulation,
        average_effect=0,
        labels=("adjusted",),
    )
    adjustment_null_pvalue[simulation] = path["adjusted_pvalue"].to_numpy()

adjustment_days = np.arange(1, N_DAYS + 1)
median_width = {
    label: np.median(adjustment_width[label], axis=0)
    for label in ("unadjusted", "adjusted")
}
stopped_share = {
    label: np.maximum.accumulate(
        adjustment_pvalue[label] <= ALPHA,
        axis=1,
    ).mean(axis=0)
    for label in ("unadjusted", "adjusted")
}
mean_estimate = {
    label: adjustment_estimate[label][:, -1].mean()
    for label in ("unadjusted", "adjusted")
}
adjustment_null_rejection_rate = (
    adjustment_null_pvalue <= ALPHA
).any(axis=1).mean()
adjustment_null_rejections = (
    adjustment_null_pvalue <= ALPHA
).any(axis=1).sum()
```

Show the figure code

``` python
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.15,
    subplot_titles=(
        "Median 95% SAVI confidence-sequence half-width",
        "Share of experiments that can stop",
    ),
)

for label, color, dash, display in [
    ("unadjusted", NEUTRAL_GREY, "dot", "Without adjustment"),
    ("adjusted", SAVI_BLUE, "solid", "With fully interacted adjustment"),
]:
    fig.add_trace(
        go.Scatter(
            x=adjustment_days,
            y=median_width[label],
            mode="lines+markers",
            line={"color": color, "dash": dash},
            name=display,
            hovertemplate="Day %{x}<br>Half-width: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=adjustment_days,
            y=stopped_share[label],
            mode="lines+markers",
            line={"color": color, "dash": dash},
            name=display,
            showlegend=False,
            hovertemplate="Day %{x}<br>Stopped: %{y:.0%}<extra></extra>",
        ),
        row=2,
        col=1,
    )

fig.add_hline(
    y=ADJUSTMENT_EFFECT,
    line_dash="dash",
    line_color=NEUTRAL_GREY,
    annotation_text="Reference: half-width = ATE magnitude (0.18)",
    annotation_position="bottom right",
    row=1,
    col=1,
)
fig.update_layout(
    template="plotly_white",
    height=590,
    hovermode="x unified",
    legend={"orientation": "h", "y": -0.18},
    margin={"l": 75, "r": 35, "t": 75, "b": 105},
)
fig.update_yaxes(title_text="Half-width", row=1, col=1)
fig.update_yaxes(
    title_text="Stopped",
    tickformat=".0%",
    range=[0, 1],
    row=2,
    col=1,
)
fig.update_xaxes(title_text="Day of experiment", dtick=2, row=2, col=1)
fig
```

The same experiments are analyzed with and without a pre-treatment covariate for regression adjustment. In this design, adjustment narrows the median SAVI confidence sequence (top) and raises the share of experiments that can stop by the end of the planned runtime (bottom).

By day twenty, the median SAVI confidence-sequence half-width falls from 0.195 without adjustment to 0.124 with adjustment. The adjusted analysis meets the stopping rule in 96% of experiments, compared with 50% without the pre-experiment metric.

At day twenty, the average treatment estimates are 0.183 without adjustment and 0.185 with adjustment, against a true average effect of 0.180. Both are close to the true value in these 200 simulations; the adjusted analysis is more precise because the pre-experimental covariate predicts the experimental outcome.

To check whether adjustment inflates false positives in this design, the simulation is repeated with an average treatment effect of zero. Across 200 experiments, the SAVI p-value fell below 0.05 by day twenty in 2 runs (1.0%).

## SAVI in PyFixest

In `PyFixest`, SAVI inference following Lindon et al. (2026) is available as a post-estimation method via the `pvalue_savi`, `evalue`, and `confint` methods.

``` python
rng = np.random.default_rng(42)
n_per_arm = 800

experiment_data = pd.DataFrame(
    {
        "treated": np.repeat([0, 1], n_per_arm),
        "converted": np.r_[
            rng.binomial(1, 0.10, n_per_arm),
            rng.binomial(1, 0.18, n_per_arm),
        ],
    }
)

fit = pf.feols("converted ~ treated", data=experiment_data, vcov="hetero")

mixture_precision = pf.optimal_mixture_precision(
    nobs=8_000,
    number_of_coefficients=2,
    alpha=0.05,
)
```

At a 5% threshold, the decision rule for `treated` has three equivalent forms:

``` python
# Rule 1: you stop when the SAVI p-value is at most 0.05.
fit.pvalue_savi(
    mixture_precision=mixture_precision,
).loc["treated"].round(3)
```

    np.float64(0.0)

``` python
# Rule 2: you stop when the SAVI e-value is at least 20 (= 1 / 0.05).
fit.evalue(
    mixture_precision=mixture_precision,
).loc["treated"].round(3)
```

    np.float64(76566.682)

``` python
# Rule 3: you stop when zero falls outside the SAVI confidence sequence.
fit.confint(
    inference_type="savi",
    mixture_precision=mixture_precision,
).loc["treated"].round(3)
```

    2.5%     0.050
    97.5%    0.168
    Name: treated, dtype: float64

We can compare the SAVI confidence sequence with a fixed-time confidence interval:

``` python
fit.confint(
    inference_type="regular",
    alpha=0.05,
).loc["treated"].round(3)
```

    2.5%     0.074
    97.5%    0.143
    Name: treated, dtype: float64

As we can see, the SAVI confidence sequence is wider than the fixed-time interval. This is also true for the fixed-time p-value:

``` python
fit.pvalue().loc["treated"].round(3)
```

    np.float64(0.0)

## Footnotes

[^1]: This is also useful for A/A baseline checks of an experimentation platform’s randomization. Repeated fixed-time looks can eventually produce a statistically significant estimate by chance; SAVI controls the probability of any such crossing within the monitored sequence.

[^2]: The holdout and canary examples assume that incoming units are independent and that the target effect remains fixed during monitoring. PyFixest’s current iid/heteroskedastic SAVI implementation does not cover repeated measurements from the same users, serial dependence, changing traffic composition, or an evolving release policy. [Lindon et al. (2026)](https://doi.org/10.1080/01621459.2026.2692052) list dependent outcomes as future work.

[^3]: Its closed-form formulas also make the method fast to compute.

[^4]: If you are a data scientist reading this, all the cool engineering teams do it, mostly because Kotlin has [coroutines](https://kotlinlang.org/docs/coroutines-overview.html).
