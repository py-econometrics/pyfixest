# When to Use Anytime-Valid Inference

Inference

AB Testing

A practical guide to monitoring randomized experiments, stopping early, and avoiding false positives from peeking.

> **TIP:**
>
> Use safe anytime-valid inference (SAVI) when an experiment result may change a decision before a fixed end date. That includes shipping a winning variant, rolling back a harmful release, or continuing an inconclusive test.
>
> If the team will wait for one agreed sample size and analyze the experiment once, ordinary fixed-horizon inference is simpler and usually more precise. SAVI buys flexibility, not extra power.

## Start with the decision rule

Looking at a dashboard does not invalidate a test. Acting on an ordinary p-value before the planned analysis can.

| What the team may do | Ordinary fixed-horizon inference | SAVI |
|----|----|----|
| Look during the test, but wait for the planned final analysis | Valid | Valid |
| Ship or roll back as soon as the evidence crosses a threshold | Not covered by the original test | Designed for this |
| Keep running because the planned result is close but inconclusive | Not covered by the original test | Valid |
| Analyze once at a fixed sample size | Usually the better fit | Valid, but typically more conservative |

If a dashboard, alert, or experiment review can move the launch date, the analysis is sequential. That is the setting SAVI addresses.

## Why ordinary p-values fail when you peek

Suppose a checkout experiment is scheduled to run for twenty days. The dashboard updates every morning. On day four, the p-value falls below 0.05 and the team considers shipping.

The p-value is not broken. It is calibrated for one analysis at a fixed sample size. It is not calibrated for the rule “check every day and stop the first time p \< 0.05.”

Repeated checks give noise more opportunities to cross the threshold. The checks are correlated, so this is not the same as running twenty independent tests, but the false-positive rate still rises above 5%. The same problem appears when a team stops early on a bad result or extends a test because p = 0.06 at the planned end.

Stopping on an unusually favorable estimate also exaggerates the reported effect. The estimate often shrinks after launch. This is the winner’s curse.

## What SAVI changes

SAVI calibrates the evidence over the full path of the experiment. Under the method’s assumptions, the probability that a null experiment ever crosses the 5% decision boundary is asymptotically no more than 5%. The number and timing of the checks do not need to be fixed in advance.

That flexibility has a cost. SAVI asks for stronger evidence early on, and its confidence sequences are wider than ordinary confidence intervals. A fixed-horizon test will often detect a modest effect more often at its planned end date. SAVI is useful when the ability to act early or wait longer is worth that loss in precision.

> **NOTE:**
>
> An e-value uses 1 as its baseline and can rise or fall as data arrive. Large values count against “no effect.” At a 5% level, the decision boundary is 20 because \\1 / 0.05 = 20\\. An e-value of 20 does not mean that the alternative is twenty times more likely than the null. It means that the evidence has reached the boundary calibrated for the sequential decision rule.
>
> PyFixest also reports `min(1, 1 / e-value)` as a sequential p-value. The two scales carry the same information: e-value \>= 20 and sequential p-value \<= 0.05 cross at the same time. A confidence sequence gives the same decision in effect size units. It excludes zero when the evidence boundary is crossed.

The examples below cover four decisions: avoid a false alarm, ship a winning variant, certify that an effect is too small to matter, and roll back a harmful canary. They also show how a pre-experiment covariate can narrow a confidence sequence. Use Play or the day slider in the single-experiment figures.

The main A/B test is planned for twenty days and treats a 2 percentage point lift as the minimum detectable effect (MDE). Its mixture precision is chosen before the simulation to make the confidence sequence narrowest at the planned end, then held fixed at every look. The operational examples use the same rule with their own planned monitoring windows.

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
    mixture_precision=MAIN_MIXTURE_PRECISION,
):
    """Run many experiments; keep the full day-by-day estimate and p-value paths."""
    estimate = np.full((n_simulations, N_DAYS), np.nan)
    regular_p = np.ones((n_simulations, N_DAYS))
    savi_p = np.ones((n_simulations, N_DAYS))
    for s in range(n_simulations):
        path = _experiment_path(
            p_control,
            p_treatment,
            seed + s,
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

## A/A tests: repeated peeking creates false alarms

Start with 500 simulated A/A tests. Both arms have a 10% conversion rate, so every rejection is a false positive.

The figure follows eighty of the experiments. Each circle is one A/A test. It changes to a cross on the first day its method reports p \< 0.05. The experiment positions are the same in both panels, so the comparison uses the same data.

Show the simulation code

``` python
mirage = _simulate_paths(p_control=0.10, p_treatment=0.10, seed=42)

fpr_single_look = (mirage["regular_p"][:, -1] <= ALPHA).mean()
fpr_daily_ordinary = (mirage["regular_p"] <= ALPHA).any(axis=1).mean()
fpr_daily_savi = (mirage["savi_p"] <= ALPHA).any(axis=1).mean()
```

Show the figure code

``` python
N_SHOWN = 80
EVIDENCE_BAR = -np.log10(ALPHA)

show_reg = mirage["regular_p"][:N_SHOWN]
show_savi = mirage["savi_p"][:N_SHOWN]
reg_stop = _stop_day(show_reg)
savi_stop = _stop_day(show_savi)

dot_x = np.tile(np.arange(1, 11), 8)
dot_y = np.repeat(np.arange(8, 0, -1), 10)


def _mirage_trace(stop, day, color):
    fired = (stop != 0) & (stop <= day)
    status = [
        f"False alarm on day {stop[i]}" if fired[i] else f"No false alarm by day {day}"
        for i in range(N_SHOWN)
    ]
    return go.Scatter(
        x=dot_x,
        y=dot_y,
        mode="markers",
        marker={
            "size": 13,
            "symbol": ["x" if value else "circle-open" for value in fired],
            "color": [
                color if value else _rgba(NEUTRAL_GREY, 0.45) for value in fired
            ],
            "line": {"width": 1.5},
        },
        customdata=np.arange(1, N_SHOWN + 1),
        text=status,
        hovertemplate="Experiment %{customdata}<br>%{text}<extra></extra>",
        showlegend=False,
    )


def _mirage_traces(day):
    return [
        _mirage_trace(reg_stop, day, ORDINARY_ORANGE),
        _mirage_trace(savi_stop, day, SAVI_BLUE),
    ]


def _mirage_title(day):
    n_reg = int(((reg_stop != 0) & (reg_stop <= day)).sum())
    n_savi = int(((savi_stop != 0) & (savi_stop <= day)).sum())
    return (
        f"Day {day}: false alarms so far"
        f"<br>Ordinary: {n_reg}/{N_SHOWN} · SAVI: {n_savi}/{N_SHOWN}"
    )


mirage_days = list(range(1, N_DAYS + 1))
fig = make_subplots(
    rows=2,
    cols=1,
    vertical_spacing=0.20,
    subplot_titles=(
        "Ordinary test, checked daily",
        "SAVI, checked daily",
    ),
)
for i, trace in enumerate(_mirage_traces(mirage_days[0])):
    fig.add_trace(trace, row=i + 1, col=1)

fig.frames = [
    go.Frame(
        data=_mirage_traces(day),
        traces=[0, 1],
        name=str(day),
        layout={"title": {"text": _mirage_title(day)}},
    )
    for day in mirage_days
]
fig.update_xaxes(visible=False, range=[0.3, 10.7])
fig.update_yaxes(visible=False, range=[0.4, 8.6])
fig.add_annotation(
    x=0.5,
    y=-0.06,
    xref="paper",
    yref="paper",
    text="○ no false alarm yet&nbsp;&nbsp;&nbsp;&nbsp;× false alarm",
    showarrow=False,
    font={"color": NEUTRAL_GREY},
)
fig.update_layout(
    title=_mirage_title(mirage_days[0]),
    title_font_size=16,
    template="plotly_white",
    height=570,
    updatemenus=PLAY_PAUSE,
    sliders=_slider(mirage_days, active=0),
    margin={"l": 45, "r": 30, "t": 120, "b": 155},
)
fig
```

Across all 500 experiments:

Show the table code

``` python
pd.DataFrame(
    {
        "Monitoring policy": [
            "Ordinary test, one look on day 20",
            "Ordinary test, checked daily",
            "SAVI, checked daily",
        ],
        "False-positive rate": [
            f"{fpr_single_look:.1%}",
            f"{fpr_daily_ordinary:.1%}",
            f"{fpr_daily_savi:.1%}",
        ],
    }
)
```

|     | Monitoring policy                 | False-positive rate |
|-----|-----------------------------------|---------------------|
| 0   | Ordinary test, one look on day 20 | 5.2%                |
| 1   | Ordinary test, checked daily      | 25.0%               |
| 2   | SAVI, checked daily               | 1.0%                |

One ordinary analysis on day twenty rejects 5.2% of the time, close to the intended 5%. Checking the ordinary p-value daily and stopping at the first rejection raises the rate to 25%. SAVI rejects 1.0% of the same null experiments.

The SAVI rate in this simulation need not equal 5%. The guarantee covers the full monitoring path, including looks after day twenty, so rejection by the planned end can be conservative.

## A positive effect: stop when the evidence is ready

Now suppose the treatment raises conversion from 10% to 14%: twice the lift the experiment was planned around. This is the case where early stopping is most valuable. The displayed run is not a hand-picked fast result; among the simulations that stop within twenty days, its SAVI stopping day is closest to the median.

The top panel shows the estimated lift. The ordinary 95% confidence interval is calibrated for one planned analysis; the 95% confidence sequence covers the full monitoring path. The bottom panel puts both p-values on the same \\-\log\_{10}(p)\\ scale.

Show the simulation code

``` python
TRUE_LIFT = 2 * PLANNED_MDE
WINNER_FIRST_SEED = 20_000

winner_candidates = _simulate_paths(
    p_control=0.10,
    p_treatment=0.10 + TRUE_LIFT,
    seed=WINNER_FIRST_SEED,
    n_simulations=N_REPRESENTATIVE_PATHS,
)
winner_stop_days = _stop_day(winner_candidates["savi_p"])
winner_seed = _representative_seed(winner_stop_days, WINNER_FIRST_SEED)

winner = _experiment_path(
    p_control=0.10,
    p_treatment=0.10 + TRUE_LIFT,
    seed=winner_seed,
    with_intervals=True,
).set_index("day", drop=False)

peek_day = int(winner.loc[winner["regular_pvalue"] <= ALPHA, "day"].iloc[0])
savi_day = int(winner.loc[winner["savi_pvalue"] <= ALPHA, "day"].iloc[0])
stop_row = winner.loc[savi_day]
days_saved = N_DAYS - savi_day

peek_lift = f"{winner.loc[peek_day, 'estimate']:.1%}"
savi_users = f"{int(stop_row['users']):,}"
savi_lift = f"{stop_row['estimate']:.1%}"
savi_cs = f"[{stop_row['cs_lower']:.1%}, {stop_row['cs_upper']:.1%}]"
```

Show the animation code

``` python
winner_days = list(range(1, N_DAYS + 1))
shown = winner.loc[winner_days]
estimate_range = [
    float(shown["cs_lower"].min()) - 0.01,
    float(shown["cs_upper"].max()) + 0.01,
]
evidence_top = float((-np.log10(shown[["regular_pvalue", "savi_pvalue"]])).max().max())


def _winner_traces(day):
    seen = winner.loc[winner["day"] <= day]
    current = winner.loc[day]
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
            name="95% confidence sequence",
            **band,
        ),
        go.Scatter(x=seen["day"], y=seen["ci_upper"], showlegend=False, **band),
        go.Scatter(
            x=seen["day"],
            y=seen["ci_lower"],
            fill="tonexty",
            fillcolor=_rgba(ORDINARY_ORANGE, 0.18),
            name="95% confidence interval",
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
                f"Day {day}<br>Confidence sequence: "
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
                f"Day {day}<br>Ordinary interval: "
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
            y=-np.log10(seen["regular_pvalue"]),
            mode="lines+markers",
            line={"color": ORDINARY_ORANGE, "dash": "dot"},
            name="Ordinary p-value",
            customdata=seen["regular_pvalue"],
            hovertemplate="Day %{x}<br>p = %{customdata:.4f}<extra></extra>",
        ),
        go.Scatter(
            x=seen["day"],
            y=-np.log10(seen["savi_pvalue"]),
            mode="lines+markers",
            line_color=SAVI_BLUE,
            name="SAVI p-value",
            customdata=seen["savi_pvalue"],
            hovertemplate="Day %{x}<br>p = %{customdata:.4f}<extra></extra>",
        ),
        go.Scatter(
            x=[peek_day] if ordinary_crossed else [None],
            y=[-np.log10(winner.loc[peek_day, "regular_pvalue"])]
            if ordinary_crossed
            else [None],
            mode="markers+text",
            marker={"color": ORDINARY_ORANGE, "size": 10, "symbol": "diamond"},
            text=["Ordinary crosses"] if ordinary_crossed else [None],
            textposition="bottom right",
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[savi_day] if savi_crossed else [None],
            y=[-np.log10(winner.loc[savi_day, "savi_pvalue"])]
            if savi_crossed
            else [None],
            mode="markers+text",
            marker={"color": SAVI_BLUE, "size": 10, "symbol": "diamond"},
            text=["SAVI: valid stop"] if savi_crossed else [None],
            textposition="top right",
            showlegend=False,
            hoverinfo="skip",
        ),
    ]


def _winner_title(day):
    users = int(winner.loc[day, "users"])
    if day < peek_day:
        decision = "Continue"
    elif day < savi_day:
        decision = "Ordinary p < 0.05; no fixed-sample decision yet"
    else:
        decision = "SAVI boundary reached; stopping is valid"
    return f"Day {day} of {N_DAYS} · {users:,} users<br>{decision}"


fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.12,
    subplot_titles=("Estimated lift", "Evidence against 'no effect'"),
)
for i, trace in enumerate(_winner_traces(winner_days[0])):
    fig.add_trace(trace, row=1 if i < 7 else 2, col=1)
fig.frames = [
    go.Frame(
        data=_winner_traces(day),
        traces=list(range(11)),
        name=str(day),
        layout={"title": {"text": _winner_title(day)}},
    )
    for day in winner_days
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
fig.update_layout(
    title=_winner_title(winner_days[0]),
    title_font_size=16,
    template="plotly_white",
    height=740,
    xaxis_range=[0.5, N_DAYS + 0.5],
    xaxis2_range=[0.5, N_DAYS + 0.5],
    yaxis_range=estimate_range,
    yaxis2_range=[0, evidence_top + 0.8],
    yaxis_tickformat=".0%",
    yaxis2_title="−log10(p-value)",
    xaxis2_title="Day of experiment",
    legend={"orientation": "h", "y": -0.48},
    updatemenus=PLAY_PAUSE,
    sliders=_slider(winner_days, active=0),
    margin={"l": 70, "r": 35, "t": 125, "b": 270},
)
fig.update_xaxes(dtick=2)
fig
```

The ordinary p-value clears 0.05 on day 3, at an estimated lift of 4.5%. That is too early for the fixed-horizon test to support a 5%-level decision. SAVI clears its boundary on day 7, after 2,800 users. The confidence sequence is \[0.3%, 8.0%\], so the team can ship and release the remaining 13 planned days.

This effect is larger than planned. The cost of sequential monitoring is easier to see at the planned 2 percentage point MDE:

Show the simulation code

``` python
handcuffs = _simulate_paths(
    p_control=0.10,
    p_treatment=0.10 + PLANNED_MDE,
    seed=10_000,
)

savi_detected = np.maximum.accumulate(handcuffs["savi_p"] <= ALPHA, axis=1)
savi_detection = savi_detected.mean(axis=0)
oneshot_power = (handcuffs["regular_p"][:, -1] <= ALPHA).mean()

savi_stop_days = _stop_day(handcuffs["savi_p"])
median_stop = int(np.median(savi_stop_days[savi_stop_days > 0]))
```

Show the figure code

``` python
days = np.arange(1, N_DAYS + 1)
oneshot_curve = np.zeros(N_DAYS)
oneshot_curve[-1] = oneshot_power

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=days,
        y=oneshot_curve,
        mode="lines+markers",
        line={"shape": "hv", "dash": "dash", "color": ORDINARY_ORANGE},
        name="Ordinary test, one look on day 20",
        hovertemplate="Day %{x}<br>Detected: %{y:.0%}<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=days,
        y=savi_detection,
        mode="lines+markers",
        line_color=SAVI_BLUE,
        name="SAVI, may stop any day",
        hovertemplate="Day %{x}<br>Detected by this day: %{y:.0%}<extra></extra>",
    )
)
fig.add_vline(
    x=median_stop,
    line_dash="dot",
    line_color=SAVI_BLUE,
    opacity=0.5,
    annotation_text=f"Median SAVI stop: day {median_stop}",
    annotation_position="top left",
    annotation_font_color=SAVI_BLUE,
)
fig.update_layout(
    title="Detection of the planned 2 pp lift",
    xaxis_title="Day of experiment",
    yaxis_title="Share detected",
    yaxis_tickformat=".0%",
    yaxis_range=[0, 1],
    hovermode="x unified",
    template="plotly_white",
    legend={"orientation": "h", "y": -0.25},
    height=460,
    margin={"l": 75, "r": 30, "t": 90, "b": 115},
)
fig
```

At day twenty, the fixed-horizon test detects the effect in 80% of experiments. SAVI has stopped in 16% by day ten and 49% by day twenty. Among experiments that stop, the median stopping day is 13.

This is the trade: SAVI can act before day twenty, but by day twenty it has detected fewer of the modest effects. Choose it for flexible timing, not as a way to make significance arrive faster.

## Regression adjustment narrows the sequence

Regression adjustment is not unique to SAVI. What matters here is that the Lindon et al. method builds anytime-valid inference around a regression coefficient. A predictive pre-treatment covariate can therefore reduce residual variation, just as it does in an ordinary fixed-horizon analysis, while the treatment test remains valid under continuous monitoring.

The simulation below deliberately breaks the working linear model. Treatment is randomized independently for each user. The untreated outcome depends nonlinearly on a centered pre-experiment metric, the treatment effect varies with that metric, and the residuals are heavy-tailed and heteroskedastic.

The adjusted regression includes the pre-experiment metric and its interaction with treatment. Because the metric is centered, the coefficient on `treated` targets the average treatment effect. Both regressions use robust standard errors, and neither regression includes the quadratic term or the changing residual scale used to generate the data. This is a teaching example rather than a reproduction of the paper’s simulation; it keeps the features needed to check the same robustness argument.

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
        row = {"day": day}
        for label, formula in [
            ("unadjusted", "outcome ~ treated"),
            ("adjusted", "outcome ~ treated * pre_metric"),
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
adjustment_null_fire_rate = (
    adjustment_null_pvalue <= ALPHA
).any(axis=1).mean()
```

Show the figure code

``` python
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.15,
    subplot_titles=(
        "Median 95% confidence-sequence half-width",
        "Share of experiments that can stop",
    ),
)

for label, color, dash, display in [
    ("unadjusted", ORDINARY_ORANGE, "dot", "Without adjustment"),
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
    annotation_text="ATE magnitude: 0.18",
    annotation_position="top right",
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

By day twenty, the median confidence-sequence half-width falls from 0.195 without adjustment to 0.124 with adjustment. The adjusted analysis has crossed the stopping boundary in 96% of experiments, compared with 50% without the pre-experiment metric.

Both treatment estimates remain centered on the randomized treatment effect, even though the fitted adjustment is not the true outcome model. The covariate helps because it predicts the outcome, not because it fixes confounding. At day twenty, the mean treatment estimates are 0.183 without adjustment and 0.185 with adjustment, against a true average effect of 0.180.

As a stress check, we reran the fully interacted analysis with an average treatment effect of zero. Across 200 experiments, the sequence crossed the 5% boundary by day twenty in 1.0% of runs. This finite-horizon rate can be below 5%; the guarantee covers the full sequence, not only twenty looks.

Use only variables measured before treatment. Adjusting for a post-treatment variable can bias the treatment coefficient.

## A negligible effect can also end the test

A checkout service is moved to a new backend. The change should not affect conversion, and the product decision allows a difference of at most one percentage point in either direction. That ±1 pp tolerance matters more than whether the effect is exactly zero.

The A/A-style test sends 2,000 users per arm per day to the old and new backends. It stops once the confidence sequence fits inside the ±1 pp tolerance. The displayed run has the median certification day among one hundred simulations that certify within the thirty-day monitoring window.

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
            name="95% confidence sequence",
            **band,
        ),
        go.Scatter(x=seen["day"], y=seen["ci_upper"], showlegend=False, **band),
        go.Scatter(
            x=seen["day"],
            y=seen["ci_lower"],
            fill="tonexty",
            fillcolor=_rgba(ORDINARY_ORANGE, 0.18),
            name="95% confidence interval",
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
                f"Day {day}<br>Confidence sequence: "
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
                f"Day {day}<br>Ordinary interval: "
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
    ]


def _backend_title(day):
    users = int(backend_path.loc[day, "users"])
    if day == cert_day:
        decision = "CS inside ±1 pp: tolerance met, test over"
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

On day 13, the confidence sequence \[-0.96%, 0.68%\] fits inside the ±1 pp tolerance. The team can stop and conclude, with the time-uniform asymptotic coverage of the sequence, that any effect is smaller than the amount it cares about.

A non-significant p-value cannot establish that two versions are equivalent. A confidence sequence inside a prespecified tolerance can.

The next plot returns to the 500 A/A experiments from the first section and records the estimated effect when each test first rejects:

Show the simulation code

``` python
naive_stop = _stop_day(mirage["regular_p"])
savi_aa_stop = _stop_day(mirage["savi_p"])


def _estimate_at_stop(stop):
    idx = np.where(stop > 0)[0]
    return np.array([mirage["estimate"][i, stop[i] - 1] for i in idx]) * 100


naive_at_stop = _estimate_at_stop(naive_stop)
savi_at_stop = _estimate_at_stop(savi_aa_stop)

naive_fire_rate = (naive_stop > 0).mean()
savi_fire_rate = (savi_aa_stop > 0).mean()
naive_mean_abs = np.abs(naive_at_stop).mean()
```

Show the figure code

``` python
jitter = np.random.default_rng(1).uniform(-0.16, 0.16, size=N_SIMULATIONS)

fig = go.Figure()
for est, base, color, label in [
    (naive_at_stop, 1, ORDINARY_ORANGE, "Naive daily peeking"),
    (savi_at_stop, 0, SAVI_BLUE, "SAVI"),
]:
    fig.add_trace(
        go.Scatter(
            x=est,
            y=base + jitter[: len(est)],
            mode="markers",
            marker={"color": _rgba(color, 0.5), "size": 7, "line": {"width": 0}},
            name=f"{label} ({len(est)} of {N_SIMULATIONS} fired)",
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
    title="'Effects' reported at the first significant result<br>(true effect: 0)",
    template="plotly_white",
    height=380,
    xaxis_title="Estimated effect at stop (pp)",
    yaxis={
        "tickmode": "array",
        "tickvals": [0, 1],
        "ticktext": ["SAVI", "Naive<br>peeking"],
        "range": [-0.5, 1.5],
    },
    legend={"orientation": "h", "y": -0.3},
    margin={"l": 80, "r": 30, "t": 90, "b": 105},
)
fig
```

Daily peeking stops 25% of these zero-effect experiments. The reported effects average 3.9 pp in magnitude. There is a hole around zero because only extreme early estimates cross the boundary. This is the winner’s curse in the null case: the stopping rule selects effects that look large even though the truth is zero. SAVI crosses its boundary in 1.0% of the same experiments.

## Canary releases: guardrails read every hour

A canary needs one product judgment before launch: how much harm is small enough to accept? Suppose the team is willing to tolerate at most a 1 pp drop in conversion. The decision boundary is then -1 pp. At each hourly review:

- **Roll back** when the confidence sequence lies entirely below -1 pp. The data now support harm larger than the tolerance.
- **Promote** when the confidence sequence lies entirely above -1 pp. Any harm still compatible with the data is within the tolerance.
- **Keep watching** while the sequence straddles -1 pp.

Both decisions use the same boundary, so the rules cannot conflict. In this example, the canary lowers conversion from 10% to 7%, with 500 users per arm arriving every hour.

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
    f"[{canary_full.loc[rollback_hour, 'cs_lower']:.1%}, "
    f"{canary_full.loc[rollback_hour, 'cs_upper']:.1%}]"
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
        name="95% confidence sequence",
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
    annotation_position="top left",
    annotation_font_color=ORDINARY_ORANGE,
)
fig.update_layout(
    title="A canary release gone wrong",
    template="plotly_white",
    height=430,
    xaxis_title="Hour since release",
    yaxis_title="Estimated effect on conversion",
    yaxis_tickformat=".0%",
    legend={"orientation": "h", "y": -0.3},
    margin={"l": 70, "r": 35, "t": 70, "b": 120},
)
fig
```

The sequence starts wide because one hour of data proves little. After 15 hours it lies entirely below the -1 pp boundary, at \[-3.9%, -1.0%\]. The rule says to roll back. A fixed-horizon test with a planned two-day runtime could not make the planned statistical decision for another 33 hours. For a harmless canary, the lower end of the sequence would eventually clear -1 pp and trigger promotion. Netflix uses sequential tests to safeguard production releases; see [Lindon, Sanden, and Shirikian (2022), “Rapid Regression Detection in Software Deployments through Sequential Testing”](https://doi.org/10.1145/3534678.3539099).

## Apply SAVI in PyFixest

At each review, append the newly completed experimental units to the existing data and refit the same regression. Here, `treated` is the randomized treatment indicator and `converted` is the primary metric.

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
# Rule 1: stop when the sequential p-value is at most 0.05.
fit.pvalue_savi(
    mixture_precision=mixture_precision,
).loc["treated"]
```

    np.float64(1.3060511149198845e-05)

``` python
# Rule 2: stop when the e-value is at least 20 (= 1 / 0.05).
fit.evalue(
    mixture_precision=mixture_precision,
).loc["treated"]
```

    np.float64(76566.6816999993)

``` python
# Rule 3: stop when zero falls outside the confidence sequence.
fit.confint(
    inference_type="savi",
    mixture_precision=mixture_precision,
).loc["treated"]
```

    2.5%     0.049847
    97.5%    0.167653
    Name: treated, dtype: float64

Since the sequential p-value is `min(1, 1 / e_value)`, the three rules always agree for a given coefficient. A dashboard can show whichever unit its audience prefers without changing the decision. Confidence sequences are often the easiest to act on because they put the result and any practical tolerance on the same scale.

The `mixture_precision` argument controls where the sequence is most sensitive over time. Here it is chosen to minimize sequence width at the planned 8,000 users. The default is 1, but whichever value you use must be chosen before monitoring and held fixed.

## What the guarantee assumes

The examples in this guide assume that treatment is randomized and that the rows entering the analysis are independent experimental units, such as users. Randomization makes the treatment coefficient interpretable as a causal effect. SAVI protects that coefficient against repeated looks; it does not repair confounding in an observational comparison.

Match the analysis unit to the randomization unit. If treatment is assigned by user, repeated page views from the same user are not independent users. Reduce the outcome to one user-level observation or use inference that accounts for the dependence.

PyFixest uses the asymptotic t-mixture version of SAVI. It is intended for large experiments where the usual regression standard-error approximation is credible, not as a finite-sample guarantee. The test is coefficient-wise: the three accessors above refer to one regression coefficient, not a joint F test.

> **WARNING:**
>
> SAVI lets you check one chosen test as often as you like. It does not correct for testing many outcomes, subgroups, variants, or coefficients. If your experiment has several primary comparisons, pair SAVI with a multiple-testing procedure.

The implementation follows [Lindon, Ham, Tingley, and Bojinov (2026), “Anytime-Valid Inference in Linear Models with Applications to Regression-Adjusted Causal Inference”](https://doi.org/10.1080/01621459.2026.2692052). The paper gives the derivation and a detailed account of the assumptions.
