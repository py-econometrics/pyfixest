# Instrumental Variables

Core Estimation

IV estimation with fixed effects, first-stage diagnostics, and the effective F-statistic.

> **NOTE:**
>
> You should have read the [Getting Started](../getting-started.llms.md) page and have `PyFixest` installed.

## Introduction

Estimation of a linear model via Ordinary Least Squares (OLS) yields biased and inconsistent estimates when a regressor is correlated with the error term — a problem known as endogeneity, which arises, for example, in the presence of unobserved confounders. **Instrumental Variable (IV)** estimation addresses this by finding a variable \\Z\\ that satisfies three conditions:

![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTkyIiBoZWlnaHQ9IjE5MiIgdmlld2JveD0iMC4wMCAwLjAwIDE5Mi4wMCAxMTYuNDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHN0eWxlPSI7IG1heC13aWR0aDogbm9uZTsgbWF4LWhlaWdodDogbm9uZSI+CjxnIGlkPSJncmFwaDAiIGNsYXNzPSJncmFwaCIgdHJhbnNmb3JtPSJzY2FsZSgxIDEpIHJvdGF0ZSgwKSB0cmFuc2xhdGUoNCAxMTIuNCkiPgo8dGl0bGU+SVY8L3RpdGxlPgo8cG9seWdvbiBmaWxsPSJ3aGl0ZSIgc3Ryb2tlPSJ0cmFuc3BhcmVudCIgcG9pbnRzPSItNCw0IC00LC0xMTIuNCAxODgsLTExMi40IDE4OCw0IC00LDQiPjwvcG9seWdvbj4KPCEtLSBaIC0tPgo8ZyBpZD0ibm9kZTEiIGNsYXNzPSJub2RlIj4KPHRpdGxlPlo8L3RpdGxlPgo8ZWxsaXBzZSBmaWxsPSJub25lIiBzdHJva2U9ImJsYWNrIiBjeD0iMTEiIGN5PSItOTcuNCIgcng9IjExIiByeT0iMTEiPjwvZWxsaXBzZT4KPHRleHQgdGV4dC1hbmNob3I9Im1pZGRsZSIgeD0iMTEiIHk9Ii05NC4xIiBmb250LWZhbWlseT0ic2VyaWYiIGZvbnQtc2l6ZT0iMTEuMDAiPlo8L3RleHQ+CjwvZz4KPCEtLSBUIC0tPgo8ZyBpZD0ibm9kZTMiIGNsYXNzPSJub2RlIj4KPHRpdGxlPlQ8L3RpdGxlPgo8ZWxsaXBzZSBmaWxsPSJub25lIiBzdHJva2U9ImJsYWNrIiBjeD0iMjkiIGN5PSItMTEiIHJ4PSIxMSIgcnk9IjExIj48L2VsbGlwc2U+Cjx0ZXh0IHRleHQtYW5jaG9yPSJtaWRkbGUiIHg9IjI5IiB5PSItNy43IiBmb250LWZhbWlseT0ic2VyaWYiIGZvbnQtc2l6ZT0iMTEuMDAiPlQ8L3RleHQ+CjwvZz4KPCEtLSBaJiM0NTsmZ3Q7VCAtLT4KPGcgaWQ9ImVkZ2UxIiBjbGFzcz0iZWRnZSI+Cjx0aXRsZT5aLSZndDtUPC90aXRsZT4KPHBhdGggZmlsbD0ibm9uZSIgc3Ryb2tlPSJibGFjayIgZD0iTTEzLjI4LC04Ni40NkMxNi40MiwtNzEuNCAyMi4wOSwtNDQuMTggMjUuNjksLTI2Ljg4IiAvPgo8cG9seWdvbiBmaWxsPSJibGFjayIgc3Ryb2tlPSJibGFjayIgcG9pbnRzPSIyNy40MywtMjcuMTEgMjYuNzQsLTIxLjg1IDI0LjAxLC0yNi4zOSAyNy40MywtMjcuMTEiPjwvcG9seWdvbj4KPC9nPgo8IS0tIFUgLS0+CjxnIGlkPSJub2RlMiIgY2xhc3M9Im5vZGUiPgo8dGl0bGU+VTwvdGl0bGU+CjxlbGxpcHNlIGZpbGw9Im5vbmUiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS1kYXNoYXJyYXk9IjUsMiIgY3g9IjEwMSIgY3k9Ii05Ny40IiByeD0iMTEiIHJ5PSIxMSI+PC9lbGxpcHNlPgo8dGV4dCB0ZXh0LWFuY2hvcj0ibWlkZGxlIiB4PSIxMDEiIHk9Ii05NC4xIiBmb250LWZhbWlseT0ic2VyaWYiIGZvbnQtc2l6ZT0iMTEuMDAiPlU8L3RleHQ+CjwvZz4KPCEtLSBVJiM0NTsmZ3Q7VCAtLT4KPGcgaWQ9ImVkZ2UzIiBjbGFzcz0iZWRnZSI+Cjx0aXRsZT5VLSZndDtUPC90aXRsZT4KPHBhdGggZmlsbD0ibm9uZSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLWRhc2hhcnJheT0iNSwyIiBkPSJNOTMuODMsLTg4LjhDODEuMTEsLTczLjUzIDU0LjUyLC00MS42MyAzOS42NSwtMjMuNzgiIC8+Cjxwb2x5Z29uIGZpbGw9ImJsYWNrIiBzdHJva2U9ImJsYWNrIiBwb2ludHM9IjQwLjY4LC0yMi4yOSAzNi4xNCwtMTkuNTcgMzcuOTksLTI0LjUzIDQwLjY4LC0yMi4yOSI+PC9wb2x5Z29uPgo8L2c+CjwhLS0gWSAtLT4KPGcgaWQ9Im5vZGU0IiBjbGFzcz0ibm9kZSI+Cjx0aXRsZT5ZPC90aXRsZT4KPGVsbGlwc2UgZmlsbD0ibm9uZSIgc3Ryb2tlPSJibGFjayIgY3g9IjE3MyIgY3k9Ii0xMSIgcng9IjExIiByeT0iMTEiPjwvZWxsaXBzZT4KPHRleHQgdGV4dC1hbmNob3I9Im1pZGRsZSIgeD0iMTczIiB5PSItNy43IiBmb250LWZhbWlseT0ic2VyaWYiIGZvbnQtc2l6ZT0iMTEuMDAiPlk8L3RleHQ+CjwvZz4KPCEtLSBVJiM0NTsmZ3Q7WSAtLT4KPGcgaWQ9ImVkZ2U0IiBjbGFzcz0iZWRnZSI+Cjx0aXRsZT5VLSZndDtZPC90aXRsZT4KPHBhdGggZmlsbD0ibm9uZSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLWRhc2hhcnJheT0iNSwyIiBkPSJNMTA4LjE3LC04OC44QzEyMC44OSwtNzMuNTMgMTQ3LjQ4LC00MS42MyAxNjIuMzUsLTIzLjc4IiAvPgo8cG9seWdvbiBmaWxsPSJibGFjayIgc3Ryb2tlPSJibGFjayIgcG9pbnRzPSIxNjQuMDEsLTI0LjUzIDE2NS44NiwtMTkuNTcgMTYxLjMyLC0yMi4yOSAxNjQuMDEsLTI0LjUzIj48L3BvbHlnb24+CjwvZz4KPCEtLSBUJiM0NTsmZ3Q7WSAtLT4KPGcgaWQ9ImVkZ2UyIiBjbGFzcz0iZWRnZSI+Cjx0aXRsZT5ULSZndDtZPC90aXRsZT4KPHBhdGggZmlsbD0ibm9uZSIgc3Ryb2tlPSJibGFjayIgZD0iTTQwLjMsLTExQzY1LjY0LC0xMSAxMjcuMSwtMTEgMTU2LjYzLC0xMSIgLz4KPHBvbHlnb24gZmlsbD0iYmxhY2siIHN0cm9rZT0iYmxhY2siIHBvaW50cz0iMTU2Ljc1LC0xMi43NSAxNjEuNzUsLTExIDE1Ni43NSwtOS4yNSAxNTYuNzUsLTEyLjc1Ij48L3BvbHlnb24+CjwvZz4KPC9nPgo8L3N2Zz4=)

**Figure 1.** IV setup. Z is the instrument, T the endogenous treatment, Y the outcome, and U an unobserved confounder. Dashed elements are unobserved. U creates a backdoor path T ← U → Y, biasing OLS.

1.  **Relevance**: \\Z\\ has a causal effect on \\T\\.
2.  **Exclusion Restriction**: \\Z\\’s causal effect on \\Y\\ is fully mediated by \\T\\.
3.  **Instrumental Unconfoundedness**: \\Z\\ has no unobserved common causes with \\Y\\.

In Figure 1, the path from \\Z\\ to \\T\\ shows relevance, the absence of a direct path from \\Z\\ to \\Y\\ encodes the exclusion restriction, and the absence of a path from the unobservable to the instrument shows instrumental unconfoundedness.

![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTkyIiBoZWlnaHQ9IjE5MiIgdmlld2JveD0iMC4wMCAwLjAwIDE5Mi4wMCAxMTYuNDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHN0eWxlPSI7IG1heC13aWR0aDogbm9uZTsgbWF4LWhlaWdodDogbm9uZSI+CjxnIGlkPSJncmFwaDAiIGNsYXNzPSJncmFwaCIgdHJhbnNmb3JtPSJzY2FsZSgxIDEpIHJvdGF0ZSgwKSB0cmFuc2xhdGUoNCAxMTIuNCkiPgo8dGl0bGU+SVYyPC90aXRsZT4KPHBvbHlnb24gZmlsbD0id2hpdGUiIHN0cm9rZT0idHJhbnNwYXJlbnQiIHBvaW50cz0iLTQsNCAtNCwtMTEyLjQgMTg4LC0xMTIuNCAxODgsNCAtNCw0Ij48L3BvbHlnb24+CjwhLS0gWiAtLT4KPGcgaWQ9Im5vZGUxIiBjbGFzcz0ibm9kZSI+Cjx0aXRsZT5aPC90aXRsZT4KPGVsbGlwc2UgZmlsbD0ibm9uZSIgc3Ryb2tlPSJibGFjayIgY3g9IjExIiBjeT0iLTk3LjQiIHJ4PSIxMSIgcnk9IjExIj48L2VsbGlwc2U+Cjx0ZXh0IHRleHQtYW5jaG9yPSJtaWRkbGUiIHg9IjExIiB5PSItOTQuMSIgZm9udC1mYW1pbHk9InNlcmlmIiBmb250LXNpemU9IjExLjAwIj5aPC90ZXh0Pgo8L2c+CjwhLS0gVGggLS0+CjxnIGlkPSJub2RlMyIgY2xhc3M9Im5vZGUiPgo8dGl0bGU+VGg8L3RpdGxlPgo8ZWxsaXBzZSBmaWxsPSJub25lIiBzdHJva2U9ImJsYWNrIiBjeD0iMjkiIGN5PSItMTEiIHJ4PSIxMSIgcnk9IjExIj48L2VsbGlwc2U+Cjx0ZXh0IHRleHQtYW5jaG9yPSJtaWRkbGUiIHg9IjI5IiB5PSItNy43IiBmb250LWZhbWlseT0ic2VyaWYiIGZvbnQtc2l6ZT0iMTEuMDAiPlTMgjwvdGV4dD4KPC9nPgo8IS0tIFomIzQ1OyZndDtUaCAtLT4KPGcgaWQ9ImVkZ2UxIiBjbGFzcz0iZWRnZSI+Cjx0aXRsZT5aLSZndDtUaDwvdGl0bGU+CjxwYXRoIGZpbGw9Im5vbmUiIHN0cm9rZT0iYmxhY2siIGQ9Ik0xMy4yOCwtODYuNDZDMTYuNDIsLTcxLjQgMjIuMDksLTQ0LjE4IDI1LjY5LC0yNi44OCIgLz4KPHBvbHlnb24gZmlsbD0iYmxhY2siIHN0cm9rZT0iYmxhY2siIHBvaW50cz0iMjcuNDMsLTI3LjExIDI2Ljc0LC0yMS44NSAyNC4wMSwtMjYuMzkgMjcuNDMsLTI3LjExIj48L3BvbHlnb24+CjwvZz4KPCEtLSBVIC0tPgo8ZyBpZD0ibm9kZTIiIGNsYXNzPSJub2RlIj4KPHRpdGxlPlU8L3RpdGxlPgo8ZWxsaXBzZSBmaWxsPSJub25lIiBzdHJva2U9ImJsYWNrIiBzdHJva2UtZGFzaGFycmF5PSI1LDIiIGN4PSIxMDEiIGN5PSItOTcuNCIgcng9IjExIiByeT0iMTEiPjwvZWxsaXBzZT4KPHRleHQgdGV4dC1hbmNob3I9Im1pZGRsZSIgeD0iMTAxIiB5PSItOTQuMSIgZm9udC1mYW1pbHk9InNlcmlmIiBmb250LXNpemU9IjExLjAwIj5VPC90ZXh0Pgo8L2c+CjwhLS0gWSAtLT4KPGcgaWQ9Im5vZGU0IiBjbGFzcz0ibm9kZSI+Cjx0aXRsZT5ZPC90aXRsZT4KPGVsbGlwc2UgZmlsbD0ibm9uZSIgc3Ryb2tlPSJibGFjayIgY3g9IjE3MyIgY3k9Ii0xMSIgcng9IjExIiByeT0iMTEiPjwvZWxsaXBzZT4KPHRleHQgdGV4dC1hbmNob3I9Im1pZGRsZSIgeD0iMTczIiB5PSItNy43IiBmb250LWZhbWlseT0ic2VyaWYiIGZvbnQtc2l6ZT0iMTEuMDAiPlk8L3RleHQ+CjwvZz4KPCEtLSBVJiM0NTsmZ3Q7WSAtLT4KPGcgaWQ9ImVkZ2UzIiBjbGFzcz0iZWRnZSI+Cjx0aXRsZT5VLSZndDtZPC90aXRsZT4KPHBhdGggZmlsbD0ibm9uZSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLWRhc2hhcnJheT0iNSwyIiBkPSJNMTA4LjE3LC04OC44QzEyMC44OSwtNzMuNTMgMTQ3LjQ4LC00MS42MyAxNjIuMzUsLTIzLjc4IiAvPgo8cG9seWdvbiBmaWxsPSJibGFjayIgc3Ryb2tlPSJibGFjayIgcG9pbnRzPSIxNjQuMDEsLTI0LjUzIDE2NS44NiwtMTkuNTcgMTYxLjMyLC0yMi4yOSAxNjQuMDEsLTI0LjUzIj48L3BvbHlnb24+CjwvZz4KPCEtLSBUaCYjNDU7Jmd0O1kgLS0+CjxnIGlkPSJlZGdlMiIgY2xhc3M9ImVkZ2UiPgo8dGl0bGU+VGgtJmd0O1k8L3RpdGxlPgo8cGF0aCBmaWxsPSJub25lIiBzdHJva2U9ImJsYWNrIiBkPSJNNDAuMywtMTFDNjUuNjQsLTExIDEyNy4xLC0xMSAxNTYuNjMsLTExIiAvPgo8cG9seWdvbiBmaWxsPSJibGFjayIgc3Ryb2tlPSJibGFjayIgcG9pbnRzPSIxNTYuNzUsLTEyLjc1IDE2MS43NSwtMTEgMTU2Ljc1LC05LjI1IDE1Ni43NSwtMTIuNzUiPjwvcG9seWdvbj4KPC9nPgo8L2c+Cjwvc3ZnPg==)

**Figure 2.** After 2SLS first stage. T is replaced by T̂ (fitted values from projecting T onto Z). The backdoor path U → T̂ is eliminated — only the exogenous variation in T driven by Z remains.

`PyFixest` estimates the IV using two-stage least squares (2SLS) estimator where it first projects \\T\\ onto \\Z\\ (and all other exogenous variables) to obtain \\\hat{T}\\, then uses \\\hat{T}\\ to estimate the causal effect of \\T\\ on \\Y\\. Because \\\hat{T}\\ is not a function of \\U\\, we can think of the dashed path from the unobserved variable as blocked or removed.

In `PyFixest`, the IV syntax is:

    Y ~ exogenous_controls | fixed_effects | endogenous ~ instrument

> **NOTE:**
>
> When panel data are available, endogeneity may also stem from time-invariant unobserved heterogeneity — unit-specific characteristics (e.g., ability, culture, geography) that are fixed over time but correlated with both treatment and outcome. `PyFixest` addresses this simultaneously by applying a within-transformation (demeaning) to absorb unit fixed effects before running 2SLS, following the FE-IV approach described in Wooldridge ([2010](#ref-wooldridge2010), Ch. 11). Crucially, after demeaning, the instrument must retain within-unit variation over time — time-invariant instruments are eliminated along with the fixed effects and cannot be used for identification. When both fixed effects and an instrument are specified, `PyFixest` therefore isolates the clean variation in treatment that is both within-unit and driven by the instrument, blocking confounding from time-invariant unobservables and time-varying endogenous confounders simultaneously.

This tutorial walks through three applications, all addressing endogeneity from **selection bias — a form of Omitted Variable Bias (OVB) where unobserved confounders drive both selection into treatment and the outcome.** Application 1 operates in an observational setting, exploiting quasi-random variation from individual-level selection. Application 2 arises in an experimental setting where encouragement is randomly assigned but treatment take-up remains subject to self-selection. Application 3 returns to an observational setting, exploiting regional-level sorting via a shift-share instrument.

``` python
import pyfixest as pf
```

## Application 1: The Motherhood Penalty

**Does having children reduce women’s earnings?**

A naive regression of earnings on fertility is biased: women with stronger career ambitions may both earn more and be less likely to have children. Since career ambition is positively correlated with earnings but negatively correlated with fertility, OLS overstates the motherhood penalty, which is the reduction in earnings mothers experience after child birth. This is a classic case of omitted variable bias (OVB): career ambition is unobserved, yet it drives both fertility decisions and earnings outcomes. In order to disentangle the true causal effect of having a child on earnings from the confounding influence of career ambition, we need to find a “quasi-random” source of variation in fertility that is independent of career ambition.

Lundborg et al. ([2017](#ref-lundborg2017)) find such an “instrument” in the quasi-random success of in-vitro fertilization (IVF) treatment. IVF success is largely determined by biological factors outside a woman’s control, making it difficult to conceive of an unobserved variable that jointly drives both the success of treatment and labor market outcomes.

### Synthetic Data

``` python
ivf_df = pf.get_ivf_data()
ivf_df.head()
```

|     | earnings  | num_children | ivf_success |
|-----|-----------|--------------|-------------|
| 0   | 7.394832  | 1.159067     | 0           |
| 1   | 12.714616 | 1.166081     | 0           |
| 2   | 9.621299  | 2.129572     | 1           |
| 3   | 11.543746 | 1.931092     | 0           |
| 4   | 9.333634  | 1.185489     | 0           |

We create a synthetic dataset with \\N = 2{,}000\\ observations. The **true causal effect** of `num_children` on `earnings` is \\\beta = -0.15\\ — this is the treatment effect the IV estimator should recover (full DGP in the [Appendix](#appendix-dgp-design-notes)).

### Naive OLS

Without accounting for endogeneity, OLS overstates the penalty because career ambition is an omitted variable that increases earnings while reducing fertility:

``` python
fit_ols = pf.feols("earnings ~ num_children", data=ivf_df)
fit_ols.summary()
```

    ###

    Estimation:  OLS
    Dep. var.: earnings
    sample: None = all
    Inference:  iid
    Observations:  2000

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | Intercept     |     10.764 |        0.059 |   181.787 |      0.000 | 10.648 |  10.880 |
    | num_children  |     -0.631 |        0.034 |   -18.441 |      0.000 | -0.698 |  -0.564 |
    ---
    RMSE: 1.126 R2: 0.145 

### IV Estimation

In the IV estimation, we use `ivf_success` as an instrument for `num_children`:

**2SLS = IV = Wald estimator**

With a single instrument \\Z\\, the 2SLS estimator is numerically identical to the IV estimator:

\\\hat{\beta}\_{IV} = \frac{\widehat{\text{Cov}}(Y,\\ Z)}{\widehat{\text{Cov}}(T,\\ Z)}\\

With a binary instrument, this simplifies to the **Wald estimator**:

\\\hat{\beta}\_{\text{Wald}} = \frac{\bar{Y}\_{Z=1} - \bar{Y}\_{Z=0}}{\bar{T}\_{Z=1} - \bar{T}\_{Z=0}}\\

In `PyFixest`, we can fit the IV model using the formula interface:

``` python
fit_iv = pf.feols("earnings ~ 1 | num_children ~ ivf_success", data=ivf_df)

pf.etable([
    fit_ols,
    fit_iv,
],
    labels={"earnings": "Earnings", "num_children": "Number of Children"},
    model_heads=["OLS", "IV"],
    caption="Motherhood Penalty: OLS vs IV",
)
```

[TABLE]

The IV estimate is closer to the true effect of -0.15. The “naive” OLS estimate is downward biased: it overstates the penalty due to omitted-variable bias from career ambition.

### First-Stage Diagnostics

In the section above, we have argued the IVF success is a credible instrument for fertility - the unconfoundedness assumption is plausible given the quasi-random nature of IVF success. But how do we know if the instrument is strong enough to yield reliable estimates? For this reason, applied econometricians routinely run diagnostic checks for “weak instruments”, which aim to validate the “relevance” assumption.

`PyFixest` provides two diagnostics for instrument strength: a standard first-stage F-statistic, and the more robust effective F-statistic from Montiel Olea and Pflueger ([2013](#ref-oleapflueger2013)) that remains valid under heteroskedasticity. Both can be accessed via the `._model_1st_stage` attribute.

``` python
# first_stage() must be called before IV_Diag() — it fits the first-stage OLS
# regression and stores the model in fit_iv._model_1st_stage, which IV_Diag() requires.
fit_iv.first_stage()
fit_iv._model_1st_stage.summary()
```

    ###

    Estimation:  OLS
    Dep. var.: num_children
    sample: None = all
    Inference:  iid
    Observations:  2000

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | Intercept     |      1.207 |        0.019 |    64.106 |      0.000 |  1.171 |   1.244 |
    | ivf_success   |      0.791 |        0.028 |    28.286 |      0.000 |  0.736 |   0.846 |
    ---
    RMSE: 0.622 R2: 0.286 

``` python
fit_iv.IV_Diag()
print(f"First-stage F-statistic : {fit_iv._f_stat_1st_stage:.2f}")
print(f"Effective F-statistic   : {fit_iv._eff_F:.2f}")
```

    First-stage F-statistic : 800.12
    Effective F-statistic   : 793.48

Both F-statistics are well above 10, which is the canonical threshold for a strong instrument. More recent work instead suggests that for reliable inference, the effective F-statistic should be significantly higher.

## Application 2: A/B Encouragement Design

**Estimating the effect of feature adoption on revenue when users don’t comply with treatment assignment.**

A tech company runs an A/B test in which half of users are *encouraged* to try a new feature, e.g. by showing them a banner. But not everyone who sees the banner actually tries out the feature. On top, some control users might discover the new feature on their own.

This setup is similar to drug trials. Suppose that medical researchers wanted to learn about the effect of taking Vitamin D on health outcomes. They could run a randomized trial in which some patients receive Vitamin D supplements for free, while control patients do not. Again, there will be imperfect compliance: some patients in the treatment group may not take their supplements, while others in the control group may take Vitamin D supplements on their own.

In such setups with imperfect compliance, one estimand of interest is the so-called **intent-to-treat (ITT)** effect, which compares outcomes between the encouraged and non-encouraged groups. However, the ITT estimates the effect of encouragement, not the effect of actual adoption. If we want to recover the effect of adoption itself, we need to use the random assignment of encouragement as an instrument for actual adoption.

All three IV assumptions are credible for this application. Relevance is satisfied because encouragement has a causal effect on adoption. The exclusion restriction is plausible because encouragement only affects revenue through adoption — there are no other channels through which seeing the banner could affect revenue. Instrumental unconfoundedness holds because encouragement is randomly assigned, so there are no unobserved confounders that jointly affect encouragement and revenue.

The only remaining assumption to discuss is **monotonicity** — the assumption that there are no “defiers” who would do the opposite of their encouragement assignment. In this context, monotonicity means that there are no users who would adopt the feature if not encouraged but would fail to adopt if encouraged.

If all assumptions hold, the IV estimate recovers the **Local Average Treatment Effect (LATE)** of adoption on revenue for the “compliers” — users whose adoption decisions are influenced by encouragement. LATE estimates are larger than the ITT because it scales up the effect of encouragement by the share of compliers in the population.

### Synthetic Data

``` python
ab_df = pf.get_encouragement_data()
ab_df.head()
```

|     | revenue  | assigned_treatment | adopted_feature | user_type |
|-----|----------|--------------------|-----------------|-----------|
| 0   | 6.791617 | 1                  | 1               | 2         |
| 1   | 7.995041 | 1                  | 1               | 2         |
| 2   | 8.204051 | 0                  | 1               | 2         |
| 3   | 4.264271 | 0                  | 0               | 1         |
| 4   | 7.980713 | 1                  | 1               | 0         |

We first create a synthetic data set with \\N = 4{,}000\\ users. The **true LATE** of `adopted_feature` on `revenue` is \\2.0\\ — this is what the IV estimate should recover.

### Three Estimands

We estimate three parameters of interest: the **reduced form** (ITT), the **first stage**, and the **IV/LATE**:

``` python
# Intent-to-treat (reduced form)
fit_itt = pf.feols("revenue ~ assigned_treatment | user_type", data=ab_df)

# First stage
fit_fs = pf.feols("adopted_feature ~ assigned_treatment | user_type", data=ab_df)

# IV / LATE
fit_late = pf.feols("revenue ~ 1 | user_type | adopted_feature ~ assigned_treatment", data=ab_df)
```

### Compare All Three

The table below places the ITT, first stage, and IV estimates side by side.

``` python
pf.etable(
    [fit_itt, fit_fs, fit_late],
    labels={
        "revenue": "Revenue",
        "adopted_feature": "Adopted Feature",
        "assigned_treatment": "Assigned Treatment",
    },
    felabels={"user_type": "User Type FE"},
    model_heads=["ITT", "First Stage", "LATE"],
    caption="A/B Encouragement Design: ITT, First Stage, and LATE",
)
```

[TABLE]

The coefficient plot compares the ITT effect of encouragement on revenue with the IV (LATE) estimate of actual feature adoption. Because not all encouraged users adopt, the LATE is larger than the ITT — scaled up by the complier share.

``` python
pf.coefplot([fit_itt, fit_late], keep="assigned_treatment|adopted_feature")
```

### IV Diagnostics

Since the instrument (`assigned_treatment`) is fully randomized, the first stage is expected to be very strong and both statistics should comfortably exceed 10.

``` python
# first_stage() must be called before IV_Diag()
fit_late.first_stage()
fit_late.IV_Diag()
print(f"First-stage F-statistic : {fit_late._f_stat_1st_stage:.2f}")
print(f"Effective F-statistic   : {fit_late._eff_F:.2f}")
```

    First-stage F-statistic : 1820.73
    Effective F-statistic   : 1832.97

## Application 3: Shift-Share (Bartik) Instruments

**Does immigration affect local wages?**

A long-standing question in labor economics is how immigration affects local wages. The challenge is that regions that attract immigrants may simultaneously have booming labor markets, biasing OLS upward: higher labor demand raises both wages and immigration, but the increased labor supply from immigration pushes wages down.

To isolate the causal effect, we need variation in immigration that is unrelated to local demand conditions. The **shift-share** (Bartik) instrument, formalized by Borusyak et al. ([2022](#ref-borusyak2022)), provides exactly this by constructing predicted local immigration from:

\\ B_r = \sum\_{k=1}^{K} s\_{rk} \cdot g_k \\

where \\s\_{rk}\\ is region \\r\\’s historical share of immigrants from origin \\k\\, and \\g_k\\ is the national inflow from origin \\k\\. The key identification assumption in Borusyak et al. ([2022](#ref-borusyak2022)) is that the *shocks* \\g_k\\ are as-good-as-randomly assigned across origin countries — uncorrelated with unobserved local labor demand in destination regions. The historical shares \\s\_{rk}\\ can themselves be endogenous (regions that historically attracted many immigrants may differ in other ways); what matters is that the national inflows that drive the instrument are exogenous.

### Synthetic Data

``` python
bartik_df = pf.get_bartik_data()
bartik_df.head()
```

|     | wages    | immigration | log_population | bartik_instrument |
|-----|----------|-------------|----------------|-------------------|
| 0   | 9.364518 | -1.905364   | 1.871830       | -1.184597         |
| 1   | 7.444654 | 0.160571    | 2.003176       | -0.354516         |
| 2   | 7.095284 | 3.199375    | 1.985372       | 1.367312          |
| 3   | 8.220384 | 1.830628    | 1.937449       | 1.700122          |
| 4   | 8.270462 | -0.135936   | 1.690355       | -1.879589         |

We create a synthetic data set with \\N = 300\\ regions. The **true causal effect** of `immigration` on `wages` is \\\beta = -0.3\\ — this is what the IV estimate should recover.

### OLS vs IV

As in the first application, we can compare the naive OLS and IV estimates.

``` python
# OLS: biased because local demand drives both immigration and wages
fit_ols_b = pf.feols("wages ~ immigration + log_population", data=bartik_df)

# IV: using the Bartik instrument
fit_iv_b = pf.feols(
    "wages ~ log_population | immigration ~ bartik_instrument",
    data=bartik_df,
)
```

``` python
pf.etable(
    [fit_ols_b, fit_iv_b],
    labels={
        "wages": "Wages",
        "immigration": "Immigration",
        "log_population": "Log Population",
    },
    model_heads=["OLS", "IV"],
    caption="Effect of Immigration on Wages: OLS vs Bartik IV",
)
```

[TABLE]

OLS attenuates the negative wage effect (or may even show a positive coefficient) because local demand is a positive confounder. The IV estimate is closer to the true effect of -0.3.

## IV Diagnostics in PyFixest

Weak instruments - instruments that are only loosely correlated with the endogenous variable - lead to biased and unreliable IV estimates. PyFixest provides two key diagnostic tools to detect this problem.

### The First-Stage F-Statistic

The `.first_stage()` method re-estimates the first-stage regression and computes the **first-stage F-statistic**, which tests \\H_0\colon \pi = 0\\ (all instrument coefficients are jointly zero). The classic rule of thumb is \\F \> 10\\ for iid errors (Stock and Yogo ([2005](#ref-stockyogo2005))).

``` python
# Re-use the motherhood penalty IV model
# Note: IV_Diag() switches vcov to hetero internally for the effective F computation.
# Reset to iid here to get the iid-based first-stage F-statistic.
fit_iv.vcov("iid")
fit_iv.first_stage()

# The F-stat is stored as an attribute after calling first_stage()
print(f"First-stage F-statistic: {fit_iv._f_stat_1st_stage:.1f}")
print(f"First-stage p-value:     {fit_iv._p_value_1st_stage:.4f}")
```

    First-stage F-statistic: 800.1
    First-stage p-value:     0.0000

### The Effective F-Statistic

The standard F-statistic can be misleading when there are multiple endogenous regressors or when errors are non-homoskedastic. The **effective F-statistic** (Montiel Olea and Pflueger ([2013](#ref-oleapflueger2013))) is a more robust measure of instrument strength that remains valid under heteroskedasticity:

\\ F\_{\text{eff}} = \frac{\hat{\pi}' Q\_{ZZ} \hat{\pi}}{\text{tr}(\hat{\Sigma} \\ Q\_{ZZ})} \\

where \\\hat{\pi}\\ are the first-stage coefficients on the excluded instruments, \\Q\_{ZZ} = Z'Z\\, and \\\hat{\Sigma}\\ is the robust variance-covariance matrix of \\\hat{\pi}\\.

The `.IV_Diag()` method computes both the standard F-statistic and the effective F-statistic in one call:

``` python
fit_iv.IV_Diag()

print(f"Standard F-statistic:  {fit_iv._f_stat_1st_stage:.1f}")
print(f"Effective F-statistic: {fit_iv._eff_F:.1f}")
```

    Standard F-statistic:  800.1
    Effective F-statistic: 793.5

> **TIP:**
>
> - [Standard Errors & Inference](../tutorials/standard-errors.llms.md) — learn about robust, cluster-robust, and bootstrap inference.
> - [Regression Tables](../tutorials/regression-tables.llms.md) — customize publication-ready output tables.
> - [`Feiv` API Reference](../reference/estimation.models.feiv_.Feiv.llms.md) — full documentation of the IV estimator class.

## References

Borusyak, Kirill, Peter Hull, and Xavier Jaravel. 2022. “Quasi-Experimental Shift-Share Research Designs.” *Review of Economic Studies* 89 (1): 181–213.

Lundborg, Petter, Erik Plug, and Astrid Würtz Rasmussen. 2017. “Can Women Have Children and a Career? IV Evidence from IVF Treatments.” *American Economic Review* 107 (6): 1611–37.

Montiel Olea, José Luis, and Carolin Pflueger. 2013. “A Robust Test for Weak Instruments.” *Journal of Business & Economic Statistics* 31 (3): 358–69.

Neal, Brady. 2020. *Introduction to Causal Inference from a Machine Learning Perspective*. [Https://www.bradyneal.com/causal-inference-course](https://www.bradyneal.com/causal-inference-course).

Stock, James H., and Motohiro Yogo. 2005. “Testing for Weak Instruments in Linear IV Regression.” In *Identification and Inference for Econometric Models: Essays in Honor of Thomas Rothenberg*, edited by Donald W. K. Andrews and James H. Stock. Cambridge University Press.

Wooldridge, Jeffrey M. 2010. *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press.
