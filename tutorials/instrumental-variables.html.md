::: {.callout-note}
## Prerequisites
You should have read the [Getting Started](../getting-started.qmd) page and have `pyfixest` installed.
:::

## Introduction

Ordinary Least Squares (OLS) delivers biased estimates when a regressor is correlated with the error term --- due to omitted variables, simultaneity, or measurement error. **Instrumental Variables (IV)** estimation solves this by finding a variable $Z$ that:

1. **Relevance**: $Z$ is correlated with the endogenous regressor $X$.
2. **Exclusion**: $Z$ affects the outcome $Y$ *only through* $X$.

The two-stage least squares (2SLS) estimator first regresses $X$ on $Z$ (and controls), then uses the predicted $\hat{X}$ in the outcome equation. In `pyfixest`, the IV syntax is:

```
Y ~ exogenous_controls | fixed_effects | endogenous ~ instrument
```

This tutorial walks through three applications --- each highlighting a different reason researchers reach for IV.

```python
import pyfixest as pf
```


## Application 1: The Motherhood Penalty

**Does having children reduce women's earnings?**

A naive regression of earnings on fertility is biased: women with stronger career ambitions may both earn more *and* have fewer children. Since career ambition is positively correlated with earnings but negatively correlated with fertility, OLS *overstates* the motherhood penalty. Lundborg, Plug & Rasmussen (2017) exploit the quasi-random success of IVF treatment as an instrument for fertility among women who sought treatment.

### Synthetic Data

```python
ivf_df = pf.get_ivf_data()
ivf_df.head()
```



### Naive OLS

Without accounting for endogeneity, OLS overstates the penalty because career ambition is an omitted variable that increases earnings while reducing fertility:

```python
fit_ols = pf.feols("earnings ~ num_children", data=ivf_df)
fit_ols.summary()
```





### IV Estimation

Using `ivf_success` as an instrument for `num_children`:

```python
fit_iv = pf.feols("earnings ~ 1 | num_children ~ ivf_success", data=ivf_df)
fit_iv.summary()
```



The IV estimate is closer to the true effect of -0.15 --- less negative than the OLS estimate, which overstates the penalty due to omitted-variable bias from career ambition.

### Compare OLS and IV

```python
pf.etable(
    [fit_ols, fit_iv],
    labels={"earnings": "Earnings", "num_children": "Number of Children"},
    caption="Motherhood Penalty: OLS vs IV",
)
```



### First-Stage Diagnostics

A strong first stage is essential for IV to work. We check the first-stage F-statistic and run `IV_Diag()`:

```python
fit_iv.first_stage()
```


```python
fit_iv.IV_Diag()
```


The first-stage F-statistic should be well above 10, confirming that IVF success is a strong predictor of fertility.


## Application 2: A/B Encouragement Design

**Estimating the effect of feature adoption on revenue when users don't comply with treatment assignment.**

A tech company runs an A/B test: half of users are *encouraged* (shown a banner) to try a new feature. But not everyone who sees the banner actually adopts the feature, and some control users discover it on their own. The simple intent-to-treat (ITT) comparison underestimates the effect on users who actually adopt. Random assignment serves as an instrument for actual adoption, and the IV estimate recovers the **Local Average Treatment Effect (LATE)** on compliers.

### Synthetic Data

```python
ab_df = pf.get_encouragement_data()
ab_df.head()
```



### Three Estimands

We estimate the **reduced form** (ITT), the **first stage**, and the **IV/LATE**:

```python
# Intent-to-treat (reduced form)
fit_itt = pf.feols("revenue ~ assigned_treatment | user_type", data=ab_df)

# First stage
fit_fs = pf.feols("adopted_feature ~ assigned_treatment | user_type", data=ab_df)

# IV / LATE
fit_late = pf.feols("revenue ~ 1 | user_type | adopted_feature ~ assigned_treatment", data=ab_df)
```


The Wald estimator says: $\text{LATE} = \frac{\text{ITT}}{\text{First Stage}} = \frac{\text{Cov}(Y, Z)}{\text{Cov}(D, Z)}$.

Let's verify:

```python
itt_coef = fit_itt.coef()["assigned_treatment"]
fs_coef = fit_fs.coef()["assigned_treatment"]
late_coef = fit_late.coef()["adopted_feature"]

print(f"ITT coefficient:         {itt_coef:.4f}")
print(f"First-stage coefficient: {fs_coef:.4f}")
print(f"Wald ratio (ITT / FS):   {itt_coef / fs_coef:.4f}")
print(f"IV/LATE coefficient:     {late_coef:.4f}")
```



### Compare All Three

```python
pf.etable(
    [fit_itt, fit_fs, fit_late],
    labels={
        "revenue": "Revenue",
        "adopted_feature": "Adopted Feature",
        "assigned_treatment": "Assigned Treatment",
    },
    felabels={"user_type": "User Type FE"},
    caption="A/B Encouragement Design: ITT, First Stage, and LATE",
)
```



```python
pf.coefplot([fit_itt, fit_late], keep="assigned_treatment|adopted_feature")
```



### IV Diagnostics

```python
fit_late.IV_Diag()
```


## Application 3: Shift-Share (Bartik) Instruments

**Does immigration affect local wages?**

A long-standing question in labor economics. The challenge: regions that attract immigrants may also have booming labor markets, biasing OLS upward. The **shift-share** (Bartik) instrument, formalized by Borusyak, Hull & Jaravel (2022), constructs predicted local immigration from:

$$
B_r = \sum_{k=1}^{K} s_{rk} \cdot g_k
$$

where $s_{rk}$ is region $r$'s historical share of immigrants from origin $k$, and $g_k$ is the national inflow from origin $k$. Because the instrument is constructed from *national* shocks interacted with *historical* shares, it is plausibly exogenous to current local labor demand.

### Synthetic Data

```python
bartik_df = pf.get_bartik_data()
bartik_df.head()
```



### OLS vs IV

```python
# OLS: biased because local demand drives both immigration and wages
fit_ols_b = pf.feols("wages ~ immigration + log_population", data=bartik_df)

# IV: using the Bartik instrument
fit_iv_b = pf.feols(
    "wages ~ log_population | immigration ~ bartik_instrument",
    data=bartik_df,
)
```


```python
pf.etable(
    [fit_ols_b, fit_iv_b],
    labels={
        "wages": "Wages",
        "immigration": "Immigration",
        "log_population": "Log Population",
    },
    caption="Effect of Immigration on Wages: OLS vs Bartik IV",
)
```



OLS attenuates the negative wage effect (or may even show a positive coefficient) because local demand is a positive confounder. The IV estimate is closer to the true effect of -0.3.

### Diagnostics

```python
fit_iv_b.first_stage()
```


```python
fit_iv_b.IV_Diag()
```


```python
pf.coefplot([fit_ols_b, fit_iv_b], keep="immigration")
```



## IV Diagnostics in PyFixest

Weak instruments --- instruments that are only loosely correlated with the endogenous variable --- lead to biased and unreliable IV estimates. PyFixest provides two key diagnostic tools to detect this problem.

### The First-Stage F-Statistic

The `.first_stage()` method re-estimates the first-stage regression and computes the **first-stage F-statistic**, which tests $H_0\colon \pi = 0$ (all instrument coefficients are jointly zero). The classic rule of thumb is $F > 10$ (Stock & Yogo, 2005).

The F-statistic adapts to your variance-covariance specification: if you fit the IV model with heteroskedasticity-robust or cluster-robust standard errors, the first-stage F-statistic is computed accordingly.

```python
# Re-use the motherhood penalty IV model
fit_iv.first_stage()

# The F-stat is stored as an attribute after calling first_stage()
print(f"First-stage F-statistic: {fit_iv._f_stat_1st_stage:.1f}")
print(f"First-stage p-value:     {fit_iv._p_value_1st_stage:.4f}")
```



### The Effective F-Statistic

The standard F-statistic can be misleading when there are multiple endogenous regressors or when errors are non-homoskedastic. The **effective F-statistic** (Olea & Pflueger, 2013) is a more robust measure of instrument strength that remains valid under heteroskedasticity:

$$
F_{\text{eff}} = \frac{\hat{\pi}' Q_{ZZ} \hat{\pi}}{\text{tr}(\hat{\Sigma} \, Q_{ZZ})}
$$

where $\hat{\pi}$ are the first-stage coefficients on the excluded instruments, $Q_{ZZ} = Z'Z$, and $\hat{\Sigma}$ is the robust variance-covariance matrix of $\hat{\pi}$.

The `.IV_Diag()` method computes both the standard F-statistic and the effective F-statistic in one call:

```python
fit_iv.IV_Diag()

print(f"Standard F-statistic:  {fit_iv._f_stat_1st_stage:.1f}")
print(f"Effective F-statistic: {fit_iv._eff_F:.1f}")
```



::: {.callout-tip}
## When to worry
- **$F < 10$**: instruments are likely too weak. Consider finding stronger instruments or using weak-instrument-robust inference.
- **Large gap between standard and effective F**: suggests that heteroskedasticity matters for instrument strength. Rely on the effective F in this case.
- **$F_{\text{eff}} \gg 10$**: strong instruments under heteroskedasticity --- you're in good shape.
:::


::: {.callout-tip}
## Next Steps
- [Standard Errors & Inference](standard-errors.qmd) --- learn about robust, cluster-robust, and bootstrap inference.
- [Regression Tables](regression-tables.qmd) --- customize publication-ready output tables.
- [`Feiv` API Reference](../reference/estimation.models.feiv_.Feiv.qmd) --- full documentation of the IV estimator class.
:::
