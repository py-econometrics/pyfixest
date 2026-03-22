::: {.callout-note}
## Outline Only
This page is a structured outline, not a full textbook treatment.
:::

## Goal

Build a practical workflow for:

1. Estimating long-term treatment effects using surrogate indices.
2. Correcting external validity issues via experimental selection correction (Athey-Chetty-Imbens style framing).

Throughout, emphasize what can be implemented directly with `pyfixest`.

## 1. Problem Setup

- We observe short-run outcomes and rich covariates in an experimental sample.
- Long-run primary outcomes are expensive, delayed, or only available in another sample.
- We need a transport/selection correction when experimental and target populations differ.

Planned `pyfixest` hooks:

- `pf.feols()` for baseline outcome/treatment models.
- `pf.feglm(..., family="logit")` / `pf.feglm(..., family="probit")` for selection/propensity components.

## 2. Surrogate Index: Intuition

- Introduce the surrogate index as a lower-dimensional score built from short-run proxies.
- Clarify identifying assumptions at a high level (surrogacy and comparability conditions).
- Explain why index construction can stabilize noisy high-dimensional proxy sets.

Planned `pyfixest` implementation sketch:

- Fit outcome model(s) mapping surrogates to long-run outcome in an auxiliary/observational sample.
- Predict surrogate-based long-run signal in the experimental sample.
- Regress predicted long-run signal on treatment with appropriate fixed effects / clustering.

Minimal code placeholders to include in full write-up:

```python
# surrogate mapping (auxiliary sample)
fit_sur = pf.feols("long_run_y ~ s1 + s2 + s3 | cohort + period", data=aux)

# transport prediction to experiment
exp["y_hat_longrun"] = fit_sur.predict(newdata=exp)

# treatment effect on predicted long-run outcome
fit_te_sur = pf.feols("y_hat_longrun ~ treat | strata", data=exp, vcov={"CRV1": "strata"})
```

## 3. Surrogate Index: Practical Design Choices

- Feature set for surrogates (pre-specified vs data-driven).
- Avoiding leakage and overfitting (sample splits / cross-fitting).
- Diagnostics: first-stage fit quality, stability across splits, sensitivity to feature sets.

Planned `pyfixest` utilities:

- `pf.etable()` for side-by-side robustness summaries.
- Fixed-effects syntax `| fe1 + fe2` for panel or grouped structure.

## 4. Experimental Selection Correction (Athey-Chetty-Imbens)

- Motivation: experimental sample may not represent target policy population.
- Define selection score / reweighting logic in outline form.
- Show how correction changes estimand from experimental ATE to target-population ATE.

Planned `pyfixest` implementation sketch:

- Estimate selection propensity (in-experiment vs target-sample indicator) with logit/probit.
- Construct inverse-odds / transport weights.
- Re-estimate treatment effect under transport weights.

Minimal code placeholders to include in full write-up:

```python
# selection model
fit_sel = pf.feglm("in_experiment ~ x1 + x2 + x3 | region", data=stacked, family="logit")
stacked["p_exp"] = fit_sel.predict(type="response")

# transport weights for experiment units
exp["w_transport"] = (1 - exp["p_exp"]) / exp["p_exp"]

# weighted transport-adjusted treatment effect
fit_transport = pf.feols(
    "outcome ~ treat | strata",
    data=exp,
    weights="w_transport",
    vcov={"CRV1": "strata"},
)
```

## 5. Joint Workflow: Surrogates + Selection Correction

- Combine both components:
  - Surrogate model for long-run prediction.
  - Selection correction for target-population transport.
- Report:
  - Naive experimental estimate.
  - Surrogate-adjusted estimate.
  - Surrogate + selection-corrected estimate.

Planned `pyfixest` output:

- Single comparison table via `pf.etable([fit_naive, fit_te_sur, fit_transport], ...)`.

## 6. Inference and Robustness (Outline)

- Cluster choice (design-based clustering level).
- Small-sample inference options when clusters are few (`wildboottest` on final stage where supported).
- Sensitivity checks:
  - Alternative surrogate sets.
  - Alternative selection models (logit vs probit).
  - Trimming/extreme-weight handling.

## 7. Limitations and Assumptions

- Surrogacy assumptions are substantive and testable only indirectly.
- Selection correction depends on observed covariate overlap.
- Extrapolation risk under weak support.

## 8. Planned References

- Athey, Chetty, Imbens (experimental selection/external validity framing).
- Surrogate index / surrogate score papers in program-evaluation and policy-learning literatures.
