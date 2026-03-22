::: {.callout-note}
## Scope
This vignette is practical by design. We focus on how to use `pyfixest` and when these models are useful in applied work.
:::

## 1. Why Poisson? (and Why Wooldridge Likes It)

A common motivation for Poisson pseudo-maximum-likelihood (PPML) is that it works well for **nonnegative outcomes** and remains valid under fairly general forms of heteroskedasticity when the conditional mean is correctly specified.

In practice, that makes Poisson attractive for many applied settings where:

- outcomes are counts or nonnegative flows,
- zeros are common,
- log-linear OLS can be fragile.

`pyfixest` provides PPML with high-dimensional fixed effects via `pf.fepois()`.

## 2. A Standard Gravity-Style Trade Example (Boring but Useful)

Below is a simple gravity-style trade panel (exporter-importer-year). This is a synthetic dataset, but the specification mirrors common trade applications.

```{python}
import numpy as np
import pandas as pd
import pyfixest as pf
```

```{python}
def make_trade_data(seed=123, n_exporters=15, n_importers=15, years=range(2010, 2016)):
    rng = np.random.default_rng(seed)

    exporters = [f"E{i}" for i in range(n_exporters)]
    importers = [f"I{j}" for j in range(n_importers)]

    rows = []
    for y in years:
        for e in exporters:
            for i in importers:
                dist = rng.uniform(200, 9000)
                fta = rng.binomial(1, 0.15)
                exporter_fe = hash(e) % 7
                importer_fe = hash(i) % 6
                year_fe = y - min(years)

                # mean function for PPML
                eta = 2.5 - 0.35 * np.log(dist) + 0.25 * fta + 0.08 * exporter_fe + 0.06 * importer_fe + 0.05 * year_fe
                mu = np.exp(eta)
                trade = rng.poisson(mu)

                rows.append((y, e, i, dist, fta, trade))

    return pd.DataFrame(rows, columns=["year", "exporter", "importer", "distance", "fta", "trade"])

trade = make_trade_data()
trade.head()
```

```{python}
fit_trade = pf.fepois(
    "trade ~ np.log(distance) + fta | exporter + importer + year",
    data=trade,
    vcov={"CRV1": "exporter"},
)
fit_trade.summary()
```

This is the basic PPML gravity workflow in `pyfixest`.

## 3. Handling Zeros: "The Log of Zeros"

One reason PPML is popular is straightforward handling of zero outcomes without ad-hoc transformations like `log(y + 1)`.

This is central in modern work on nonnegative outcomes, including recent discussion in:

- Bellégo, Benatia, and Pape (2024), *Dealing with Logs and Zeros in Regression Models* ([QJE](https://academic.oup.com/qje/article-abstract/139/2/891/7473710)).

Quick contrast:

```{python}
share_zeros = (trade["trade"] == 0).mean()
share_zeros
```

With PPML, zero observations are naturally part of the likelihood through the mean function.

## 4. Poisson for a Simple DiD

Poisson DiD is useful when the outcome is a count or nonnegative rate and treatment is staggered or panel-based.

```{python}
def make_count_did(seed=42, n_states=30, years=range(2010, 2018)):
    rng = np.random.default_rng(seed)
    states = [f"S{s}" for s in range(n_states)]

    rows = []
    treated_states = set(states[: n_states // 2])
    post_start = 2014

    for y in years:
        for s in states:
            treated = int(s in treated_states)
            post = int(y >= post_start)
            did = treated * post

            # latent state/year effects + treatment effect in post
            state_fe = (hash(s) % 9) / 10
            year_fe = (y - min(years)) / 10
            eta = 1.2 + state_fe + year_fe + 0.20 * did
            mu = np.exp(eta)
            y_count = rng.poisson(mu)

            rows.append((s, y, treated, post, did, y_count))

    return pd.DataFrame(rows, columns=["state", "year", "treated", "post", "did", "y_count"])

did_df = make_count_did()

fit_did_pois = pf.fepois(
    "y_count ~ did | state + year",
    data=did_df,
    vcov={"CRV1": "state"},
)
fit_did_pois.summary()
```

If your outcome is nonnegative and has many zeros, this can be a practical alternative to log-linear OLS DiD.

## 5. Logit/Probit for Doubly Debiased AB Testing with Propensity Scores

For observational AB tests (or experiments with imperfect randomization), a common strategy is to combine:

- a **propensity score model** `P(T=1|X)`, and
- an **outcome model** `E[Y|T,X]`.

This yields doubly robust / doubly debiased estimators (AIPW-style scores).

### Simulate an AB setting

```{python}
def make_ab_data(seed=1, n=5000):
    rng = np.random.default_rng(seed)

    segment = rng.integers(0, 20, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)

    # non-random treatment assignment
    p = 1 / (1 + np.exp(-(-0.2 + 0.7 * x1 - 0.4 * x2 + 0.15 * (segment % 4))))
    t = rng.binomial(1, p)

    # outcome with true effect tau
    tau = 0.25
    y = tau * t + 0.6 * x1 + 0.3 * x2 + 0.2 * (segment % 5) + rng.normal(scale=1.0, size=n)

    return pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2, "segment": segment})

ab = make_ab_data()
ab.head()
```

### Propensity via logit/probit (with fixed effects)

```{python}
# both support fixed effects via the | syntax
ps_logit = pf.feglm("t ~ x1 + x2 | segment", data=ab, family="logit")
ps_probit = pf.feglm("t ~ x1 + x2 | segment", data=ab, family="probit")

ab["ehat_logit"] = np.clip(ps_logit.predict(type="response"), 0.01, 0.99)
ab["ehat_probit"] = np.clip(ps_probit.predict(type="response"), 0.01, 0.99)
```

### Outcome regressions and a simple AIPW score

```{python}
mu1 = pf.feols("y ~ x1 + x2 | segment", data=ab.loc[ab["t"] == 1]).predict(newdata=ab)
mu0 = pf.feols("y ~ x1 + x2 | segment", data=ab.loc[ab["t"] == 0]).predict(newdata=ab)

ab["mu1"] = mu1
ab["mu0"] = mu0

# AIPW score for ATE using logit propensity
ab["score_aipw_logit"] = (
    ab["mu1"] - ab["mu0"]
    + ab["t"] * (ab["y"] - ab["mu1"]) / ab["ehat_logit"]
    - (1 - ab["t"]) * (ab["y"] - ab["mu0"]) / (1 - ab["ehat_logit"])
)

ate_aipw_logit = ab["score_aipw_logit"].mean()
ate_aipw_logit
```

This is the basic doubly debiased pattern: estimate treatment propensities with `feglm` (logit/probit), combine with outcome models, and form an orthogonal score.

## 6. GLMs with Marginal Effects (Placeholder)

::: {.callout-note}
## Coming Soon
This section will add a full workflow for marginal effects after `feglm()` estimation, including:

- average marginal effects (AMEs),
- subgroup/conditional marginal effects,
- inference with robust and clustered standard errors,
- comparison of coefficient-scale vs probability-scale interpretation.
:::

Planned examples:

- Binary outcome with `family="logit"` and `family="probit"`.
- Post-estimation marginal effects tables suitable for reporting.
- Practical guidance on when marginal effects change substantive conclusions relative to raw GLM coefficients.

## 7. Fixed Effects Support (Summary)

Both Poisson and GLM estimators in `pyfixest` support fixed effects using the same formula syntax:

- Poisson: `pf.fepois("y ~ x | fe1 + fe2", ...)`
- GLM families (`logit`, `probit`, `gaussian`): `pf.feglm("y ~ x | fe1 + fe2", family="logit", ...)`

That shared syntax makes it easy to move between linear, Poisson, and binary-response workflows while keeping your FE structure aligned.

## Where to Go Next

- [Difference-in-Differences](difference-in-differences.qmd): richer DiD estimators and event studies.
- [Standard Errors & Inference](standard-errors.qmd): clustered inference, randomization inference, and wild bootstrap.
- [How-To: Variance Reduction in AB Tests](../how-to/panel_variance_reduction.qmd): AB testing workflows with controls and panel structure.
