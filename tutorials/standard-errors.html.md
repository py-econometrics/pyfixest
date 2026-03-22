::: {.callout-note}
## Outline
This page is a structured outline for a full inference tutorial.
:::

## 1. Why Inference Matters

- Estimation answers "what is the effect estimate?"
- Inference answers "how uncertain is this estimate?"

We distinguish two broad frameworks:

- **Model-based inference**: uncertainty is derived from assumptions on the error process (iid, heteroskedasticity, clustering).
- **Design-based inference**: uncertainty is derived from the assignment mechanism (e.g. randomization inference).

In practice, choose the inference approach that matches how variation is generated in your data.

## 2. Baseline: IID Errors

The simplest benchmark assumes errors are independent and identically distributed.

- Independence: no cross-observation dependence in errors.
- Identical variance: constant error variance across observations.

```python
import pyfixest as pf

data = pf.get_data()
fit_iid = pf.feols("Y ~ X1 + X2 | f1", data=data, vcov="iid")
fit_iid.summary()
```

Use this as a baseline, not as a default in grouped/panel contexts.

## 3. Heteroskedasticity-Robust Inference

If error variance differs across observations, iid standard errors can be misleading.

Common robust choices:

- `HC1` / `"hetero"`: common default.
- `HC2`, `HC3`: often preferred in smaller samples.

```python
fit_hc1 = pf.feols("Y ~ X1 + X2 | f1", data=data, vcov="HC1")
fit_hc3 = pf.feols("Y ~ X1 + X2 | f1", data=data, vcov="HC3")

pf.etable([fit_iid, fit_hc1, fit_hc3])
```

## 4. Cluster-Robust Inference

If errors are correlated within groups (states, firms, schools, users), use cluster-robust inference.

```python
fit_crv1 = pf.feols("Y ~ X1 + X2 | f1", data=data, vcov={"CRV1": "group_id"})
fit_crv3 = pf.feols("Y ~ X1 + X2 | f1", data=data, vcov={"CRV3": "group_id"})
```

Multi-way clustering is supported as well:

```python
fit_twoway = pf.feols(
    "Y ~ X1 + X2 | f1",
    data=data,
    vcov={"CRV1": "f1 + f2"},
)
```

## 5. Which Level Should You Cluster On?

Core principle: cluster at the level where shocks or treatment assignment induce dependence.

Practical checklist:

1. At what level is treatment assigned?
2. At what level can residual shocks be correlated?
3. Is there serial correlation over time within units?

Common mistakes:

- Clustering too finely and missing true dependence.
- Clustering on arbitrary identifiers unrelated to assignment/dependence.

## 6. Fisherian Randomization Inference

Randomization inference (RI) is design-based:

- It uses the assignment mechanism directly.
- It can be compelling in finite samples and experimental settings.

In PyFixest:

```python
# syntax depends on assignment structure and test setup
# shown here as a placeholder call pattern
ri_res = fit_crv1.ritest(resampvar="X1=0", reps=1000)
ri_res
```

Use RI when random assignment is well-defined and central to identification.

## 7. Why Wild Cluster Bootstrap?

Asymptotic cluster-robust methods can over-reject when:

- The number of clusters `G` is small.
- Cluster sizes are highly unbalanced.

Wild cluster bootstrap addresses this by resampling at the cluster level in a way that is more reliable in these settings.

```python
wcb_res = fit_crv1.wildboottest(param="X1", reps=9999, cluster="group_id")
wcb_res
```

## 8. Simulation Study: `feols` + `wildboottest`

Goal: compare rejection rates under the null (`beta = 0`) for:

- cluster-robust asymptotic p-values
- wild cluster bootstrap p-values

Design sketch:

1. Simulate clustered data with no true treatment effect.
2. Fit `feols` with clustered vcov.
3. Test `H0: beta = 0` using conventional CRV p-values.
4. Test the same null via `wildboottest`.
5. Repeat many times and compare empirical rejection rates.

```python
import numpy as np
import pandas as pd


def sim_once(seed, G=20, min_n=20, max_n=300):
    rng = np.random.default_rng(seed)
    cluster_sizes = rng.integers(min_n, max_n + 1, size=G)
    g = np.repeat(np.arange(G), cluster_sizes)
    n = g.size

    x = rng.normal(size=n)
    u_g = rng.normal(scale=1.0, size=G)[g]          # cluster shock
    e = rng.normal(scale=1.0, size=n)               # idiosyncratic shock
    y = 1.0 + 0.0 * x + u_g + e                     # true beta = 0

    df = pd.DataFrame({"y": y, "x": x, "g": g})
    fit = pf.feols("y ~ x", data=df, vcov={"CRV1": "g"})

    p_crv = fit.pvalue().loc["x"]
    p_wcb = fit.wildboottest(param="x", reps=999, cluster="g").loc["Pr(>|t|)"]
    return p_crv, p_wcb


def sim_rejection_rate(B=200, alpha=0.05):
    out = [sim_once(seed=i) for i in range(B)]
    p_crv = np.array([x[0] for x in out])
    p_wcb = np.array([x[1] for x in out])
    return {
        "CRV1 reject rate": float((p_crv < alpha).mean()),
        "WCB reject rate": float((p_wcb < alpha).mean()),
    }
```

Interpretation target:

- If `CRV1` reject rate is above nominal size (e.g. > 0.05), inference is too liberal.
- If `WCB` is closer to nominal size, bootstrap improves finite-sample reliability.

## 9. Concluding Inference Workflow

Suggested workflow:

1. Start from design and dependence structure.
2. Use robust/clustered vcov aligned with that structure.
3. If clusters are few or very uneven, use wild cluster bootstrap.
4. If assignment is randomized, add randomization inference as a design-based check.

## 10. Multiple Testing

When testing many hypotheses, single-test p-values are not enough.
Multiple-testing corrections and workflow guidance are covered in a dedicated
resource and should be applied when moving from a single estimand to a family
of hypotheses.
