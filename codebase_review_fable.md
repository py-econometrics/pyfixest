# PyFixest codebase review

Scope: the `pyfixest/estimation` core (models, internals, formula, plan/runner), plus the pieces of `utils` and `report` they touch. Perspective: senior Python engineer; outer API stays fixed, internals may change. Everything marked "confirmed" was reproduced by running the code in this working tree; where noted, the same failure reproduces on the released 0.60.0 wheel, so it is a long-standing issue rather than a regression from the current branch.

Summary: the recent refactor (config → plan → runner, the `demeaners` strategy dataclasses, `fit_glm_irls`, `DemeanCache`, the Rust core boundary) is genuinely good — clear ownership, frozen dataclasses, pure functions. The main WLS/aweights/fweights numerics are correct (I cross-checked coefficients and HC1 SEs against statsmodels, and fweights against row-expansion; both match to machine precision). The problems are concentrated in `Feols`, which is a 2,500-line god class whose arrays silently change meaning as the fit pipeline runs, and in the weights wiring, where one attribute name (`_weights`) carries three different meanings depending on model type and pipeline stage.

---

## 1. Confirmed bugs

### B1. Clustering by interacted fixed effects crashes

`pf.feols("Y ~ X1 | f1^f2", data, vcov={"CRV1": "f1^f2"})` raises `KeyError: "None of [Index(['f1^f2'] ...)] are in the [columns]"`. Also fails on 0.60.0.

Root cause is twofold. In `_deparse_vcov_input` (models/feols_.py:2484):

```python
if clustervar and "^" in clustervar:
```

`clustervar` is a *list*, so this tests whether some element *equals* `"^"` — it never fires for real names like `"f1^f2"`. And even with the intended `any("^" in x for x in clustervar)`, nothing creates the interacted column in `_data` anymore (the docstring for `copy_data` still claims "you'll find a column with interacted fixed effects in the data set", which suggests the old model-matrix path created it and the formula refactor dropped that side effect). Fix: materialize the interaction column in `prepare_cluster_state` (it already handles `^` → `_` renaming for nested-FE counting), and delete the dead membership test.

### B2. `vcov` as a list crashes with `UnboundLocalError`

`_check_vcov_input` (models/feols_.py:2378, 2390–2394) explicitly validates list inputs ("vcov list must contain columns in the data"), but `_deparse_vcov_input` then falls through every `elif` — for a list, `vcov_type_detail in ["iid", ...]` is always False — and dies on unbound `is_clustered`. Confirmed: `fit.vcov(["f1", "f2"])` → `UnboundLocalError`. Either support lists (natural reading: multiway CRV1) or reject them up front; the current state validates then crashes. A trailing `else: raise ValueError(...)` in the deparse chain would have caught this class of bug.

### B3. `fepois`/`feglm` with `weights_type="fweights"` crashes

Confirmed: `ValueError: cannot reshape array of size 997 into shape (2034,1)` (also broken on 0.60.0, with a different error). `Feglm.get_fit` (models/feglm_.py:198) does `self._weights.reshape((self._N, 1))`, but for fweights `_N` is the weight *sum*, not the row count — `_N_rows` is the right variable. The separation-drop branch (feglm_.py:134) has the mirrored bug: it sets `self._N = self._Y.shape[0]`, silently redefining `_N` to row count even for fweights. Decide whether GLMs support fweights; if yes, fix both spots and add an fweights-vs-expanded-rows equivalence test; if no, reject in the API with a clear error instead of a reshape crash.

### B4. `update()` silently returns wrong results for weighted models

`update()` (models/feols_.py:2284) appends raw `X_new` rows to `self._X`, which after `wls_transform()` is the sqrt(w)-scaled matrix, and ignores weights for the new rows entirely. Confirmed numerically: for a WLS fit, `update()` gives [0.9292, −1.0157] where the true refit gives [0.9192, −1.0113]; for unweighted models it is exact. There is already a guard for `_has_fixef`; add the same `NotImplementedError` for `_has_weights` (or handle weights properly). Two smaller things in the same method: it claims Sherman–Morrison but re-inverts `X'X` from scratch, and `inplace=True` leaves `_u_hat` in the raw domain while the rest of the object expects the WLS domain.

### B5. `Feglm.predict` / `Fepois.predict` silently swallow `interval` and `alpha`

Both overrides accept `interval="prediction"` in their signature, then forward only `newdata`, `type`, `atol`, `btol` to `super().predict(...)` (feglm_.py:343, fepois_.py:244). Confirmed: `fepois(...).predict(interval="prediction")` returns a plain ndarray — no error, no intervals. `se_fit=True` raises properly; `interval` should raise the same `NotImplementedError`.

### B6. `wald_test(distribution="F")` silently becomes a chi2 test for any custom restriction

models/feols_.py:1076–1092. The condition `not np.array_equal(R, np.eye(k)) or not np.all(q == 0)` coerces *every* non-trivial hypothesis to chi2, so the F distribution is only ever used for the default joint-zero test — asking for an F test of `R = [[0,1,0]]` gets you a chi2 p-value plus a warning whose text also misstates the condition ("and" vs the actual "or"). fixest computes proper F tests for arbitrary R. If the small-sample F for general restrictions is intentionally unsupported, raise instead of warning-and-substituting; silently changing the distribution is the worst of both options. Also, the chi2 branch still assigns `self._f_statistic`, which is misleading.

### B7. `wildboottest()` prints instead of raising on missing dependency

models/feols_.py:1232–1237: `except ImportError: print(...)` — then execution continues and dies with `NameError: WildboottestCL is not defined`. Raise an `ImportError` with the install hint (the ritest module does this correctly).

### B8. Dead, contradictory flag logic in `Feols.__init__`

models/feols_.py:349–354:

```python
if self._weights_name is not None:
    self._supports_wildboottest = False
self._supports_wildboottest = True          # overwrites the line above
if self._has_weights or self._is_iv:
    self._supports_wildboottest = False
```

The final state happens to be right; the first two lines are dead and confuse readers about which condition governs. Delete the first assignment.

### B9. Unreachable code in `ccv()`

models/feols_.py:1495–1508: ~14 lines after `return pd.concat(...)` are dead (a second `_ccv` code path that can never run). Delete.

### B10. `get_ssc`'s `"HAC-TS"` branch is unreachable

utils/utils.py:230: `df_t = N - df_k if vcov_type in ["iid", "hetero", "HAC-TS"] else G - 1`, but no caller ever passes `"HAC-TS"` — `Feols.vcov` passes `"HAC"` for both time-series NW and panel NW/DK. So time-series Newey–West gets `df_t = T − 1`. The stranded string suggests the intent was `N − k` degrees of freedom for pure time-series HAC; worth checking against fixest and either wiring the distinction through or deleting the dead token.

### B11. `Feiv` demeans the endogenous variable, then throws the result away

`Feiv.demean` stores the demeaned endogenous variable in `self._endogvard`; `Feiv.to_array` (feiv_.py:203) then converts `self._endogvar` — the *un-demeaned* frame — and `wls_transform` scales that by sqrt(w). `_endogvard` is never read again, and `_endogvar` is never consumed after the scaling (the second-stage X already contains the endogenous column). So today this is dead state; the trap is that `_endogvar` looks like "demeaned + weighted endogenous variable" and is neither. Either fix `to_array` to use `_endogvard` or delete both attributes after the demean step.

### B12. `eff_F()` mutates the second stage's vcov bookkeeping

feiv_.py:476–478: when the model was fit with iid errors, `eff_F` sets `self._vcov_type_detail = "hetero"` on the *Feiv* object (the second stage) and refits only the first-stage vcov. After calling `IV_Diag()`, the main model reports "hetero" inference while `_vcov` is still the iid matrix. Use a local variable for the first-stage choice; don't touch second-stage state.

---

## 2. The weights problem

You flagged this, and it is real. Three distinct concepts share the name `_weights`, and arrays flip between the raw and the sqrt(w)-scaled ("WLS domain") representation without the name changing.

What `_weights` can mean today: user-supplied weights (`Feols` after `prepare_model_matrix`), ones (unweighted fit), or final IRLS working weights — because `Feglm.get_fit` (feglm_.py:195) *overwrites* `self._weights = fit.W`. The clearest symptom of the design is `Fepois.get_fit` (fepois_.py:140), which must rescue `user_weights = self._weights.flatten().copy()` *before* calling `super().get_fit()` clobbers the attribute. Any consumer of `_weights` on a GLM gets working weights whether it wants them or not: `fixef()` (feols_.py:1770, 1817) weights the FE recovery by sqrt(IRLS W) when user weights are present but does an unweighted solve when they are not — two different estimators selected by an unrelated flag; `vcov_hetero`'s fweights correction (internals/vcov_.py:73–75) would divide by IRLS weights instead of frequency weights (currently unreachable only because B3 crashes first).

Meanwhile, the "which domain is this array in" question has no single answer. After `get_fit()`, `_Y`, `_X`, `_Z`, `_u_hat`, `_scores` are all in the WLS domain; `_Y_untransformed` is raw; `_X_untransformed` is demeaned-but-unweighted; `resid()` un-weights `_u_hat` on the fly; `_model_matrix_one_hot` (feols_.py:1546) un-weights `self._X` by dividing by sqrt(w) even though `_X_untransformed` holds exactly that array — two parallel mechanisms for the same recovery. `predict()` carries a stale comment ("divide by sqrt(weights) as self._X is 'weighted'", feols_.py:1919) above code that doesn't divide. Every consumer has to privately know the convention, and the convention differs by class.

The math is currently right in the supported paths — I verified WLS/HC1 against statsmodels and fweights against row expansion — but B3, B4, B11 and the stale comments are all downstream of the same ambiguity, and the next contributor will add another.

### Proposed redesign (deliberately minimal)

First, one invariant: `_weights` always means user weights (ones if none), set once in `prepare_model_matrix` and never reassigned. `Feglm` already stores `_irls_weights = fit.W`; make that the *only* home for working weights and fix the handful of GLM consumers (`fixef`, `resid`-adjacent code, vcov plumbing) to name which one they want. This deletes the rescue-copy dance in `Fepois.get_fit` and is a small, mechanical change.

Second, make the WLS transform produce new names instead of mutating in place. Today `get_fit` runs `demean() → to_array() → drop_multicol_vars() → wls_transform()`, each stage overwriting `self._Y`/`self._X` with something semantically different (DataFrame → demeaned DataFrame → ndarray → collinearity-pruned → sqrt(w)-scaled). Suggestion: keep `_Y`/`_X` as the demeaned, unweighted arrays (which also lets you delete `_X_untransformed`), and store the transformed pair under explicit names, e.g. `_Y_wls`/`_X_wls`, fed to `fit_ols`. The rename is grep-driven and forces every consumer — vcov, bootstrap, decompose, predict — to state which domain it needs; ambiguity becomes a `AttributeError` at the call site instead of a silent wrong number. Add a short "domain map" table to the `Feols` class docstring (attribute → raw / demeaned / WLS domain → where set) and prune the current 150-line attribute list, which has drifted from reality.

Third, lock it in with tests that encode the invariants rather than specific numbers: fweights vs expanded-rows equivalence for feols and fepois across iid/HC1/CRV1; aweights vs statsmodels WLS; `predict()`, `resid()`, `fixef()` returning raw-domain quantities under weights; and `model._weights is user weights` after a GLM fit. The expansion test alone would have caught B3.

What I would *not* do: no weights wrapper class, no per-array domain enum, no immutable model-state object. The two renames plus tests get 90% of the safety for 10% of the churn.

---

## 3. Design-level code smells

`Feols` is a god class. 2,500 lines covering estimation, four vcov families, Wald tests, wild bootstrap, CCV, randomization inference, Gelbach decomposition, FE recovery, prediction, and Sherman–Morrison updates. The extraction pattern already in the codebase (thin method delegating to `post_estimation/` module, as `ritest` mostly does) is the right one — `wildboottest`, `ccv`, and the body of `predict` are the next candidates. No new abstractions needed, just relocation, which also removes most of the lazy `import_module("pyfixest.estimation")` calls that currently dodge circular imports in three places (`_vcov_crv3_slow`, `first_stage`, `ccv`) — those are a symptom of estimation logic living at the wrong layer.

Attribute lifecycle is implicit. Attributes are "enriched outside the class" (the code's own comment, feols_.py:357), `na_index` is "initiated outside of the class", `_lag`/`_time_id`/`_panel_id` exist only if the HAC branch ran, and `_clear_attributes` deletes attributes wholesale so `lean=True` models fail later with bare `AttributeError` rather than a message like "this model was fit with lean=True". `add_fixest_multi_context` (feols_.py:886) is no longer called by anything — dead code from the pre-refactor era; delete it. A small improvement with no architecture change: initialize the branch-dependent attributes to `None` in `__init__`, and route post-`lean` access through a tiny `_require(attr, hint)` helper that raises a helpful error.

Vcov state is stringly typed and scattered. `_deparse_vcov_input` returns a 4-tuple that lands in `_vcov_type`, `_vcov_type_detail`, `_is_clustered`, `_clustervar`, with HAC extras stored separately. B1, B2, and B12 all live in this seam. A frozen `VcovSpec` dataclass (kind, detail, clustervars, lag, time_id, panel_id) produced by one parser with an exhaustive `else: raise` would collapse the failure modes; the `vcov()` dispatch then matches on `spec.kind`.

Inconsistent handling of the `data` argument in `vcov()`. The method converts `data` via narwhals and then discards the result (feols_.py:658–664 — the converted frame is never used; `prepare_cluster_state` re-converts internally). The CRV branch honors `data`; the HAC branch reads `self._data` only. Either thread `data` through both or drop the parameter's promise for HAC explicitly.

`fit_one` calls `FIT.get_inference()` immediately after `FIT.vcov(...)`, but `vcov()` already ends with `self.get_inference()` — a harmless duplicate that suggests the ownership of "who triggers inference" was never settled. Pick one (vcov owning it is fine) and delete the other call.

Assertions as input validation. `ccv()` and `_check_vcov_input` validate user input with `assert`, which produces `AssertionError` instead of `TypeError`/`ValueError` and disappears under `python -O`. Convert to explicit raises; this also standardizes the error types your tests can rely on.

Silent fallback in the torch demeaner. `LsmrDemeaner.demean` falls back to the Rust MAP backend when torch isn't importable (demeaners.py:333–343) with no warning, despite the class having a `warn_on_cpu_fallback` knob for the milder case. A user who explicitly requested `backend="torch"` should at least get a `UserWarning`, arguably an error.

Determinism nits in `Feiv`. `_non_exo_instruments = list(set(...) - set(...))` (feiv_.py:258) gives set-dependent ordering; derive it by filtering `_coefnames_z` to keep a stable order. `first_stage()` also refits from scratch via a full `feols()` call (formula re-parse, model matrix rebuild) on every IV fit; fine for now, but worth a comment that it's a known cost, since the demeaned arrays for the first stage are already in the cache.

Docstring drift. This is pervasive enough to mislead: `ssc()` documents defaults (`k_fixef "none"`, `G_df "conventional"`) that contradict the signature (`"nonnested"`, `"min"`) and claims "currently the only option" for a three-option parameter; `predict()` docstrings say type defaults to "response" while the signature default is "link", and repeat `atol`/`btol` entries twice; the `Feols` attribute docstring documents attributes that no longer exist (`_tZXinv`) and misses ones that do. Since the three `predict` docstrings are near-copies, consider documenting once on `Feols.predict` and keeping subclass docstrings to the delta.

Small performance/cleanliness items, none urgent: `FixestMulti.wildboottest` grows a DataFrame by `pd.concat` inside the loop (quadratic; collect dicts, concat once); `fetch_model` silently int-casts string indices; `run_crv_loop` returns `df_k` from the last loop iteration only; `simultaneous_crit_val`'s `msqrt` uses `np.linalg.inv(eig_vecs)` where `.T` suffices for a symmetric matrix; `get_data` has a dead `df[df == "nan"] = np.nan` on float columns; `tidy()` re-runs `get_inference()` on every accessor call (`coef()`, `se()`, etc. each rebuild the full tidy frame — cheap per call, but it makes accessors side-effectful since they overwrite `_se`/`_pvalue`).

Repo hygiene: `estimation/deprecated/` (FormulaParser, model_matrix_fixest_, ~1,500 lines) plus the root-level shims `estimation/feols_.py` etc. are fine as a transition, but give them a removal version; `coverage.xml` and `__pycache__` directories are sitting in the tree.

---

## 4. Priorities

If I had one afternoon: fix B1–B5 (user-visible crashes and silent wrong numbers), delete the dead code (B8, B9, B11, `add_fixest_multi_context`, `ccv` tail), and add the fweights-expansion equivalence tests. The `_weights`-invariant change plus the `_Y_wls`/`_X_wls` rename is a focused follow-up PR — mechanical, high leverage, and it retires the whole category of "is this array weighted yet?" bugs. `VcovSpec` and the `Feols` slimming are worthwhile but can trail behind; they touch more surface with less immediate payoff.
