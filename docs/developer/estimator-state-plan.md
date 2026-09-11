# Pyfixest 1.0: authoritative components, completed results, and a final switch to within-backed LSMR

This document records the delivery plan. Current method boundaries are defined
in [architecture.md](architecture.md#public-and-internal-estimator-methods).
The method-naming step below is an additional PR; it does not change the scope
of the component-publication PR currently under review (#1533).

## 1. Design and delivery contract

Migrate fitted models from overlapping private attributes to explicit components with defined numerical domains, ownership, and retention. Construct completed results rather than progressively populating `Feols` through method calls.

The agreed principles are:

- **One authoritative representation per quantity and numerical domain.** Remove duplicate stored fields and obsolete aliases. Shared references and derived views are allowed; a component does not require a separate memory allocation.
- **Positive component definitions.** Each component has a specific statistical or computational responsibility; none is a container for miscellaneous remaining fields.
- **Calculations return values.** Preparation, fitting, inference, and post-estimation functions return typed results.
- **Completed construction.** Constructors bind completed components. Ordinary result methods do not attach fields, replace statistical state, or populate hidden caches.
- **Functional covariance changes.** `with_vcov(...)` returns a new model or collection. Until that migration, `vcov(...)` remains both a public API and an internal entry point. Remove the in-place `vcov(...)` API in PR 7 for 1.0.
- **State for inspection.** Components expose fitted state through one interface for internal calculations and user inspection. Mutating their arrays or tables is unsupported and may invalidate results. Blanket public-state immutability and defensive copying are deferred.
- **Preserve input semantics.** Respect the existing `copy_data` option and avoid unintended mutation of caller data or its writeability during fitting. Copy only when the established input contract or a calculation requires independent storage.
- **Protect shared execution resources.** Keep ordinary NumPy read-only flags on shared demeaning-cache buffers and reordered cache selections to catch accidental internal writes that could corrupt other fits.
- **Eager ordinary results.** Standard fitted outputs, requested inference, applicable default diagnostics, and supported fixed-effect coefficients are available before returning an ordinary retained model.
- **Eager describes availability, not a required recovery algorithm.** Fixed-effect coefficients may come directly from estimation or from a completion-time recovery calculation.
- **Narrow internal work.** Lean results and internal refits may omit unnecessary computation and retention. Custom analyses remain on demand and return independent results.
- **Make internal method names explicit, then remove the obsolete lifecycle.** All estimator/result `get_*` methods are internal. In PR 1a, rename them to `_get_*` consistently across model classes, mixins, and multiple-estimation adapters. Remove the resulting internal methods in the later component/construction PRs; do not retain unprefixed compatibility aliases. Unrelated APIs such as `get_data()` are outside this migration.
- **Keep legitimate standalone fields.** Explicitly declared labels and other coherent model metadata need not be wrapped in a dataclass.

Breaking state-access and mutation APIs is authorized for 1.0. Numerical targets and estimator support remain unchanged unless a separate, documented correctness change is necessary.

**Delivery is strictly serial:** implement one PR, verify it, obtain human review, adjust it, and wait for human merge. Begin the next PR only afterward, from updated `master`.

The **last PR** changes the default FE absorption strategy to the existing `pf.LsmrDemeaner(backend="within")`. It does not introduce a joint regression solver or change the separate structural-coefficient solver.

## 2. Component ownership and public APIs

### Ownership ledger

Before migrating fields, record their meaning, numerical domain, writers, readers, lifetime, and destination. Classify each as component-owned, derived, standalone, or temporary.

Retaining an expensive derived quantity in its owning component is acceptable. Avoid repeated expensive computation merely to eliminate storage.

### Components

| Component | Responsibility |
|---|---|
| `ModelMatrix` | Formula-materialized response, regressor, FE, IV, weight, and offset roles before within transformation or solver weighting. |
| `ObservationWeights` | User-scale weights and analytic/frequency interpretation; `None` represents an unweighted fit. |
| `SampleInfo` | Retained-row identities, excluded-row index set, physical/effective observation counts, and exclusion counts by filtering stage. |
| `ColumnSelection` | Selected and dropped names/positions, separately identifying structural-design and instrument selections. |
| `WithinLinearData` / `WithinIvData` | Unpremultiplied within-scale arrays, including IV-specific instrument and endogenous roles. |
| `GlmWorkingState` | Final working response/design/weights, predictors, means, and response/working residuals. |
| `FitGeometry` | Bread and retained normal-equation products required for inference and post-estimation; IV has a typed extension. |
| `InferenceResult` | Covariance, coefficient-level inference, distribution, degrees of freedom, SSC information, and cluster summary. |
| Performance components | Completed linear/Gaussian measures and estimator-specific goodness-of-fit summaries. |
| Fixed-effects component | Completed coefficients, level mappings, normalization information, and computation settings. |
| `RefitSpec` | Specification and settings required to reproduce preparation and estimation faithfully. |

Define estimator result components positively:

- **Linear:** coefficients, response-scale residuals, fitted values, and weighted scores.
- **IV:** structural coefficients, structural residuals/fitted values, and instrument-based weighted scores. First stages and diagnostics have explicit typed ownership.
- **GLM:** coefficients, scores, deviance, convergence status, iteration count, and a reference to completed working state. Do not recopy its arrays into additional fields.
- **Quantile:** coefficients, residuals/fitted values, objective value, convergence information, and solver products required by supported inference.

Reuse and refine existing `OlsFit`, `IvFit`, `GlmFit`, `PerformanceMeasures`, and `FixedEffect` structures.

Specific ownership decisions:

- Move observation counts to `SampleInfo`; remove redundant model and weight-component count fields.
- Derive coefficient count and empty-design status from column selection.
- Keep scores in estimator results and bread/cross-products in geometry.
- Make `_tZy` solver-local.
- Keep observation-sized cluster preparation separate from inference summaries.
- Remove the default duplicate `_response`. Formula-scale calculations use formula state while available; later performance access reads completed performance.
- Keep solver projection coefficients separate from final fitted fixed-effect coefficients.
- Keep labels as declared metadata rather than creating a miscellaneous component.

### Sample exclusions

Add exclusion **counts by filtering stage** in the metadata PR. Capture newly removed rows when formula missing-value filtering, nonfinite filtering, singleton removal, and separation occur.

Preserve the excluded-row index set for alignment. Do not infer per-row reasons afterward. Per-row exclusion labels remain outside this migration.

### Public access and methods

Expose component types from `pyfixest.estimation.state`, using snake-case instance names:

```python
fit.model_matrix.dependent
fit.observation_weights.values
fit.sample.n_effective
fit.within_data.design
fit.inference.covariance
fit.performance.r2

updated_fit = fit.with_vcov({"CRV1": "firm"})
```

Keep practitioner-facing formatting and behavior methods such as `coef()`, `resid()`, `predict()`, `fixef()`, and `tidy()`. The existing `vcov()` is also public and used internally; its public contract remains in force until PR 7 replaces it with `with_vcov()`.

All `get_*` methods on estimator/result classes, their mixins, and adapters are
internal, including `get_fit()`, `get_inference()`, and `get_performance()`.
Prefix these names with `_` in PR 1a. Later remove `_get_inference()` and
`_get_performance()` entirely, and remove `_get_fit()` and other mutating
fitting lifecycle methods when completed construction replaces them.

Internal methods require their callers to provide the necessary state; they
are not restricted to fitting. In particular, public `vcov()` calls internal
inference computation after fitting. Keep unsupported-operation and
missing-data validation at public entry points, and do not introduce new
silent no-op guards or user-facing error contracts for internal helpers as
part of the naming change.

`with_vcov` retains current covariance-specification, keyword, and supplemental-data inputs. It shares unchanged estimation state and constructs new inference plus covariance-dependent diagnostics. Do not silently reinterpret old `vcov()` calls as nonmutating.

Custom Wald tests, decomposition, resampling inference, and optional diagnostics return dedicated results. Simulation draws and related plotting belong to those results, not attached model fields.

## 3. State access, construction, eager completion, and retention

### Keep state access simple

Component ownership defines meaning, writers, readers, and lifetime. It does
not require separate physical storage for each component. Internal calculations
and user inspection use the same named, typed properties, such as
`model_matrix.dependent`; do not introduce duplicate public/private accessors
solely to provide defensive copies. Name column-name metadata explicitly, for
example `_dependent_column_names`.

Components expose fitted state for inspection. Mutating their contents is
unsupported and may invalidate results. There is no guarantee that table
access returns a detached copy or that every component array is read-only.
Users who need editable data should make an explicit copy.

Preserve the existing `copy_data` semantics and do not change caller-array
writeability as a side effect of fitting. Avoid redundant internal copies of
formula, within, or working arrays. Dataclasses bind components; they do not
automatically copy or freeze supplied arrays. Solver scratch remains writable.

Keep the specific protection required for shared execution resources:

- Flag shared demeaned cache entries read-only before publication.
- Cache growth publishes a new entry without invalidating existing views.
- Flag reordered cache selections read-only for a consistent cache interface.
- The no-FE cache path passes inputs through without copying or freezing them.

Ordinary supported library operations must not accidentally mutate another
result's shared state. Comprehensive public mutation protection, immutable
pandas wrappers, and protection against deliberate flag changes remain outside
this migration unless a demonstrated use case warrants a separate proposal.

### Completed single and multiple results

Replace the mutating lifecycle with typed transformations:

```python
prepared = prepare_model(spec, data, resources)
estimated = estimate(prepared, requirements)
geometry = build_geometry(prepared, estimated)
inference = compute_inference(prepared, estimated, geometry, inference_spec)
completed = complete_standard_results(prepared, estimated, inference, requirements)
retained = apply_retention(completed, retention_policy)

return construct_result(retained)
```

Estimator adapters own numerical differences. The shared runner coordinates transformations without accumulating estimator-specific numerical branches.

For multiple estimation, build complete children first, establish their order and labels, and construct the immutable container last. `fetch_model()` and `to_list()` read that completed structure.

Reporting reads container labels or derives a local disambiguation mapping. It must not modify children’s model names or plot labels.

Functional multi-model covariance updates construct every replacement child before constructing the replacement container. Any failure leaves every original child unchanged.

### Eager completion

Ordinary retained models contain:

- Coefficients, fitted values, residuals, and convergence information.
- Requested inference.
- Applicable standard performance measures and default tests.
- Existing default IV first stages and diagnostics.
- Supported fitted fixed-effect coefficients.

Before the final LSMR PR, use existing fixed-effect recovery algorithms, normalization, and default tolerances as the completion mechanism.

Default `fixef()` formats the completed component; prediction reuses it. Explicit nondefault recovery tolerances may perform a local calculation without replacing stored state.

Do not expand currently unsupported estimator combinations through this work.

Internal refits receive typed product requirements. Coefficient-only refits reproduce estimation semantics but omit unused fixed-effect recovery, reporting state, and nested diagnostics. They return narrow internal results rather than incomplete fitted models.

### Retention and errors

| Mode | Contract |
|---|---|
| Default | Retain ordinary completed components and data required by supported post-estimation operations. |
| `store_data=False` | Drop raw data and full materialized formula tables; retain fitted numerical state, necessary formula metadata, and completed supported fixed effects. |
| `lean=True` | Retain compact coefficient, naming, sample-summary, inference, and performance state; omit unnecessary observation-sized state and recovery work. |

Apply retention during construction and recursively to first stages and multi-model children. Audit contexts, closures, shared buffers, and resources for indirect retention.
A small view can keep a larger allocation alive: make a targeted copy when
needed to release storage excluded by the retention policy, rather than copying
every component by default.

Introduce `MissingModelDataError` in the retained-storage PR. Messages identify the operation, missing state, and storage option or supplemental-data remedy. Keep estimator-support errors separate.

Align supplemental data through retained sample information, not merely matching row counts.

Mutable demeaning caches remain execution resources bounded by compatible work. Preserve supported preconditioner reuse without retaining cache dictionaries on statistical results.

Thin report methods import the report module inside their bodies to preserve the circular-import boundary.

## 4. Serial PR sequence

Each PR is reviewed, adjusted, and human-merged before starting its successor.

| PR | Work and completion criterion |
|---|---|
| **1 — Revised #1522** | Publish existing components, migrate consumers, remove superseded private names/aliases, preserve input semantics and shared-cache protection, and document inspection-only access paths. Existing migrated components have one authoritative representation. |
| **1a — Internal estimator method names** | Rename estimator/result `get_*` methods to `_get_*` across definitions, overrides, protocols, callers, adapters, and tests. Remove public-reference entries for those internal methods. Keep `vcov()` public and internally callable with unchanged behavior. Introduce no numerical, retention, or lifecycle-order changes and no unprefixed compatibility aliases. |
| **2 — Adapt #1523** | Introduce `MissingModelDataError`, component-based retention, operation guards, and recursive retention tests, including storage retained indirectly through shared views. |
| **3 — Adapt #1524** | Establish faithful typed refit specifications preserving weights, offsets, sample rules, context, solver settings, and demeaners. |
| **4 — Adapt #1525** | Preserve transactionality until functional inference replaces it; verify later-child failures cannot partially update earlier children. |
| **5 — Adapt #1526** | Return typed Wald results and migrate consumers. |
| **6 — Adapt #1527** | Integrate GLM refit inference after its prerequisites and external evidence. |
| **7 — Authoritative inference** | Introduce complete inference components and `with_vcov`; remove in-place updates and `_get_inference()`. Preserve DiD inference behavior and external-covariance adoption without additional SSC. |
| **8 — Fit products and metadata** | Introduce sample counts, column selection, geometry, positively defined estimator results, and directly retained performance components. Remove `_get_performance()` and duplicate numerical fields. |
| **9 — Completed construction** | Replace mutating lifecycle methods including `_get_fit()`, construct children before containers, introduce narrow internal requirements, and remove placeholders/setters/cleanup by arbitrary deletion. |
| **10 — Eager fixed effects and final retention** | Extract recovery into a completed-result calculation, make supported coefficients eager for ordinary retained fits, and finish resource lifetime and retention handling. Keep the component independent of how coefficients were obtained. |
| **11 — Consumers and final removal audit** | Migrate post-estimation, DiD, and reporting; use method-local report imports; remove remaining obsolete lifecycle methods and scaffolding; complete documentation. |
| **12 — within coefficients and default LSMR absorption** | Propagate solver coefficients into the established fixed-effects component, validate and benchmark the implementation, then switch the default FE absorption strategy. |

Astra owns contracts, numerical decisions, and integration. Sol may perform bounded inventories, migrations, and specified tests within the active PR.

### PR 1a: method naming without behavior changes

Deliver this PR after human merge of PR 1 and before the remaining sequence.
Keep the existing PR numbers; `1a` is an insertion, not a renumbering. Already
prepared later branches must migrate their added callers and overrides when
updated onto the merged naming PR, following the repository's approval rules
for any history rewrite.

- Inventory all estimator/result `get_*` definitions, including inherited
  mixin methods and multi-model adapters. The known names are `get_fit()`,
  `get_inference()`, and `get_performance()`; the convention applies to the
  full inventory, not only these examples.
- Change definitions, subclass overrides, protocol members, internal calls,
  tests, monkeypatches, and documentation together. Keep existing `_get_*`
  methods as they are and leave unrelated dataset/helper APIs alone.
- Preserve execution order, method arguments and returns, calculation inputs,
  retained state, and supported estimator combinations. Do not implement
  `with_vcov()` or remove the calculations in this PR.
- Verify fitting and public `vcov()` after fitting, including multiple models,
  inheritance dispatch, and retained-data requirements. Use the release
  contract, existing targeted tests, and applicable repository checks; do not
  add a redundant numerical matrix solely for the rename.
- Keep the current rule discoverable in `architecture.md` so subsequent
  agents do not infer that the legacy `get_*` spellings were public APIs.

### Final PR: solver-provided fixed effects and the default change

Use the existing `pf.LsmrDemeaner(backend="within")`, with its established configuration defaults. Preserve explicit MAP and other backend selections. Leave the structural-coefficient solver unchanged.

The bundled `within` result already contains projection coefficients and coefficient-layout metadata; the current wrapper discards them. Extend the Rust/Python boundary with a typed output carrying the needed coefficients, layout, identification information, and convergence metadata.

For linear models, obtain final fixed effects from the projection solves:

```text
alpha = alpha_y - alpha_X @ beta
```

Here `alpha_y` and `alpha_X` correspond to residualizing the response and the selected structural regressors against the same FE design and observation weights. They must use identical coefficient layouts and column ordering.

For GLMs, use projection products consistent with the final accepted coefficient iterate, working weights, and offset treatment. Do not combine stale iteration coefficients with final model outputs.

Further requirements:

- Map solver coefficient layout to existing FE level metadata and normalization conventions.
- Distinguish unidentified directions from ordinary reference-level normalization.
- Preserve public fixed-effect contributions and predictions; do not expose raw solver conventions accidentally.
- Projection coefficients may live in the compatible execution cache while multiple fits reuse them. Retain only combined fitted coefficients on completed models.
- Do not retain a full per-regressor FE coefficient matrix on each fitted result.
- Keep the current single-FE MAP fallback and no-FE behavior. Supply their completed fixed-effects component through the appropriate existing recovery path.
- Avoid a separate recovery solve on within-backed paths once solver products provide the required coefficients correctly.
- Retain explicit nondefault recovery-tolerance behavior without hidden mutation.
- Switch the default only after the external comparisons, backend checks, and performance measurements pass.

This final default change is a separate numerical/default-policy change. Statistical targets remain unchanged, but any expected floating-point differences must be explicitly documented and externally validated.

## 5. Verification and release acceptance

### Numerical coverage

Cover unweighted, analytic-weighted, and frequency-weighted fits; FE and IV configurations; structural/instrument collinearity; GLM offsets and separation; splits and multi-model estimation; both multi-quantile methods; supported backends; and retention modes.

Use canonical live R `fixest` and R `quantreg` comparisons. Preserve existing tolerances during the invariant architecture migration.

For fixed effects, compare normalized coefficients where appropriate, aggregate FE contributions, and in-sample/out-of-sample predictions.

For the final LSMR PR, additionally test coefficient-layout mapping, disconnected or unidentified FE structures, normalization, weighted projection reconstruction, multiple-estimation reuse, and final GLM iteration consistency.

### Ownership and lifecycle tests

Verify that:

- Removed aliases and internal lifecycle methods are absent.
- After PR 1a, unprefixed estimator `get_*` methods are absent and internal
  dispatch uses `_get_*`; public `vcov()` remains usable until PR 7.
- Internal consumers and user inspection use the same component properties.
- Shared cache views and reordered selections reject ordinary element writes.
- Cache reuse and growth leave earlier fits' views valid and unchanged.
- Preparation preserves the existing `copy_data` semantics and does not change user input writeability.
- Do not require detached public tables, universally read-only arrays, or immunity to unsupported user edits.
- Exclusion counts reflect actual filtering order without double-counting.
- Functional covariance updates leave originals unchanged on success and failure.
- Multi containers publish only completed children.
- Selection and reporting do not mutate children or labels.
- Report wrappers work under both estimation-first and reporting-first imports.
- Ordinary queries neither attach fields nor populate caches.
- Internal refits omit unnecessary completion work.
- Missing state raises `MissingModelDataError`.
- Recursive retention releases omitted state and indirect references.
- Solver-provided FE coefficients avoid redundant recovery and do not retain unnecessary projection matrices.

### Performance and verification cadence

Measure default FE fitting, repeated prediction, retained/peak memory, and resampling workloads before and after eager recovery. Repeat these comparisons in the final LSMR PR against explicit MAP, including easy, difficult, single-FE, and multi-FE designs.

Run the release contract during invariant-refactor edit loops:

```sh
pixi run -e py312 test-release-contract
```

Require passed cases, not skipped baselines. Each PR receives targeted tests and changed-file quality checks. Run required Python, external-reference, platform, and documentation suites according to repository policy on the exact submitted head.

Do not widen tolerances to absorb architectural regressions. For the final default change, document any intentional numerical deltas with the release contract’s reasoned declarations and corresponding external evidence.

### Completion

The migration is complete when component contracts are documented; duplicates and obsolete lifecycle methods are removed; single and multiple results are constructed complete; covariance updates are functional; ordinary queries do not mutate state; fixed effects are eager where required; retention, input semantics, and shared-cache protection pass verification; and the final validated default uses within-backed LSMR absorption.

Keep issue #1532’s allocation-free demeaning-kernel rewrite separate. This migration preserves unweighted `None` state but does not require that kernel rewrite.

Automatic Gaussian fit-statistics reporting (#1535) and saturated-OLS
adjusted-R² handling (#1536) are separate behavior changes. Neither is part of
component publication or the method-naming PR; component migration preserves
the measures and behavior available at its reviewed base.
