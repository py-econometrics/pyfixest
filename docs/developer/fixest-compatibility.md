# Fixest compatibility

R `fixest` is the default behavioral reference for overlapping pyfixest
features. Match its user-facing names, defaults, formula behavior, estimates,
and inference unless there is a documented reason not to.

An intentional difference is complete only when it has:

- a precise statement of both behaviors;
- a user or architectural rationale;
- an external reference version;
- permanent tests for the chosen pyfixest behavior and the observed difference;
- human-maintainer review.

Newly observed discrepancies are not automatically intentional. Open or link an
issue and investigate them before adding them to this ledger.

## Compatibility ledger

| Area | Pyfixest behavior | `fixest` behavior | Rationale | Tests | Status |
|---|---|---|---|---|---|
| Stepwise grouping | Parenthesizes each stepwise replacement so surrounding formula operators apply to the entire step. | `fixest` 0.14.0 rejects stepwise calls combined with other operators. | Allows cumulative interactions without silently changing the requested model; see [#1774](https://github.com/py-econometrics/pyfixest/issues/1774). | `tests/test_formula_parse.py::TestMultipleEstimationExpansion::test_expand_all_multiple_estimation`; `tests/test_formula_parse.py::test_correct_number_of_models` verifies that all expanded models are fitted. | Intentional extension for 0.70.0 |
| Gaussian GLM inference | `feglm(family="gaussian")` matches `feols()`, base R `lm`, base R `glm`, and `fixest::feols` for OLS behavior and small-sample corrections. | `fixest::feglm(family="gaussian")` applies GLM small-sample corrections that differ slightly from `fixest::feols`. | A Gaussian identity-link model should agree with pyfixest OLS and base R's Gaussian linear-model behavior. | `tests/test_vs_fixest.py::test_feglm_gaussian_reference_behavior`; confirmed with R 4.5.3 and `fixest` 0.14.0 on 2026-08-25 | Intentional; documented for 0.70.0 |
| Empty first interaction cell | With an intercept or fixed effects, `i(a, b)` without explicit references may omit a different observed cell and warn about collinearity. | `fixest` 0.14.0 uses the first observed cell. | Accepted: equivalent fitted model and inference after aligning the reference; see [#1768](https://github.com/py-econometrics/pyfixest/issues/1768). | — | Intentional; documented for 0.70.0 |
| Clustering on interactions | `vcov={"CRV1": "f1:f2"}` clusters observations by each distinct `(f1, f2)` pair. | `cluster = ~f1^f2` clusters by the same pairs. | Pyfixest uses `:` consistently with its fixed-effect interaction syntax. | `tests/test_vs_fixest.py::test_cluster_interactions_against_fixest`; [fixest VCOV documentation](https://lrberge.github.io/fixest/reference/vcov.fixest.html) | Intentional syntax difference; documented for 0.70.0 |

## Adding an entry

Add an entry in the same PR that introduces or formalizes the difference. Name
the exact external package version in the test or its reference artifact. Link
the issue or decision record when the rationale is too large for the table.

Compatibility work that restores parity should update or remove the ledger
entry and retain a regression test.
