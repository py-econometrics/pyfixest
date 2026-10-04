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
| Clustering on interactions | `vcov={"CRV1": "f1^f2"}` raises `ValueError`; users add the interacted variable as a data column and cluster on it. | `cluster = ~f1^f2` clusters on the interaction of `f1` and `f2`. | Cluster variables are read from the data by name, and interacted fixed-effect columns are not materialized there. Earlier versions silently rewrote `^` to `_` and then failed on the missing column. | `tests/test_errors.py::test_vcov_spec_rejects_malformed_input`; `fixest` 0.14.0 behavior recorded 2026-09-18 | Unsupported; documented for 0.70.0 |
| Empty first interaction cell | With an intercept or fixed effects, `i(a, b)` without explicit references may omit a different observed cell and warn about collinearity. | `fixest` 0.14.0 uses the first observed cell. | Accepted: equivalent fitted model and inference after aligning the reference; see [#1768](https://github.com/py-econometrics/pyfixest/issues/1768). | — | Intentional; documented for 0.70.0 |
| Mixed-case string levels | String levels are sorted by code point, so uppercase sorts before lowercase: with levels `apple`, `Banana`, `cherry`, `i(m)` and `C(m)` use `Banana` as the reference. | `fixest` 0.14.0 sorts levels by R's collation locale: under `en_US.UTF-8` the reference is `apple`; under the `C` locale it matches pyfixest. | Code-point order is deterministic and locale-independent; `i(m, ref="apple")` reproduces the `en_US.UTF-8` result. See [#1802](https://github.com/py-econometrics/pyfixest/issues/1802). | Confirmed with R 4.5.3 and `fixest` 0.14.0 on 2026-10-04 | Intentional; documented for 0.70.0 |
| `i(a)*X` slopes | Follows formulaic's `C(a)*X` convention: `X` is the slope of the reference level, and `a::a2:X`, `a::a3:X` are differences from it. | `fixest` 0.14.0 drops the last interaction `X:a::a3` as collinear, so `X` is the slope of the last level, and `X:a::a1`, `X:a::a2` are differences from it. | Same fitted model; changing it would need custom scoping rules inside formulaic. `i(a) + X + i(a, X, ref="a3")` reproduces fixest's coefficients, and `i(a) + i(a, X)` gives one slope per level in both packages. See [#1803](https://github.com/py-econometrics/pyfixest/issues/1803). | Confirmed with R 4.5.3 and `fixest` 0.14.0 on 2026-10-04 | Intentional; documented for 0.70.0 |

## Adding an entry

Add an entry in the same PR that introduces or formalizes the difference. Name
the exact external package version in the test or its reference artifact. Link
the issue or decision record when the rationale is too large for the table.

Compatibility work that restores parity should update or remove the ledger
entry and retain a regression test.
