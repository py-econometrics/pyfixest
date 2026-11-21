# Wilkinson conventions win for formula semantics

Where the Wilkinson-formula conventions, as Formulaic defines them, differ
from `fixest`'s formula rules, pyfixest follows the Wilkinson conventions,
even though `fixest` is otherwise the behavioral reference. Matching `fixest`
would need custom scoping rules inside the formula engine, and pyfixest
formulas would then mean something different from the same formula in other
Formulaic-based Python packages. The decision holds whichever engine parses
the formula.

## Consequences

- Each such difference is still a ledger entry in
  [fixest-compatibility.md](../fixest-compatibility.md) that cites this
  decision: covariate powers, `i(a)*X` slopes, implicit `i()` references
  without an intercept, and term deduplication of redundant endogenous
  variables.
- Where a difference changes the fitted model rather than its
  parameterization, the user guide explains it.
