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
  [fixest-compatibility.md](../developer/fixest-compatibility.md) that cites this
  decision: covariate powers, `i(a)*X` slopes, implicit `i()` references
  without an intercept, and term deduplication of redundant endogenous
  variables.
- Where a difference changes the fitted model rather than its
  parameterization, the user guide explains it.

## Evaluated expressions

The formula parser counts outputs of expressions as terms. For example,
`I(Y + Y2)` is a single dependent expression. Only a top-level plus can
separate dependent variables. Varying-slope expressions (such as `z**2`) and
interacted levels (such as `f1 - f2`) are evaluated with Python expression
semantics. Reinterpreting them as formula terms can silently change the fit.
The parser accepts `f1:f2[z]` with the slope on the last factor and rejects a
slope on an earlier factor or on multiple factors. Retain backticks around
Unicode names the Formulaic tokenizer cannot read unquoted.

Stepwise operators accept whitespace before the opening parenthesis and across
arguments. As in `fixest`, there is no `mvsw0`. Warn and fit once when repeated
terms cause multiple steps to resolve to the same model. A power of an IV term
may deduplicate it. Reject a power that expands into new terms whose IV roles
are ambiguous.
