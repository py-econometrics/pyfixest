# /review-pr — Phase 4 (Review loop) of .agents/feature-pr.md

Execute Phase 4 of `.agents/feature-pr.md`: restore `BASE_REF`/`MERGE_BASE`
from `.claude/plans/<branch-name>.md`, review everything that would ship
(`git status --short`, `git diff "$MERGE_BASE"`, `git diff --cached
"$MERGE_BASE"`), apply the five review points, and run the check sequence
(targeted tests → the three lint hooks one at a time → `pixi run test-py` if
shared internals were touched). Max 2 passes.

This command adds a **plan-vs-implementation audit** — distinct from code
review — answering "did we build what Phase 1 said," reported as:

```
## Review report — pass [1|2]

### Plan audit
- [slice]: implemented as planned | deviated: [what + why] — per plan checkbox

### Diff hygiene
formatting churn / generated artifacts / lockfiles / unrelated refactors:
none remaining | removed: [...] | follow-up issue noted: [...]

### Placement & surface
logic in model classes that belongs in post_estimation/ or internals/: [...]
public surface vs feols/fepois/feglm/quantreg conventions: [...]

### Exports & docs
__init__.py (__all__, _lazy_imports) / per-function Parameters docstrings /
docs/_quarto.yml / how-to vignette: each done or justified-not-needed

### Checks
[command] → pass/fail (paste the failing tail, not a paraphrase)

### Remaining before handoff
- [ ] ...
```

Deviations from the plan are findings to report, not errors to hide —
some are improvements; say which. If pass 2 still leaves failures, apply
the Escalation rule instead of starting a third pass.

This command additionally confirms Phase 3 actually completed — the plan
file has a `## Numerical validation` section with no open items — and reports
it under **Checks** as `Phase 3: completed` or `Phase 3: not completed —
[open items]` (a Checks failure, not a footnote left across passes). This
check is not yet part of `.agents/feature-pr.md` itself; see
`.claude/README.md` → "Proposed changes to .agents/feature-pr.md" for the
diff proposed for the maintainer's review.
