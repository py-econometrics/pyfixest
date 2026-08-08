# /validate-numerics — Phase 3 (Numerical validation) of .agents/feature-pr.md

Execute Phase 3 of `.agents/feature-pr.md`: validate against the strongest
available reference, in that file's stated order of preference (R via rpy2 →
stored verified output in `tests/data/` → brute-force reimplementation →
closed-form collapse → seeded Monte Carlo). The dependency rule for reference
packages and the test-shape guidance (heavily parametrized public-API
integration tests, `tests/test_vs_fixest.py` shape) are defined there and in
`AGENTS.md` → "Testing".

This command adds a required output: append a **validation report** to
`.claude/plans/<branch-name>.md`:

```
## Numerical validation — [date]
Reference used: [which rung of the ladder, and why not a higher one]
Paths covered: [no-FE / FE, unweighted / weighted, clustered, IV, multi-estimation
               — mark each covered or justified-out-of-scope]
Invariants asserted: [vcov symmetry + positive diagonal, row-order invariance, ...]
Tolerances: [each rtol/atol with its one-line justification]
Marks: [against_r_core / against_r_extended; conftest _rpy2_test_files updated? y/n]
Open doubts: [anything not established — be explicit]
```

Hard rules:
- Shapes matching is not validation. If no reference on the ladder can be
  established, that is the third Escalation trigger in
  `.agents/feature-pr.md` — stop and report; do not declare success.
- Never loosen a tolerance to make a test pass without writing the
  justification for why the looser bound is expected.
