---
name: pr-review
description: Review pyfixest diffs for compatibility, numerical correctness, unsupported estimation paths, and repository policy.
---

# Review a pyfixest PR

Use this skill for an explicit PR review or the final self-review before
handoff. Resolve the actual base as described in
`docs/developer/git-and-pr-style.md`, then review the complete diff, including
uncommitted and untracked files.

## Review priorities

Prioritize findings that can produce silently wrong numbers:

1. weights, fixed effects, IV, multiple estimation, vcov/SSC, or stripped-data
   paths with undefined or untested behavior;
2. numerical changes without a permanent external reference;
3. public behavior that differs from `fixest` without a documented, tested
   decision;
4. estimator-specific logic leaking into shared runners or model classes;
5. input mutation, unstable RNG, unjustified tolerances, or missing convergence
   checks;
6. avoidable test-suite growth: one-off tests that belong in an existing matrix,
   duplicated fixtures/adapters/assertions, redundant coverage, or
   disproportionate runtime;
7. optional-dependency failures, export/doc gaps, generated churn, and
   unmeasured performance claims.

Check that a new estimator is an add-on and that every claimed support path is
tested or rejected explicitly.

## Spend verification budget on unresolved risk

Select checks with the `change-verification` skill, but review differs from
authoring in what it should re-run. Treat checks named in a PR brief as evidence
requirements, not as instructions to rerun every command locally. First resolve
the exact head SHA and inventory completed CI for that SHA, then run only the
cheapest check that can answer each remaining review question.

Do not duplicate an expensive successful exact-head CI job locally without a
concrete reason. Escalate to a broad suite only when a narrow check fails or
leaves material uncertainty, the diff crosses subsystems, equivalent exact-head
CI is missing, stale, cancelled, or non-equivalent, or repository policy
requires that suite. Before starting a multi-minute local check, state which
unresolved risk it addresses and why existing CI is insufficient.

For a stack, inspect every layer's diff and run targeted checks wherever
behavior changes. If the stated acceptance boundary is the cumulative stack, run
the broad regression suite once on the final head; earlier issues that the final
layer fixes are layering observations, not final blockers. Require broad
per-layer runs only when each layer must be independently releasable or the user
explicitly requests them.

## Output

Report actionable findings first, ordered by severity, with tight file/line
references and the concrete failure mode. Separate questions from findings.
Say when no findings remain. Classify material verification as locally run,
satisfied by exact-head CI, not applicable, or not run, and explain any
remaining uncertainty. Do not count redundant local and CI runs as independent
confidence. A green suite never overrides a concrete code finding.

Automated review does not approve a PR. A human maintainer must review every
layer before merge.
