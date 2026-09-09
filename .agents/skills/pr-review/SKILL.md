---
name: pr-review
description: Reviews a pyfixest diff for fixest compatibility, numerical correctness, unsupported estimation paths, test-suite growth, and repository policy. Use for any explicit PR or diff review and for the final self-review before handoff.
---

# Review a pyfixest PR

Input: the complete diff against the resolved base, established as described
under "Establish the base" in `docs/developer/git-and-pr-style.md`, including
uncommitted and untracked
files, plus the head SHA and the CI already completed for it. Output: the
findings report below.

## Priorities

A silently wrong number is the highest review concern, so order findings by
their ability to produce one:

1. weights, fixed effects, IV, multiple estimation, vcov/SSC, or stripped-data
   paths with undefined or untested behavior;
2. numerical changes without a permanent external reference;
3. public behavior that differs from `fixest` without an entry in
   `docs/developer/fixest-compatibility.md`;
4. estimator-specific logic leaking into shared runners or model classes;
5. input mutation, unstable RNG, unjustified tolerances, missing convergence
   checks, or an undeclared behaviour change in
   `tests/test_release_contract.py`;
6. avoidable test-suite growth by the rules under "Test design" in
   `docs/developer/testing.md`;
7. optional-dependency failures, export or docs gaps, generated churn, and
   unmeasured performance claims.

Check that a new estimator is an add-on and that every claimed support path
is tested or rejected explicitly.

## Trace numerical changes through the code

For numerical changes and estimator-state refactors, trace each changed value
through construction, fitting, inference, post-estimation, and cleanup:
its mathematical role and scale, row and column ordering, ownership, and
lifetime. Map changed scores, weights, projections, and covariance expressions
to the estimator equations, including weight factors already folded into
another value. For copies, shared views, read-only flags, and frozen objects,
distinguish attribute rebinding from element mutation and state who may mutate
the underlying data. Identify the concrete guarantee or avoided work behind an
optimization; performance claims need measurements.

Check whether accurate econometric names, meaningful intermediate results, and
control flow expose the invariants a reader needs to verify the
implementation. Enduring contracts and useful mathematical explanations belong
in code documentation; migration history and review-only motivation belong in
the PR or a design note. Before a wrapper or shared accessor is removed, audit
its callers, subclass overrides, and supported estimator paths; fewer
abstractions are not automatically clearer or safer.

Separate demonstrated defects, pre-existing issues, intentional compatibility
decisions, questions, and optional maintainability improvements. Answer
questions directly and support disagreements with evidence rather than
treating unfamiliar syntax or a naming preference as a correctness bug.

## Spend verification budget on unresolved risk

Review differs from authoring in what to re-run. Treat checks named in a PR
brief as evidence requirements, not commands to repeat. Inventory the CI
completed for the exact head SHA first, then run only the cheapest check that
answers each remaining question. Escalate to a broad suite only when a narrow
check fails or leaves material uncertainty, the diff crosses subsystems,
equivalent exact-head CI is missing or stale, or the "Selection matrix" in
`testing.md` requires it. Before starting a multi-minute local check, state
which unresolved risk it
addresses and why existing CI is insufficient.

For a stack, review every layer's diff. When the stated acceptance boundary
is the cumulative stack, an issue that a later layer fixes is a layering
observation, not a blocker for that cumulative stack. It remains a blocker
for an affected layer being accepted independently.

## Output

Report actionable findings first, ordered by severity, each with a file and
line reference and the concrete failure mode. Separate questions from
findings, and say when no findings remain. Report verification with the status
vocabulary of the `change-verification` skill, noting which checks exact-head
CI satisfied, and explain any remaining uncertainty. Do not count redundant
local and CI runs as independent confidence. A green suite never overrides a
concrete code finding, and this review does not replace the human maintainer
review required before merge.
