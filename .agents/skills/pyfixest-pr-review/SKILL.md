---
name: pyfixest-pr-review
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

## Output

Report actionable findings first, ordered by severity, with tight file/line
references and the concrete failure mode. Separate questions from findings.
Say when no findings remain, but list material checks not performed.

Automated review does not approve a PR. A human maintainer must review every
layer before merge.
