---
name: pr-draft-summary
description: Prepare reviewer-ready pyfixest single or stacked draft PRs after history curation and verification.
---

# Prepare the review handoff

Use this skill only after implementation, history curation, and required local
verification are complete. Read `docs/developer/git-and-pr-style.md`.

## Choose the delivery shape

Use one PR for a small cohesive change. Prefer a stack when two or more
independently reviewable concerns form coherent dependency layers. Each layer
must be testable against its immediate parent and contain no unrelated cleanup.

Before submission, inspect every layer's log/diff and the cumulative diff.

## Draft the PR body

Keep the handoff brief. For an ordinary PR, aim for 100–200 words and use only
the applicable sections of `.github/pull_request_template.md`. Use a short
`type(scope): outcome` title, one outcome-first paragraph, and at most a few
bullets for material review risks or non-obvious decisions.

Do not repeat information GitHub already shows: file lists, commit-by-commit
narratives, base/head SHAs, or implementation diaries. Group related passing
checks on one line. Give detail only to failures, deferred checks, and caveats
that change review decisions. Use a longer body only when the support matrix or
a numerical deviation genuinely needs it.

For numerical or estimator changes, add only the material facts:

- `fixest` parity or the intentional deviation and support limits;
- external reference and tolerance rationale;
- failed or deferred verification and any performance impact.

Link to tests or policy for details already visible in the diff instead of
copying the full support matrix into the body.

For a stack, identify the immediate parent only when it is not obvious from the
PR base. Do not force estimator-specific boilerplate into documentation, CI, or
maintenance PRs.

Submit new PRs as drafts. Mark a layer ready only when required checks pass or
clearly identified long checks are running in CI. Require human-maintainer
approval on every layer.

Stop at handoff. Do not merge the PR or invoke `gh stack merge`.
