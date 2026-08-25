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

Begin with an outcome-first opening, then complete the applicable sections of
`.github/pull_request_template.md`. For numerical or estimator changes include:

- intent and user-visible behavior;
- architecture classification and nearest precedent;
- `fixest` parity or documented deviation;
- weights/FE/IV/multiple-estimation/backend support;
- external reference, version, data/seed, tolerances, and compared outputs;
- exact verification commands, statuses, and durations;
- deferred checks and where they will run;
- docs/changelog and performance evidence.

For every agent-authored PR, also report the PR or stack-layer shape, immediate
parent, commit narrative, and deferred checks. Do not force estimator-specific
boilerplate into documentation, CI, or maintenance PRs.

Submit new PRs as drafts. Mark a layer ready only when required checks pass or
clearly identified long checks are running in CI. Require human-maintainer
approval on every layer.

Stop at handoff. Do not merge the PR or invoke `gh stack merge`.
