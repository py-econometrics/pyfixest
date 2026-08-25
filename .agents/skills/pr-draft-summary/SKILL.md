---
name: pr-draft-summary
description: Prepare reviewer-ready pyfixest single or stacked draft PRs after history curation and verification.
---

# Prepare the review handoff

Use this skill only after implementation, history curation, and required local
verification are complete.

## Choose the delivery shape

Use one PR for a small cohesive change. Prefer a stack when two or more
independently reviewable concerns form coherent dependency layers. Each layer
must be testable against its immediate parent and contain no unrelated cleanup.

Before submission, inspect every layer's log/diff and the cumulative diff.

## Draft the PR body

Complete `.github/pull_request_template.md` with:

- intent and user-visible behavior;
- architecture classification and nearest precedent;
- `fixest` parity or documented deviation;
- weights/FE/IV/multiple-estimation/backend support;
- external reference, version, data/seed, tolerances, and compared outputs;
- exact verification commands, statuses, and durations;
- deferred checks and where they will run;
- commit narrative, docs/changelog, and performance evidence.

Submit new PRs as drafts. Mark a layer ready only when required checks pass or
clearly identified long checks are running in CI. Require human-maintainer
approval on every layer.

Stop at handoff. Do not merge the PR or invoke `gh stack merge`.
