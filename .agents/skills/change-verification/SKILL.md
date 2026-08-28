---
name: change-verification
description: Select, run, and report pyfixest checks before handing off code, tests, documentation, CI, or metadata changes.
---

# Verify a pyfixest change

Use this skill after implementation stabilizes and before handoff.

## Select checks by risk

Resolve and record the actual PR base and merge base as described in
`docs/developer/git-and-pr-style.md`; do not assume local `master` is current.
Inspect every changed path against that base and classify the change using the
selection matrix in `docs/developer/testing.md`. Start with targeted tests and
changed-file format/lint/type checks, then run the PR baseline and affected
domain suites.

- Numerical/API changes require the relevant public integration tests and
  external-reference suite. `test-py` is a Python-only regression baseline and
  never substitutes for an external numerical comparison. Use the fast R suite
  for edit feedback before the applicable canonical R suite.
- HAC changes require the single-threaded HAC suite.
- Rust changes require kernel/reference tests and platform CI.
- Performance-sensitive changes require before/after evidence.

Documentation commands are costly and opt-in. Do not run `docs-build` or
`docs-render` merely because repository guidance, workflow metadata, or a PR
template changed.

- Run `docs-build` only when public Python docstrings, quartodoc registration,
  or API-reference configuration changes.
- For content under `docs/`, render only the affected page when practical.
- Run `docs-render` only for site-wide configuration, navigation, templates, or
  cross-page changes.
- Changes limited to `AGENTS.md`, `.agents/`, `.github/` templates, or other
  workflow metadata need direct validation and `git diff --check`, not a docs
  build.
- Executable docstrings and documentation examples still require executing the
  changed example and checking the affected rendered page.

Unknown paths require the conservative PR baseline. For a stack, repeat the
selection for each layer against its immediate parent, then inspect the
cumulative top against the trunk.

## Report truthfully

For every applicable check record:

- status: passed, failed, deferred, or not run;
- exact command;
- elapsed time;
- reason and destination for a deferred check.

Run long domain suites after the design stabilizes. A deferred or CI-only check
is not a pass. Do not claim completion while a mandatory local check is
unreported or failing.

Write this report directly in the handoff or PR body. Do not introduce a
generated verification artifact unless the task specifically requires one.
