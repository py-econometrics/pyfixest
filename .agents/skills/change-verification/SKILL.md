---
name: change-verification
description: Select, run, and report pyfixest checks before handing off code, tests, documentation, CI, or metadata changes.
---

# Verify a pyfixest change

Use this skill after implementation stabilizes and before handoff.

Classify the diff and preview the selected checks:

```bash
pixi run agent-scope --base <immediate-parent>
pixi run agent-verify --base <immediate-parent> --tier pr --dry-run
```

## Select checks by risk

Inspect the diff against the actual PR base. Start with targeted tests and
changed-file format/lint/type checks, then run the PR baseline and affected
domain suites described in `AGENTS.md`.

- Numerical/API changes require the relevant public integration tests and
  external-reference suite.
- HAC changes require the single-threaded HAC suite.
- Rust changes require kernel/reference tests and platform CI.
- Executable docstrings require their body and affected reference page.
- Performance-sensitive changes require before/after evidence.

Unknown paths select the conservative PR baseline.

Run the required tier after reviewing the dry-run plan. Use `--tier domain`
for applicable long suites. A CI-eligible check may be deferred explicitly with
`--defer CHECK_ID=REASON`; a required local check still makes verification
fail when deferred.

## Report truthfully

For every applicable check record:

- status: passed, failed, deferred, or not run;
- exact command;
- elapsed time;
- reason and destination for a deferred check.

Run long domain suites after the design stabilizes. A deferred or CI-only check
is not a pass. Do not claim completion while a mandatory local check is
unreported or failing.

Use `--json-output <explicit-path>` when a machine-readable handoff artifact
is useful. The verifier writes nothing by default.
