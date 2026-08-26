# Git and pull-request style

This document defines the style for presenting pyfixest changes to reviewers.
Use the [`pr-draft-summary`](../../.agents/skills/pr-draft-summary/SKILL.md)
skill to prepare the PR and the
[`history-curation`](../../.agents/skills/history-curation/SKILL.md) skill before
rewriting agent-owned commits. The skills define the procedures and safety
gates; this document defines the branch, commit, and PR conventions they apply.

## Establish the base

Use the PR's actual base branch, not an assumed local `master`. For an existing
PR, read `baseRefName` from GitHub. Fetch the corresponding remote branch when
network access permits, then record the base ref and merge-base SHA. If a fetch
fails, use the available remote-tracking ref and report that it may be stale.

For a stack, record each layer's immediate parent. Review and verify each layer
against that parent, then inspect the cumulative top layer against the trunk.

## Branch names

Use `<type>/<short-kebab-case-intent>`, where `type` is normally `feat`, `fix`,
`refactor`, `perf`, `test`, `docs`, `ci`, `build`, or `chore`. Name the
reviewer-visible outcome, not the authoring tool, agent, issue number alone, or
position in a stack.

Examples: `feat/oriv`, `fix/cluster-df`, `docs/agent-workflow`. Each stack
branch names its independently reviewable layer.

## Commits

Use `type(scope): imperative summary`, with the scope omitted when it adds no
information. Keep the subject precise, normally about 50–60 characters, with
no trailing period or hand-written PR number.

Each commit addresses one reviewer concern, is small enough to review
independently, and passes its applicable targeted checks. Pair tests with the
behavior they establish. A body is optional; when needed, explain why the
change exists, an important constraint, or a non-obvious compatibility choice.
Do not narrate mechanics already visible in the diff.

Prefer `fix(vcov): preserve clustered degrees of freedom` to `fix tests`, and
`test: cover weighted Poisson inference` to `add more tests`.

## Pull-request opening

Open with the outcome and motivation. The first one or two paragraphs should
say what changes, why it matters, where it belongs architecturally, and any
important non-goals. Do not begin with a file list, implementation diary, or
test-command dump.

For example:

> Adds ORIV as a standalone estimator for measurement-error correction. It
> preserves the shared estimation core and follows the existing estimator API
> conventions.
>
> The implementation is validated against Stata. Fixed effects are supported;
> weights and multiple estimation remain explicitly unsupported.

After the opening, complete only the applicable sections of the repository PR
template. Keep ordinary PR bodies under about 200 words. GitHub
already shows the files, commits, branches, and base SHA; repeat them only when
a non-obvious stack relationship affects review. Group successful checks and
give detail to failures, deferred checks, numerical deviations, and support
limits. Human approval is represented by GitHub review state, not by an author
checkbox.
