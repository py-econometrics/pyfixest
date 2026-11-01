# pyfixest — guide for coding agents

pyfixest brings R's `fixest` workflow to Python: fixed-effects estimation,
formula syntax, inference, and post-estimation, with Rust kernels for hot loops.

## Essentials

- Mirror `fixest` in overlapping user-facing behavior unless a difference is
  intentional; [compatibility policy](docs/developer/fixest-compatibility.md)
  owns those decisions. Never silently fit a different model or return numbers
  for an unsupported path.
- Start with the nearest implementation and its tests. Complete the request
  in small, self-contained changes that are easy to review; see
  [PR sizing](docs/developer/git-and-pr-style.md#stacks-and-commits).
  Keep independent fixes separate, including numerical corrections discovered
  during an invariant refactor.
- Prefer code a maintainer can follow from inputs to results. An abstraction
  earns its place by clarifying an operation, owning an invariant, or serving
  real callers—not by adding another layer to pass through.
- Use `pixi run` for Python, pytest, lint, docs, and R commands. Use
  `pixi task list` to discover tasks. Select checks from the
  [testing policy](docs/developer/testing.md#selection-matrix), including
  before/after benchmarks for performance-relevant estimation changes.

## Load only the workflow the task needs

| Task | Read |
|---|---|
| Implement estimator, inference, formula, post-estimation, public API, or shared-core changes | [implementation-strategy](.agents/skills/implementation-strategy/SKILL.md) |
| Select or report checks for a change | [change-verification](.agents/skills/change-verification/SKILL.md) |
| Review a diff, self-review, or walk through changed code together | [pr-review](.agents/skills/pr-review/SKILL.md), choosing the requested mode |
| Prepare commits, push, open a PR, or curate history | [pr-handoff](.agents/skills/pr-handoff/SKILL.md) |
| Create or edit a GitHub issue | [issue style](docs/developer/git-and-pr-style.md#issues) |

A question or prose correction does not require an implementation plan.
Reuse decisions and checks already established unless the change invalidates
them. Internal workflow steps need not appear as checklists in replies or PRs.

`docs/developer/` owns detailed policy; skills apply it. Read the relevant
sections, not every linked document. [Architecture](docs/developer/architecture.md)
owns code conventions and extension seams; [Git and PR style](docs/developer/git-and-pr-style.md)
owns scope and publication. These documents are not part of the rendered site.
`CLAUDE.md` imports this file; `.claude/skills/` links to the shared skills.
`docs/skills.md` is guidance for package users, not contributors.

## Boundaries

- Preserve unrelated user changes. Change `pixi.lock` or `Cargo.lock` only for
  intentional dependency work; avoid unrelated formatting.
- Never hand-edit generated `docs/reference/**`, `docs/_freeze/**`,
  `docs/_site/**`, `.coverage`, or `coverage.xml`.
- Follow [Git approval policy](docs/developer/git-and-pr-style.md#approval-and-handoff):
  never commit to `master`, rewrite history without approval for that specific
  rewrite, or merge agent-authored work.
- Edit contributor guidance only when the task concerns it. Otherwise propose
  a discovered policy gap at handoff instead of adding rules during a code fix.
