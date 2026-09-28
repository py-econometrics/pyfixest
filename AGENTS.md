# pyfixest — guide for coding agents

pyfixest ports R's `fixest` to Python: fixed-effects estimation, fixest formula
syntax, post-estimation tools, and Rust kernels for hot loops.

## Core principles

1. **Mirror `fixest`** in user-facing behavior, naming, and defaults unless an
   intentional difference is documented and tested.
2. **Follow the nearest implementation.** Find the in-repo precedent and its
   tests before introducing a new pattern.
3. **Never return silently wrong numbers.** Define and test supported estimation
   paths; reject unsupported combinations explicitly. Numerical changes need
   external-reference evidence under the testing policy.

## Choose the relevant workflow

Read only the skill and policy sections needed for the task. Reuse decisions
and evidence already established in the current task unless the change
invalidates them. A prose-only edit does not require an estimator strategy.

| Task | Procedure | Policy |
|---|---|---|
| Implement estimator, API behavior, inference, formula, post-estimation, or shared-core changes | [implementation-strategy](.agents/skills/implementation-strategy/SKILL.md) | [Architecture](docs/developer/architecture.md): boundaries and extension seams |
| Verify a change before handoff | [change-verification](.agents/skills/change-verification/SKILL.md) | [Testing](docs/developer/testing.md#selection-matrix): required checks by change type |
| Review a diff or self-review before handoff | [pr-review](.agents/skills/pr-review/SKILL.md) | Relevant architecture, testing, and scope policies |
| Prepare commits, push, or submit a draft PR | [pr-handoff](.agents/skills/pr-handoff/SKILL.md) | [Git and PR style](docs/developer/git-and-pr-style.md) |
| Change intentional `fixest` differences | Relevant implementation/review skill | [Compatibility ledger](docs/developer/fixest-compatibility.md) |
| Create or edit a GitHub issue | Read the linked policy section | [Issue style](docs/developer/git-and-pr-style.md#issues) |

`docs/developer/` owns policy; skills apply it. Keep full rules and explanations
in one owner, with brief reminders where useful. These developer documents are
not rendered as part of the Quarto site. `CLAUDE.md` imports this file; keep
tool-specific copies of the rules out of the repository. The analytics prompt
at `docs/skills.md` serves users, not this contributor workflow.

## Working conventions

- Use `pixi run` for all Python, pytest, lint, docs, and R commands. Bare tools
  may miss dependencies or the compiled extension. Find commands and check
  selection in the testing policy; use `pixi task list` to discover tasks.
- Use econometric names such as `scores`, `meat`, `bread`, `u_hat`, and
  `clustid`; cite the method's paper in the implementing function.
- Use `from __future__ import annotations`, PEP 604 unions, keyword arguments
  for internal calls, and `NDArray[np.float64]` in stubs. Put shared typed option
  aliases in `pyfixest/estimation/internals/literals.py`.
- Validate options at the API boundary with `ValueError` naming allowed values;
  use the flat classes in `pyfixest/errors/` for domain failures.
- Guard optional dependencies at import time; raise an actionable error naming
  the pip extra only when that path is used.
- Use `np.random.default_rng(seed)`, never global seeding. Never mutate user
  input except through the documented `copy_data=False` path.
- Keep estimation-specific work out of generic runners. Architecture owns the
  [wiring recipes](docs/developer/architecture.md#repository-map-and-extension-seams),
  [support contracts](docs/developer/architecture.md#result-and-numerical-boundaries),
  and [documentation requirements](docs/developer/architecture.md#public-documentation).

## Boundaries

- Follow [Git approval policy](docs/developer/git-and-pr-style.md#approval-and-handoff):
  never commit to `master`, rewrite history without approval for that specific
  rewrite, or merge agent-authored work.
- Change `pixi.lock` or `Cargo.lock` only for intentional dependency changes.
- Do not hand-edit generated `docs/reference/**`, `docs/_freeze/**`,
  `docs/_site/**`, `.coverage`, or `coverage.xml`.
- Preserve unrelated user changes and avoid unrelated formatting.
- Edit `AGENTS.md`, `.agents/skills/`, or `docs/developer/` only when the task
  concerns guidance. Otherwise propose discovered policy gaps at handoff.
