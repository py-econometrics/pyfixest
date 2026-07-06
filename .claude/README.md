# pyfixest agent toolkit — thin modular layer over .agents/feature-pr.md

## Design decision

This is a **hybrid**, not a pymc-marketing-style full toolkit.

`.agents/feature-pr.md` stays the single, tool-neutral source of truth for
the workflow, **unmodified by this toolkit** — it is written and owned by the
maintainer, and that ownership is worth protecting (pymc-marketing's
`.claude/` shows what happens without a single source of truth: workflow
logic duplicated across 40+ files drifted until several files still
referenced a different project entirely).

What this layer adds is only what a single linear file cannot provide, plus a
small number of behaviors that go *beyond* what `feature-pr.md` currently
documents. Those are called out explicitly below as **proposed changes** —
implemented at the command level so the toolkit can use them today, but not
folded into `feature-pr.md` itself. That file changes only when the
maintainer reviews the diff and decides to merge it.

| Borrowed from pymc-marketing | Where it lives here |
|---|---|
| Phase-level entry points (jump into any phase) | one slash-command per phase |
| Locator/analyzer sub-agent split for parallel discovery | `.claude/agents/` |
| Plans persisted as checkbox files (survive lost shell state) | `.claude/plans/<branch>.md` |
| Standalone "did we build what we planned" audit with a report template | inside `/review-pr` |
| Explicit ask/don't-ask rules per phase | stated in each command |
| Structured stop-and-present on plan mismatch | inside `/implement-slice` |

Rule that keeps this from rotting: **commands never restate workflow rules —
they point to the phase and add orchestration only.** If a command's addition
is really a workflow *rule* rather than orchestration, it belongs in
`feature-pr.md` — which means proposing it there, not editing it unilaterally.

## Layout

```
.claude/
  README.md             # this file
  commands/
    feature-pr.md        # full pipeline (Phases 1–5)
    discover.md          # Phase 1 + sub-agents + persisted plan + staleness check
    implement-slice.md   # Phase 2, one slice, bounded
    validate-numerics.md # Phase 3 + validation report
    review-pr.md         # Phase 4 + plan-vs-implementation audit + Phase 3 check
    rewrite-history.md   # Phase 5 + confirmation gate
  agents/
    codebase-locator.md  # WHERE code lives (grep/glob only)
    codebase-analyzer.md # HOW code works (file:line, no edits)
  plans/                 # local scratch — gitignored
```

`.agents/feature-pr.md` itself ships **unchanged** in this PR.

## Status

Dry-run and full-run tested end to end on a real PR (#1275,
`weightingboottest`): `/discover` → `/review-pr` (twice) → `/implement-slice`
(four slices) → a mid-flight upstream rebase → `/review-pr` again. The
proposed changes below are what that run surfaced as missing.

## Proposed changes to .agents/feature-pr.md

`.agents/feature-pr.md` is the maintainer's reference. Everything here is a
**proposal**: a quoted diff, the reason, and — for the three marked
"(evidence: #1275)" — what actually happened on a live PR that the current
file didn't anticipate. The maintainer decides what, if anything, to merge.
Until then, the behaviors already work at the command level (noted per item)
so the toolkit isn't blocked on the decision.

**1. Phase 1, step 6** — after "Write a plan of at most ten lines... Then
start.", add:
> Persist it to `.claude/plans/<branch>.md` (or an equivalent local scratch
> path in other tools) with each slice as a checkbox; later phases restore
> `BASE_REF`/`MERGE_BASE` from this file.

*Implemented today in:* `discover.md` ("Persisted plan" section).

**2. New "Interaction rules" paragraph**, after the Inputs paragraph:
> Interaction rules: during Phase 1, ask no clarification questions — resolve
> ambiguity by reading the repo; ask only if the task itself is undecidable.
> From Phase 2 on, the only interruptions are the plan-mismatch report and
> the Escalation report; Phase 5's rewrite additionally requires explicit
> user confirmation before `git reset --soft`.

*Implemented today in:* `discover.md` ("Interaction rule"),
`implement-slice.md` ("Interaction rule"), `rewrite-history.md`
(confirmation gate).

**3. Phase 2** — a fourth stop condition alongside the three-failure rule:
> A fourth stop condition, alongside the three-failure rule: if reality
> contradicts the plan (the analogue doesn't fit, a wiring point doesn't
> exist), stop and present expected vs. found vs. options; do not silently
> re-plan.

*Implemented today in:* `implement-slice.md` ("Plan mismatch").

**4. Phase 1, step 1** — a staleness check, new paragraph after computing
`MERGE_BASE` (evidence: #1275):
> Also check staleness before planning: if
> `git rev-list --count HEAD..$BASE_REF` is large (rule of thumb: more than
> ~15 commits) or `git merge-tree --write-tree HEAD "$BASE_REF"` reports
> conflicts, record it in the plan under a `## Staleness` heading (commits-
> behind count, and either "no conflicts" or the conflicting file list)
> before writing the rest of the plan. A large distance with real conflicts
> means the diff may need re-deriving after a merge, not just re-run against
> a stale base — say so up front rather than discovering it mid-
> implementation or at Phase 4.

Why: on #1275, `master` moved 58 commits (a full GLM architecture refactor —
`Fepois` folded under `Feglm`, a shared `GlmFamily`/`fit_glm_irls` kernel)
while the PR was in flight. Nothing in the current Phase 1 or Phase 4 would
have surfaced that before implementation started; it was only discovered when
`git push` came back `CONFLICTING`.

*Implemented today in:* `discover.md` ("Staleness check").

**5. Phase 4** — a sixth review point (evidence: #1275):
> Confirm Phase 3 actually completed: the plan file has a
> `## Numerical validation` section, and it lists no unresolved open items.
> A missing or still-open Phase 3 is itself a Checks failure — report it as
> `Phase 3: not completed — [open items]`, not a footnote left unresolved
> across review passes.

Why: on #1275, Phase 3 was flagged "open, non-blocking" in the very first
`/discover` pass and never revisited — including after a rebase that
replaced the hand-rolled IRLS fitting with a shared kernel, exactly the kind
of change that should trigger re-validation. Existing tests passing is a
signal, not the reference-backed validation Phase 3 asks for, and nothing
forced the reckoning.

*Implemented today in:* `review-pr.md` (`### Checks` addition).

**6. Escalation rule** — a named tier distinct from the three existing
triggers (evidence: #1275):
> A distinct, non-failure case: the codebase has changed shape since Phase
> 1's plan was written (a large upstream merge, a landed refactor touching
> the same files — see the Phase 1 staleness check). This is not something to
> escalate-and-report; it is a signal to re-enter Phase 1 with the new state
> — write a fresh plan (a new `## Restart` block in the same plan file,
> keeping prior slices' checkmarks for the record) and confirm it with the
> user before resuming Phase 2, the same way the original plan would be.
> This is a bigger jolt than Phase 2's plan-mismatch stop (a single slice's
> analogue not fitting) — distinguish the two.

Why: this is exactly what #1275 needed when the merge conflict turned out to
span a full GLM refactor, not a two-file conflict. Nothing in the workflow
names this case, so the response (stop, investigate deeply, write a fresh
plan, get explicit sign-off before implementing) came from judgment in the
moment rather than a documented playbook. It worked, but a first-time agent
without that judgment call available has nowhere to look.

*Not implemented at the command level* — this is a cross-cutting behavior,
not one command's addition, so there's no natural file to house a shadow
implementation in. It's proposed as-is; until decided, treat a large base
shift as this section describes by default.

## Testing it on a live PR

```
claude
> /discover clean up PR #<n>          # or: gh pr checkout <n> first
# inspect .claude/plans/<branch>.md, adjust if needed
> /implement-slice                     # repeat per slice (skip if only reviewing)
> /validate-numerics
> /review-pr
> /rewrite-history                     # only if the branch is yours to rewrite
```

For a pure review of an existing PR without changing it, `/discover` then
`/review-pr` alone already produce the plan audit and hygiene report.
