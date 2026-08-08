# /discover — Phase 1 (Discovery) of .agents/feature-pr.md

Execute Phase 1 of `.agents/feature-pr.md` exactly as written. This command
adds four things: parallel sub-agents, a persisted plan file, an interaction
rule, and a staleness check.

## Sub-agents

After establishing `BASE_REF`/`MERGE_BASE` and collecting the change surface
(steps 1–2), dispatch in parallel:

- **codebase-locator** — map every touched or implicated file to a subsystem
  (step 3) and propose the nearest analogue (step 5).
- **codebase-analyzer** — read the nearest analogue end to end and report its
  wiring points and the numerical reference its tests use.

Then read the key files the agents identified yourself before writing the
plan. Sub-agent summaries inform the plan; they do not replace reading.

## Persisted plan

Write the ten-line plan from step 6 to `.claude/plans/<branch-name>.md`
(create the directory if needed; it is local scratch, never committed).
Format: the plan lines, each future slice as a `- [ ]` checkbox, and the
literal `BASE_REF` and `MERGE_BASE` values. Later phases restore
`MERGE_BASE` from this file instead of recomputing from a possibly-changed
HEAD.

## Staleness check

Right after computing `MERGE_BASE`, check whether the base has moved far
enough that the plan itself might need to account for it: if
`git rev-list --count HEAD..$BASE_REF` is large (rule of thumb: more than
~15 commits) or `git merge-tree --write-tree HEAD "$BASE_REF"` reports
conflicts, add a `## Staleness` heading to the plan file with the
commits-behind count and either "no conflicts" or the conflicting file list.
Say this before writing the rest of the plan — a large distance with real
conflicts means the diff may need re-deriving after a merge, not just re-run
against a stale base. This check is not yet part of `.agents/feature-pr.md`
itself; see `.claude/README.md` → "Proposed changes to .agents/feature-pr.md"
for the diff proposed for the maintainer's review.

## Interaction rule

Ask no clarification questions during discovery — resolve ambiguity by
reading the repo. The one exception: if the task itself is undecidable (two
contradictory goals, or the referenced issue/PR does not exist), stop and
ask before writing any plan.

Finish by printing the plan and stating which slice `/implement-slice`
should start with.
