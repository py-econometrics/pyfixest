# /rewrite-history — Phase 5 (Commit history rewrite) of .agents/feature-pr.md

Execute Phase 5 of `.agents/feature-pr.md`. Everything is defined there:
the four preconditions (feature branch, clean `git status --short`, branch
is yours to rewrite or force-push approved, `MERGE_BASE` set with the
`TIP` escape hatch recorded), the soft-reset recipe, and the target shape
(two to five commits mirroring the Phase 2 slices, short imperative
subjects, each commit passing its own targeted tests).

This command adds:

- Restore `MERGE_BASE` from `.claude/plans/<branch-name>.md`; record
  `TIP=$(git rev-parse HEAD)` in that file too, so recovery survives a lost
  shell session.
- **Confirmation gate**: before running `git reset --soft`, print the
  precondition checklist with each item's actual observed value and the
  planned commit slicing, and wait for the user's go-ahead. History rewrite
  is the one destructive step in the workflow; it never runs unprompted.
- If any precondition fails, stop and say which one and what would resolve
  it (commit, stash, or user decision). Never proceed with a dirty tree.
- If the branch belongs to an outside contributor, flag it here and again in
  the handoff summary, per Phase 5.
