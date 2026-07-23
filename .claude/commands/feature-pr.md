# /feature-pr — run the full feature-PR workflow

Read `AGENTS.md`, then execute `.agents/feature-pr.md` end to end
(Phases 1–5). That file is the single source of truth for the process;
this command adds only orchestration:

- Phase 1: run `/discover` (uses the locator/analyzer sub-agents and
  persists the plan to `.claude/plans/<branch>.md`).
- Phase 2: run `/implement-slice` once per slice, in plan order.
- Phase 3: run `/validate-numerics`.
- Phase 4: run `/review-pr`.
- Phase 5: run `/rewrite-history` only after its preconditions hold.

Argument: an issue reference, a PR number to clean
(`gh pr checkout <n>`), or a task description. If none is given, ask for one
— this is the only clarification question permitted before Phase 1 starts.

Respect every bound in `.agents/feature-pr.md`: max 3 iterations per slice,
max 2 review passes, and the Escalation rule. When a bound is hit, stop and
produce the escalation report — never widen the change to keep going.
