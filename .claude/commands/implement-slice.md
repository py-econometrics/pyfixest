# /implement-slice — Phase 2 (Implementation loop) of .agents/feature-pr.md

Execute Phase 2 of `.agents/feature-pr.md` for **one slice**. Argument: the
slice name, or by default the first unchecked `- [ ]` slice in
`.claude/plans/<branch-name>.md`. If no plan file exists, run `/discover`
first — do not implement from an unwritten plan.

All rules live in Phase 2 of `.agents/feature-pr.md` (smallest coherent
change, narrowest tests via
`pixi run -e py312-r pytest <files> -x -q --no-cov`, also run the touched
subsystem's existing test files, the AGENTS.md rules PRs most often break).
This command adds:

- **Bound**: max 3 iterations on this slice. On the third failure of the
  same kind, stop and produce the Escalation report from
  `.agents/feature-pr.md` — do not try a fourth angle.
- **Plan mismatch**: if reality contradicts the plan (the analogue doesn't
  fit, a wiring point doesn't exist), stop and present:

  ```
  Issue in slice [name]:
  Expected (plan): ...
  Found: ...
  Why it matters: ...
  Options: ...
  ```

  and wait for the user. Do not silently re-plan.
- **On success**: check the slice's box in the plan file, note any dropped
  churn or follow-up items there, and report which slice is next.

Interaction rule: within a slice, work autonomously; the only mid-slice
questions are the plan-mismatch template above and escalation.
