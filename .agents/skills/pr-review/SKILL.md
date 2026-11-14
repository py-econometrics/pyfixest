---
name: pr-review
description: Review a pyfixest diff for defects and scope, self-review a change, or guide a requested walkthrough. Explanations use the walkthrough mode rather than launching a full audit.
---

# Review or explain a pyfixest diff

Resolve the [intended base](../../../docs/developer/git-and-pr-style.md#establish-the-base)
and include the relevant working changes. For a stack, distinguish the current
layer from its ancestors and the cumulative result. If the displayed diff
contains unrelated files, fix the comparison before explaining them.

## Guided walkthrough

When the user wants to review together, follow their pace. Start with the
concrete problem and the before/after behavior, then suggest the first changed
function to inspect. Explain one function or hunk at a time when requested.

Use a small, consistent example: show the formula or input, the relevant
intermediate values, and how the result changes. Define names before using
them. Connect numerical code to its equation, including shapes, scales,
ordering, and weights where they matter. Distinguish outputs you ran from
illustrations. Answer the current question directly; a syntax question does
not require a fresh review or a test run. Explain tests by the behavior and
failure they distinguish, unless the user asks for their mechanics.

Questions about alternatives call for an assessment, not an unsolicited
rewrite. Apply requested edits and use the checks appropriate to those edits.

## Defect review and self-review

For guidance or documentation, check accuracy, links, consistency, and scope.
Numerical tracing applies only to code that can affect results or state.

Prioritize silently wrong results. Trace the changed values through their
callers: construction, fitting, inference, post-estimation, and cleanup as
applicable. Check mathematical meaning, row/column alignment, weight factors
already applied, ownership, and lifetime. Frozen fields do not make contained
arrays immutable. Before deleting a wrapper or accessor, inspect subclass
overrides and callers; fewer functions alone do not prove a better design.

Check the [supported paths](../../../docs/developer/architecture.md#result-and-numerical-boundaries),
independent [numerical evidence](../../../docs/developer/testing.md#choosing-a-numerical-reference),
and documented compatibility decisions. Audit tests for distinct failures they
catch, not just their count or agreement with the implementation. Apply the
[scope policy](../../../docs/developer/git-and-pr-style.md#change-scope) to
independent improvements discovered in review.

Use [existing verification evidence](../../../docs/developer/testing.md#runtime-tiers)
before rerunning checks. A green suite does not rebut a demonstrated defect.
For a cumulative stack review, a later layer's fix resolves the final-stack
finding; it does not make the earlier layer independently acceptable.

Report actionable findings first, with file/line references and a concrete
failure mode. Separate introduced defects, pre-existing issues, compatibility
decisions, questions, and optional cleanup. Say when no findings remain; do
not manufacture blockers from naming preferences or unfamiliar syntax.
Summarize verification and remaining uncertainty without a second checklist.
