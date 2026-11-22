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

Check that the diff does what its linked issue or request asked, and that it
follows the [coding standards](../../../CODING_STANDARDS.md). For guidance or
documentation, check accuracy, links, consistency, and scope.
Numerical tracing applies only to code that can affect results or state.

Find [behavior changes](../../../docs/developer/git-and-pr-style.md#change-scope)
by running code, not by reading the diff. For each check the diff adds or
rewrites, call the public function on the parent branch with inputs the new
check rejects: edge values (NaN, inf, empty, `True`, a single group) and every
option or estimator path that reaches the check. An input that returned a
result there is a behavior change. For a check the diff removes or relaxes,
look for inputs that used to raise. Also run the PR's new tests on the parent:
a test that expects an error and gets a result marks one. Report each, and
whether the request asked for it.

Probe each raise, warning, or assert message the diff adds or rewords the same
way. Name a public call that reaches it and run that call; the message you read
is the one it prints. A line you cannot reach is an invariant: report it for
deletion or a bug-report message, however well the text reads. Check that any
cause the message names matches what the probe shows, for example by repeating
the call with `lean=True` and `store_data=False`.

Prioritize silently wrong results. Trace the changed values through their
callers: construction, fitting, inference, post-estimation, and cleanup as
applicable. Check mathematical meaning, row/column alignment, weight factors
already applied, ownership, and lifetime. Frozen fields do not make contained
arrays immutable. Before deleting a wrapper or accessor, inspect subclass
overrides and callers; fewer functions alone do not prove a better design.

Check the [supported paths](../../../CODING_STANDARDS.md#results-and-numerical-boundaries),
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
