# Git and pull-request style

This file owns change scope, Git authorization, and how work is presented to
reviewers. The [handoff skill](../../.agents/skills/pr-handoff/SKILL.md) applies it.

## Establish the base

Use the comparison the user asked for. Otherwise use an existing PR's actual
`baseRefName`; for local work, establish its intended parent from the branch
and task. Do not assume that the current checkout or a stale local `master`
is the base. Fetch the remote base when possible and record its merge-base;
if unavailable, state which local ref was used and that it may be stale.

Include relevant uncommitted and untracked changes. For a stack, inspect each
layer against its immediate parent and the cumulative result against trunk.
Confirm the diff viewer uses the same comparison. Files inherited from a
parent or already squash-merged are not new changes in this layer.

## Change scope

Complete the requested outcome with the smallest coherent change. Include
adjacent fixes or refactors only when requested, accepted, or necessary for
that outcome. A pre-existing bug does not automatically become a prerequisite.
In an invariant refactor, preserve behavior and separate newly discovered
numerical fixes. Identify independent follow-ups; open issues when authorized.

Prefer a local simplification over a new framework, registry, or compatibility
layer unless the latter solves a concrete problem. Conversely, do not reduce
line count by hiding the algorithm or dropping necessary support and evidence.
Keep scratch reports and debugging artifacts out of the PR unless requested.

## Branch names

Use `<type>/<short-kebab-case-intent>`, normally `feat`, `fix`, `refactor`,
`perf`, `test`, `docs`, `ci`, `build`, or `chore`. Name the outcome, not the
agent, issue number alone, or position in a stack: `fix/cluster-df`, for example.

## Approval and handoff

Never commit to `master` or rewrite contributor-owned history. Rewriting
agent-owned history requires explicit approval for that specific rewrite in
the current conversation, including unpushed history. A request to implement,
commit, push, or prepare a PR is not rewrite approval. Reuse approval already
given; do not ask again for the same action.

Before an unapproved rewrite, prepare the affected branches, immediate
parents, original tip SHAs, ownership, pushed/review state, dependent branches,
and exact commands. Proceed with a clean worktree on named branches other
than `master`, verified parents, and recoverable recorded tips. Verify that
history-only cleanup preserves the final tree. Never rewrite silently after
review starts.

Submit agent-authored work as draft PRs. Human maintainer review is required
for every PR and stack layer. Agents never merge their own work or invoke
`gh stack merge`. A review-ready handoff may have required long checks visibly
running in CI, with their status reported; merge readiness requires the
[testing evidence](testing.md#verification-reporting) to be complete.

## Stacks and commits

Prefer small, self-contained PRs, each addressing one coherent reviewer
concern with its implementation, tests, and necessary documentation together.
Each PR should be understandable and testable against its declared base.
When a PR grows too large to review comfortably or mixes separable concerns,
narrow it or split it into smaller coherent PRs. Choose boundaries by behavior,
dependency, and reviewability; use no fixed file- or line-count limit.

Keep independent changes in separate PRs; stack only when one layer actually
depends on another. Each layer must work and pass its applicable targeted
checks against its immediate parent. Sharing a file is not by itself a
dependency. Broad checks can run once at the cumulative tip unless layers
must be independently releasable.

Use `type(scope): imperative summary`; omit an unhelpful scope. Keep the
subject short and precise, with no trailing period or hand-written PR number.
Each commit addresses one reviewer concern, pairs behavior with its tests,
and passes its applicable targeted checks.
An optional body explains a non-obvious reason or constraint, not the mechanics
already visible in the diff. No fixed commit sequence is required.

## Pull-request opening

Start with the problem and resulting behavior. A concrete before/after example
often explains more than a list of files. Use a short paragraph or a few
bullets, then the applicable parts of the repository PR template. Scale detail
to the change; omit implementation diaries and internal workflow checklists.

Include only material compatibility differences, support limits, numerical
references, tolerance decisions, performance effects, and verification gaps.
Link tests or longer explanations instead of copying a support matrix. Follow
[verification reporting](testing.md#verification-reporting) for a compact check
summary. If scope changes, rewrite the title and body around the final diff.
Human approval belongs in GitHub review state, not an author checkbox.

## Issues

Use at most two short sentences unless the user asks for more: the concrete
problem or proposed change, then the desired outcome and relevant issue/PR
link. Omit headings, checklists, implementation plans, and background stories;
do not move overflow into comments. Include a reproduction when requested.
