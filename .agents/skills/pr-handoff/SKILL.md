---
name: pr-handoff
description: Prepare requested pyfixest commits, pushes, and draft PRs, or an explicitly requested history cleanup. Does not authorize publication or require history rewriting.
---

# Hand off work

Read the applicable sections of
[Git and PR style](../../../docs/developer/git-and-pr-style.md): base, branch
names, PR sizing, commits, PR opening, and approval. Reuse verification already selected
by `change-verification`; handoff is not a fresh test cycle.

Inspect the final diff and agent-owned history against the intended parent.
Keep coherent commits. If curation is useful, prepare the concrete rewrite
described in "Approval and handoff" and use existing authorization or obtain
approval for that specific rewrite. A new follow-up commit is often sufficient.

After an approved rewrite, verify the final tree against the recorded tip and
inspect each layer's diff. For a stack, work bottom-up and verify descendant
parents before updating them; use the installed stack tool's help rather than
assuming a command sequence.

Write the PR around its final outcome and material evidence, using the
repository template. Carry out the requested commit/push/PR actions, submit
agent-authored PRs as drafts, and stop at handoff. If only a local diff was
requested, leave it local and uncommitted.
