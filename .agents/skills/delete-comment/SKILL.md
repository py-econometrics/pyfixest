---
name: delete-comment
description: Audit comments in pyfixest, remove redundant and stale prose, record implementation follow-ups, keep local constraints, and move cross-module decisions into ADRs.
---

# Delete comments

Use the requested scope or default to the tracked Python source and stub files
under `pyfixest/`. Tokenize the files to find actual comments, then read each
one with its surrounding code. Consecutive lines can be reviewed together when
they explain the same thing. Keep source locations and dispositions for the
summary.

- **Tautology** repeats the code. Delete it.
- **Excuse** explains away a shortcut or workaround. Document the concrete
  implementation problem and desired behavior in an existing or new issue
  when publication is authorized, or in the requested summary. Delete the
  comment once its follow-up is captured. Keep the implementation change
  separate.
- **Stale** refers to obsolete behavior, tooling, or commented-out code.
  Confirm against the current implementation, then delete it.
- **Constraint** gives a reason the adjacent code cannot show: an invariant,
  a required order, a library or hardware limit, the source of a formula.
  Without it a reader might "fix" the code. Keep it beside the code and
  tighten its wording. It is not an ADR, however durable.
- **Decision Record** explains a choice between real alternatives that spans
  modules and is costly to reverse. Capture the decision, the rejected
  alternative, and the reason in a related [ADR](../../../docs/adr/) or a
  new one, then delete the comment. An ADR states only what stays true when
  the files it mentions are rewritten. Record intentional differences from R
  in the [compatibility ledger](../../../docs/developer/fixest-compatibility.md).

Preserve license and attribution notices, interpreter declarations, and
tooling directives until their requirements have another home. Record the disposition of deferred comments in the summary. Follow the [issue style](../../../docs/developer/git-and-pr-style.md#issues)
when preparing GitHub issues.

Confirm the diff changes only comments and related
whitespace in the package. Report comment-only lines removed separately from
inline comments stripped, and count what remains of each.
