---
name: change-verification
description: Select and report proportionate checks for a pyfixest diff before handoff, or when verification is requested. Reuse valid evidence; this is not a trigger to rerun every suite.
---

# Verify a change

Inspect the complete diff, including relevant uncommitted and untracked files,
against the [intended base](../../../docs/developer/git-and-pr-style.md#establish-the-base).

Use the [selection matrix](../../../docs/developer/testing.md#selection-matrix)
to identify required evidence. Read only the applicable testing sections:
numerical references and release contract for numerical work, benchmarks for
performance-relevant changes, or link/configuration checks for guidance.

Inventory evidence already available for this code state, including CI. Run
the missing checks under [runtime tiers](../../../docs/developer/testing.md#runtime-tiers).
A later handoff or prose edit does not itself invalidate a passing code check.

Report the result under
[verification reporting](../../../docs/developer/testing.md#verification-reporting).
Resolve failures and material uncertainty; distinguish completed local work
from outstanding merge evidence. Do not claim a skipped comparison passed.
