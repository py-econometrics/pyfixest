# Workflow: implement or clean up a pyfixest feature PR

This is the tool-neutral feature workflow reached through `AGENTS.md`. Read
the root guide first. Keep attempts bounded and do not widen a task to make a
check pass.

## Phase 1 — Discover and classify

1. Resolve the PR base and record literal values for `BASE_REF` and
   `MERGE_BASE`. Prefer the remote PR base; if fetch/auth fails, use the
   current remote-tracking ref and disclose that it may be stale.
2. Inspect the issue or PR description, changed files, and full diff.
3. Map changes to API, models, internals, post-estimation, formula, Rust/core,
   tests, docs, CI, packaging, or dependencies.
4. Remove formatting-only churn, generated artifacts, unrelated refactors, and
   accidental lock changes unless explicitly in scope.
5. Read the nearest implementation and its tests end to end.
6. Classify the design as an estimator add-on, post-estimation feature, shared
   primitive, kernel, or justified core change. Core expansion needs maintainer
   design approval before implementation.
7. Write a short plan containing the base, files, wiring, support matrix,
   external numerical reference, test tiers, and excluded churn.

Use [`implementation-strategy`](skills/implementation-strategy/SKILL.md) for
this classification.

## Phase 2 — Implement in reviewable slices

Use at most three attempts per slice. A typical feature is divided into:

1. contracts/helpers and tests;
2. implementation/wiring and integration tests;
3. exports, documentation, and changelog.

Run the narrowest meaningful test after each edit:

```bash
pixi run -e py312-r pytest <relevant tests> -x -q --no-cov
```

Methods remain thin; numerical logic stays in functions. Reuse formula, data,
cluster, and RNG helpers. Validate at the API boundary. Define weights, fixed
effects, IV, multiple-estimation, `lean`, and `store_data` behavior
explicitly.

## Phase 3 — Establish numerical correctness

For every new estimator, add a permanent comparison with existing software.
For numerical changes to an existing estimator, do the same wherever behavior
overlaps an external implementation. Prefer live R packages available through
conda, then CRAN-only packages, then stored output with its generator.

The comparison records versions, seed/data, formula, vcov/SSC, weights, named
outputs, and justified tolerances. Cover supported fixed-effect, weighted, IV,
clustered, and multiple-estimation paths. Add closed-form, brute-force, edge,
and simulation checks as useful, but never in place of the external reference
for a new estimator.

If no existing software can validate a new estimator, stop: it is not
merge-ready under project policy.

Use [`numerical-validation`](skills/numerical-validation/SKILL.md) to choose
and record the reference.

## Phase 4 — Verify and review

Inspect committed, staged, uncommitted, and untracked work. Run checks from
narrowest to broadest: targeted tests, changed-file format/lint/type checks, PR
baseline, then required domain suites. Long suites run after the design
stabilizes. Record commands, results, and durations; identify deferred checks
and where they will run.

Preview the deterministic selection with:

```bash
pixi run agent-verify --base <immediate-parent> --tier pr --dry-run
```

Then run the required tier without `--dry-run`. Use `--tier domain` after
numerical, docs, HAC, Rust, or reference changes stabilize.

Review the diff twice at most for:

- estimator-specific logic leaking into the core;
- public compatibility and intentional fixest differences;
- silent unsupported weights/FE/IV/multiple-estimation behavior;
- input-data mutation and stripped-data failures;
- unjustified tolerances or missing external references;
- generated or unrelated churn;
- missing exports, executable examples, guides, and changelog entries;
- unmeasured performance claims or Python hot loops that need a kernel.

Use [`change-verification`](skills/change-verification/SKILL.md) for test
selection and reporting, then
[`pyfixest-pr-review`](skills/pyfixest-pr-review/SKILL.md) for the final
diff review.

## Phase 5 — Curate commit history

Curate each PR layer before its first remote submission. For a stack, work from
the bottom layer upward and treat the immediate parent as the layer base.

Preconditions:

1. The current branch is named and is not `master`.
2. `git status --short` is clean.
3. The immediate parent and merge base are verified.
4. The branch is agent-owned and unpushed, or the user explicitly approved a
   rewrite. Never rewrite contributor-owned history.
5. Record the original tip SHA before changing history.

Inspect `git log --oneline <parent>..HEAD`. Keep a few coherent commits,
usually contracts/helpers/tests → implementation/wiring/tests → exports/docs.
Remove WIP, fixup, accidental, and formatting-only commits. Each reconstructed
commit must pass its applicable targeted checks.

Use interactive rebase where available. In an agent sandbox, the guarded
alternative is:

```bash
git reset --soft <verified-layer-base>
# stage and recommit one coherent slice at a time
```

Afterward, verify that the shipped tree did not change:

```bash
git diff --exit-code <original-tip> HEAD
git log --oneline <parent>..HEAD
```

For dependent layers, run `gh stack rebase --upstack`. After review starts,
add review-response commits rather than rewriting silently. A final cleanup
requires maintainer approval, `gh stack push` (never raw
`git push --force`), and renewed approval if GitHub dismisses stale reviews.

Use [`history-curation`](skills/history-curation/SKILL.md) for this
safety-sensitive phase.

## Phase 6 — Prepare the PR or stack

Use one PR for a small cohesive change. Prefer `gh stack` when two or more
independently reviewable concerns can form coherent layers. Split by dependency
and reviewer concern; keep unrelated changes out.

Before submission:

1. Verify each layer against its immediate parent.
2. Verify the cumulative top against the trunk.
3. Inspect `gh stack view` and every layer's log/diff.
4. Prepare the PR template with architecture placement, support matrix,
   external reference, exact commands/durations, deferred checks, commit
   narrative, docs, changelog, and performance evidence.
5. Submit as drafts. Mark ready only when required checks pass or explicitly
   identified long checks are running in CI.

Every layer requires human-maintainer approval. Agents stop at review handoff
and never run `gh stack merge`.

Use [`pr-draft-summary`](skills/pr-draft-summary/SKILL.md) to prepare the PR
body and final handoff.

## Escalation

Stop and report exact files, commands, output, and the best hypothesis when:

- the same failure repeats three times;
- the design requires unapproved shared-core expansion;
- correctness against an external reference cannot be established;
- an unsafe history rewrite would be required; or
- the fix appears to need out-of-scope lockfiles, generated files, or unrelated
  refactoring.
