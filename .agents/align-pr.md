# Workflow: align a contributor PR with house style

Audit an existing PR against `AGENTS.md` and produce either a paste-ready
review or a fixed-up local branch, so the maintainer does as little as
possible. `.agents/feature-pr.md` is for *building* a change; this workflow is
for *auditing and repairing* one that already exists. Read `AGENTS.md` first —
it is the rulebook; this file is only the loop that applies it.

Input: a PR number (given several, run the whole workflow per PR
independently; never mix branches). Two modes, chosen per invocation:

- **report** (default) — findings only: one markdown review the maintainer can
  paste into GitHub. Post nothing yourself unless explicitly asked.
- **fix** — apply the mechanical and structural fixes on a local branch
  `align/pr-<n>` cut from the PR head; leave judgment calls as findings.

Hard rules in both modes: never push anywhere, never force-push or rewrite the
contributor's branch, never post comments or reviews, unless the maintainer
explicitly asks for that step. `align/pr-<n>` is a local copy and is yours to
rewrite.

## Phase 0 — Intake (run once)

1. `gh pr checkout <n>`; in fix mode branch off to `align/pr-<n>`. Record
   `BASE_REF` and `MERGE_BASE` exactly as in `.agents/feature-pr.md` Phase 1.
2. Staleness check: list files the PR touches that no longer exist on the base
   (`git diff --name-only "$MERGE_BASE"` vs `git ls-tree -r --name-only
   "$BASE_REF"`). A PR written against a pre-refactor tree (moved docs pages,
   relocated modules) gets a finding naming the moved targets; in fix mode
   attempt one rebase onto `BASE_REF` — conflicts beyond the trivial become a
   finding, not a fight.
3. Classify the change surface into the Phase 1 buckets of `feature-pr.md`
   (feature logic / tests / docs / formatting churn / generated artifacts /
   unrelated refactor) and identify the change type — post-estimation feature,
   vcov type, API function, Rust kernel, estimation-time option. The type
   selects the wiring checklist from AGENTS.md "Where new code goes".
4. Check for overlapping open PRs (`gh pr list --search "<topic>"`). Duplicate
   or complementary efforts (e.g. a NumPy implementation of a kernel another
   PR is porting to Rust) are a maintainer question — suggest how they could
   be joined (one PR's code often becomes the other's test reference), do not
   resolve it yourself.

## Phase 1 — Mechanical audit

This phase is automated. Run, in order:

```bash
pixi run -e lint prek run ruff-check --files <changed>   # RNG, style, typing
pixi run python scripts/check_house_style.py <changed>   # tree checks
pixi run python scripts/check_house_style.py --diff "$MERGE_BASE"
```

Every hit is a finding; in fix mode, apply it. Do not hand-report what these
already catch, and do not re-derive their rules here — `check_house_style.py`
is the detector list, and its module docstring says how to extend it. When a
review finding turns out to be mechanically detectable, add a check there
rather than a row here.

Two mechanical rules have no detector yet — check them by eye:

- **Narration comments.** Comments restating the next line; keep only the ones
  stating a constraint the code cannot.
- **Public/private twin.** A public `f()` whose body is only `return _f(...)`
  with the same signature is one function too many.

## Phase 2 — Structural audit (placement and wiring)

Walk the AGENTS.md "Where new code goes" checklist for the change type from
Phase 0, item by item, and record every missing wiring point. The recurring
gaps, in the order they bite:

1. **Logic mass in a model class.** Count added lines in `models/*.py`.
   Anything beyond validate-and-delegate (plus the docstring) moves to
   `post_estimation/` or `internals/`; the method keeps only input checks and
   the call.
2. **vcov wiring.** A new vcov type needs all of: the `literals.py` entry,
   `_check_vcov_input` / `_deparse_vcov_input`, a small `_vcov_<name>`
   dispatch method, meat/bread math in `internals/vcov_utils.py` (or Rust),
   ssc through `_make_ssc_kwargs`, and threading through `FixestMulti.vcov()`
   (and quantreg where applicable). Ad-hoc `self._<name>_*` state scraped out
   of `vcov_kwargs.get(...)` inside `Feols.vcov` is the tell that this
   checklist was skipped.
3. **API surface.** New function: export chain (`api/__init__.py`,
   `estimation/__init__.py`, `pyfixest/__init__.py` — `__all__`,
   `_lazy_imports`, `_direct_module_imports`), quartodoc `contents` in
   `docs/_quarto.yml`, a changelog entry in `docs/changelog.qmd`, and a
   signature whose order matches the siblings (`fml, data, vcov, …`).
4. **Hot loops.** A per-observation or per-pair double loop in Python or numba
   becomes a Rust kernel in `src/`, alongside the existing ones; the contributed
   Python implementation is kept as the test reference, not the shipped path.
5. **Unsupported paths.** For each of: IV (`Feiv`), GLM/Poisson, quantreg,
   weights (`aweights`/`fweights`), fixed effects, multiple estimation
   (`FixestMulti`), `lean=True` / `store_data=False` — the feature either
   supports it *and tests it*, or raises informatively
   (`NotImplementedError`, `VcovTypeNotSupportedError`). Silent wrong numbers
   on these paths are the number-one review concern; flag any path that is
   neither tested nor blocked.

## Phase 3 — Econometrics audit

The correctness bar from AGENTS.md → "Testing", strongest gap first:

1. **External reference.** At least one of: R via rpy2 (marked
   `against_r_core`/`against_r_extended`), stored Stata/R output under
   `tests/data/` with its generator, a brute-force reimplementation in the
   test, a closed-form collapse onto an existing estimator, or a seeded Monte
   Carlo size/coverage property. Shape checks and "SE is in a plausible
   range" do not count. If R `fixest` itself supports the feature, a
   comparison against it is the expected reference — its absence is always a
   finding.
2. **Tolerances.** Any `rtol`/`atol` at or looser than `1e-2`, or one loosened
   relative to sibling tests, needs a written justification; absent one, flag
   it — do not tighten blindly in fix mode, since the looseness may be hiding
   a real discrepancy.
3. **Cheap invariants.** vcov symmetric with positive diagonal, row-order
   invariance, weights collapsing to unweighted at equal weights — present
   where the feature makes them cheap.
4. **Spot-check the math** against the paper cited in the docstring: kernel
   weights, small-sample corrections, the IV projection
   (`tXZ @ tZZinv @ meat @ tZZinv @ tZX`). If you cannot establish
   correctness, say so explicitly in the report — never bless numbers you
   could not verify, and never "fix" econometrics you are unsure about.

## Phase 4 — Output

**Report mode.** One markdown review per PR, findings ordered by severity:
(1) wrong or unverifiable econometrics, (2) silently-wrong-number paths,
(3) missing reference tests, (4) wiring/placement gaps, (5) mechanical style.
Each finding: `file:line`, the AGENTS.md rule (quote its phrase), and the
concrete fix. Close with **Questions for the maintainer** — naming choices,
scope, duplicate PRs, default parameter values: decisions this workflow must
not make alone. Contributors are volunteers: open with what the PR does well,
and phrase findings against the written guide, not against the person.

Ratchet (both modes): a finding you could not anchor to an AGENTS.md phrase is
a gap in the guide — propose the AGENTS.md edit (and, if mechanically
checkable, the Phase 1 table row or lint rule) alongside the review.

**Fix mode.** On `align/pr-<n>`: apply Phase 1 fixes, then Phase 2, in
batches; after each batch run the narrowest relevant tests plus the three lint
hooks (max 3 red-green iterations per batch, then stop and downgrade the rest
to findings). Finish with the commit-history rewrite from `feature-pr.md`
Phase 5. Hand off: what was fixed, what remains as findings (including
everything from Phase 3 you did not touch), the diff stat, and the ready
branch name — pushing it or opening a PR is the maintainer's call.

## Escalation

Same triggers as `feature-pr.md`: the same failure three times, generated
files in the way, or correctness that cannot be established → stop and
report. In addition, always escalate rather than decide: user-facing naming,
API shape changes, default values with econometric content (cutoffs,
kernels, ssc), and whether overlapping PRs should be merged.
