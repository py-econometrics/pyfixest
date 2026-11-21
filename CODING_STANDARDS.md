# pyfixest coding standards

The rules a reviewer checks on every diff. Decisions and their reasons are
ADRs in [`docs/developer/adr/`](docs/developer/adr/); differences from `fixest`
are in the [compatibility ledger](docs/developer/fixest-compatibility.md).
Rules that ruff or ty enforce are not repeated here.

## Design

- Follow the nearest implementation. Estimator-specific work lives in its own
  modules, model methods orchestrate, and numerical work lives in standalone
  functions. Put measured, non-vectorizable hot loops in Rust; use NumPy where
  it is clear and sufficient.
- Prefer code a maintainer can follow from inputs to results. An abstraction
  earns its place by clarifying an operation, owning an invariant, or serving
  real callers, not by adding a layer to pass through. Before removing a short
  method, check its callers and subclass overrides: it may be a dispatch
  boundary.
- Keep one authoritative representation of each value: no parallel access
  paths, and no copies without an ownership reason.
- Model methods validate, unpack state, call numerical functions with keyword
  arguments, and store or return results. Numerical functions take arrays and
  return small typed results with documented shapes. Splitting a solver loop
  must not obscure the math or harm compilation.
- Constructors assemble configuration and child objects; they do not estimate.
  `FixestMulti` is a container and does no numerical work.

## Results and numerical boundaries

- Define and test a new estimator's or inference feature's behavior under
  `aweights`, `fweights`, fixed effects, IV, multiple estimation, retention,
  and the relevant backends. For an existing feature, revisit only the paths
  the change affects.
- Check capabilities before reading estimation state. An unsupported
  combination raises `NotImplementedError` naming it; a supported operation
  whose state was removed raises a retention error.
- Never reinterpret GLM working arrays or quantile output as linear-model
  state. Reject a refit path that cannot replay the original estimation.
- `vcov()` is the only operation that updates a fitted model in place. Any
  other operation that cannot rebuild a complete fitted model returns its value.
- Never apply a weight twice because its first application is hidden in
  another component.

## Names

- Public names, docstrings, errors, and docs use [glossary](GLOSSARY.md)
  terms. A term the glossary lists under _Avoid_ is a finding.
- Names expose mathematical meaning, numerical scale, and ordering. Code has
  two array scales the glossary does not cover: formula data (filtered,
  formula-materialized values) and solver data (local arrays transformed for
  a solve, such as `design_sqrt_weighted`).
- Use econometric names such as `scores`, `meat`, `bread`, `u_hat`, and
  `clustid` where they match the quantity; no aliases for the same quantity.

## Code

- PEP 604 unions, keyword arguments for internal calls, and
  `NDArray[np.float64]` in stubs. Shared typed option aliases live in
  `estimation/internals/literals.py`.
- Validate options at the API boundary with a `ValueError` naming the allowed
  values; raise the `pyfixest/errors/` classes for domain failures. An error
  names the rejected input and, when useful, a supported alternative.
- Guard optional imports, and raise an error naming the pip extra when the
  path that needs one is used.
- Use `np.random.default_rng(seed)`, never global seeding. Never mutate user
  input except through the documented `copy_data=False` path.
- One module per public estimation function; model module names end in `_`.

## Documentation

- Internal docstrings start with what the function does. Add shapes, units,
  ordering, ownership, or math only where the contract is not obvious. Cite
  the method's paper at its numerical implementation. No tutorials in helpers,
  no migration narration.
- Public functions, methods, and classes have NumPy docstrings with complete
  Parameters and Returns, executable `{python}` examples, root-relative `.qmd`
  links, and linked papers. New public entries need exports and quartodoc
  registration; explain how users obtain a result object.
- Describe behavior plainly. "As in fixest" belongs in tests and the ledger;
  explain a difference to users only where it affects their choices. Extend
  the nearest user guide rather than adding a vignette per change.
- Add a concise changelog entry for features, behavior or default changes, bug
  fixes, deprecations, performance changes, and new contributor tooling. Skip
  internal refactors, guidance edits, CI tweaks, and typo fixes.
- Preserve stable public behavior or use an agreed deprecation path; a cleanup
  is not permission to break it.
