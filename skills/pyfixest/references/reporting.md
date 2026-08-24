# Tables, plots, and model reporting

Use public result methods for data and package-level functions for presentation.

- `fit.tidy()` returns a tidy table of estimates and inference.
- `fit.summary()` or `pf.summary(...)` prints model summaries.
- `pf.etable(...)` builds regression tables from a model, list, or
  `FixestMulti`.
- `pf.coefplot(...)` plots coefficient estimates.
- `pf.iplot(...)` focuses on coefficients created by `i()`.
- `pf.qplot(...)` compares quantile-regression estimates.

Read the bundled
[regression-tables tutorial](../../../pyfixest/docs/pages/tutorials/regression-tables.md)
before selecting `etable` formatting, labels, coefficient layouts, or output
types. Check the installed signatures for plotting backends and optional
dependencies.

Keep analysis separate from rendering: first fit and validate the models, then
pass the result objects to a reporting function. Do not extract private model
state to construct a table.
