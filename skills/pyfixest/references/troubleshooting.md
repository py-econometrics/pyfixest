# Troubleshooting and model state

When a call fails or a result method is unavailable:

1. Confirm the installed PyFixest version and inspect the installed signature.
2. Reduce the example to one estimator, one formula, and a small data sample.
3. Check column names, data types, missing values, formula syntax, and whether
   optional dependencies are installed.
4. Check whether weights, fixed effects, IV, or a multiple-estimation result
   changes feature support.
5. Check whether `store_data=False` or `lean=True` removed state needed by the
   requested post-estimation operation.
6. Preserve the full exception and warnings; do not catch them and guess a
   substitute estimator.

For formula failures, use the bundled
[formula guide](../../../pyfixest/docs/pages/tutorials/formula-syntax.md). For
demeaning or convergence problems, use the
[difficult-fixed-effects explanation](../../../pyfixest/docs/pages/explanation/difficult-fixed-effects.md).
For differences from R `fixest`, consult the
[comparison guide](../../../pyfixest/docs/pages/explanation/compare-fixest-pyfixest.md).

Underscore-prefixed attributes are internal implementation details. If the
public API does not expose a value, consult the installed API documentation or
ask for a supported path rather than depending on private state.
