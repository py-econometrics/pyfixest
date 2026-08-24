# Formula syntax and multiple estimation

PyFixest combines Formulaic expressions with fixest-style fixed effects and
multiple-estimation operators. Read the bundled
[formula guide](../../../pyfixest/docs/pages/tutorials/formula-syntax.md) before
generating a complex formula.

## Common forms

```python
pf.feols("Y ~ X1 + X2", data=df)
pf.feols("Y ~ X1 + X2 | firm + year", data=df)
pf.feols("Y ~ X1 + [X2 ~ Z1] | firm:year", data=df)
```

- Put fixed effects after `|`.
- Use `:` for interacted fixed effects, such as `firm:year`. The old `^`
  spelling is deprecated.
- Write IV terms in brackets as `[endogenous ~ instruments]`. Legacy three-part
  IV formulas remain accepted during deprecation but should not be generated.
- Use Formulaic operators and transforms for ordinary terms. `X1 * X2` expands
  to main effects and their interaction; `X1:X2` is the interaction only.
- Use `i(variable, ref=...)` for indicators and event-study-style interactions.

## Multiple estimation

`sw`, `sw0`, `csw`, `csw0`, and `mvsw` expand one formula into a model grid.
Multiple dependent variables, `split=`, and `fsplit=` can also produce multiple
models. The result is a `FixestMulti`, not a single fitted model.

Avoid manually expanding a large grid until you have checked whether a native
operator expresses it. Native expansion can reuse work across models.
