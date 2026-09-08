---
name: pyfixest
description: Routes coding agents into PyFixest's bundled, version-matched documentation for fixed-effects regression in Python (feols, fepois, feglm, quantreg, difference-in-differences estimators), standard errors and inference, and regression tables. Use when code imports pyfixest or a task mentions fixed effects, clustered standard errors, IV, event studies, or fixest-style formulas.
---

# PyFixest

The installed package ships its own documentation. Locate it and check that it
is present:

    python -c "import importlib.resources as r; p = r.files('pyfixest') / 'docs'; print(p, (p / 'llms.txt').is_file())"

If it prints `True`, open `cheatsheet.llms.md` in that directory and use its
"Where to go for each task" table to choose the relevant documentation.
`llms.txt` lists every page and describes each. If it prints `False`, this is a
source checkout or a release older than the bundled docs: in a checkout start
with `docs/cheatsheet.qmd`; otherwise use https://pyfixest.org/cheatsheet.html,
which may describe a different version than the one installed.

## Core facts

```python
import pyfixest as pf
data = pf.get_data().dropna()
# fixed effects follow the first |, : interacts them (`^` is deprecated)
pf.feols("Y ~ X1 + X2 | f1 + f2", data=data)
pf.feols("Y ~ X1 | f1:f2", data=data)
# IV: a [endogenous ~ instruments] term on the right-hand side
# (the fixest-style "Y ~ X2 | f1 | X1 ~ Z1" still works but is deprecated)
pf.feols("Y ~ X2 + [X1 ~ Z1] | f1", data=data)
# i(cat) expands a categorical, ref drops a level, i(cat, x) gives slopes on x
pf.feols("Y ~ i(f1, ref=1.0) + i(f1, X2)", data=data)
# vcov spellings; NW and DK additionally need vcov_kwargs with time metadata
pf.feols("Y ~ X1", data=data, vcov="iid")
pf.feols("Y ~ X1", data=data, vcov="hetero")
pf.feols("Y ~ X1", data=data, vcov="HC3")
pf.feols("Y ~ X1", data=data, vcov={"CRV1": "f1"})
pf.feols("Y ~ X1", data=data, vcov={"CRV3": "f1"})
# two-way clustering
pf.feols("Y ~ X1", data=data, vcov={"CRV1": "f1 + f2"})
pf.etable([pf.feols("Y ~ X1 | f1", data=data)], type="md", keep="X1")
```

## Workflow

1. Choose the closest task in the cheat sheet's task table and open its narrative
   documentation.
2. Before choosing `vcov`, check the support-limits table in
   `tutorials/standard-errors.llms.md`; before writing a formula, check
   `tutorials/formula-syntax.llms.md`.
3. Check the installed API reference or function signature before writing code.
4. Search `llms.txt` only when the cheat sheet does not identify a suitable
   page.
5. Prefer public `pyfixest` functions and result methods; do not depend on
   underscore-prefixed state.
6. State unsupported combinations instead of silently substituting another
   estimator or inference method.
