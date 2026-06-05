PyFixest exposes several demeaner backends via the `demeaner=` argument.
All backends produce numerically equivalent results; the choice only affects **performance**, which
itself depends on the structure of the fixed effects and the size of the dataset.

For a deep dive on *why* some fixed-effects problems are harder than others — and detailed
benchmarks comparing all backends — see
[When Are Fixed Effects Estimations Difficult?](../explanation/difficult-fixed-effects.md).

```{python}
import pyfixest as pf
from pyfixest.demeaners import LsmrDemeaner, MapDemeaner

df = pf.get_data()
```

## Quick Reference

What are **dense / well-connected** vs **sparse / weakly-connected** fixed-effect structures?
The typical example for "dense" fixed effects is a balanced panel - each unit appears exactly once for every single time period. In contrast, a "sparse" or "weakly" connected fixed effect structure arises when units appear in only a few time periods. For example, in a matched employer-employee panel, each worker may only appear in a handful of firms.


| Backend | When to use |
|---|---|
| `MapDemeaner(backend="rust")` | **Default.** Fast for dense fixed effects with well-connected graphs. No optional dependencies required. |
| `MapDemeaner(backend="numba")` | An alternative MAP implementation written in `numba`. Requires installation via `pip install pyfixest[numba]`. |
| `LsmrDemeaner()` | Uses modified LSMR with additive Schwarz preconditioning by default via the `within` crate. Best for **sparse** fixed-effect structures (e.g. matched employer-employee data). Also supports no preconditioner - which we never recommend - and a diagonal preconditioner, which can be a good choice for dense fixed effects.|
| `LsmrDemeaner(backend="torch", device="cpu")` | LSMR solver on CPU via PyTorch. Requires `pytorch`.|
| `LsmrDemeaner(backend="torch", device="cuda")` | LSMR solver on CUDA GPUs via PyTorch. Experimental. Requires `pytorch`, plus a CUDA-enabled PyTorch build (see [PyTorch install instructions](https://pytorch.org/get-started/locally/)). |
| `LsmrDemeaner(backend="torch", device="mps", precision="float32")` | LSMR solver on Apple Silicon GPUs via PyTorch MPS. Experimental. In our experience, MPS provides little performance improvement, but we'd love to hear about differing experiences! Requires `pytorch`. |

Since release `0.60.0`, `pyfixest` defaults to the Rust-backed MAP demeaner.
Numba is now an optional dependency: you can install it via `pip install pyfixest[numba]`
if you want to use `MapDemeaner(backend="numba")` or the fast `ritest` path.
The torch-based LSMR backends require `pytorch`, which you can install manually or via `pip install pyfixest[torch]`. For GPU acceleration on CUDA, you additionally need a CUDA-enabled PyTorch wheel from the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

::: {.callout-warning}
The `jax` MAP demeaner backend and the `cupy` / `scipy` LSMR demeaner backends
are deprecated and will be removed in a future release. Switch to the
recommendations above:

- Users running `MapDemeaner(backend="jax")` on CPU should use the default
  `MapDemeaner()` (rust MAP).
- Users running `MapDemeaner(backend="jax")` or
  `LsmrDemeaner(backend="cupy", device="cuda")` on GPU should use
  `LsmrDemeaner(backend="torch", device="cuda")`.
- Users running the legacy `scipy` LSMR backend (which corresponds to
  `LsmrDemeaner(backend="cupy", device="cpu")`) should use
  `LsmrDemeaner()` (the default `within` backend).
:::

## Usage

To select a demeaner backend, you need to pass an instance of the desired demeaner to
the `demeaner` argument of `feols`:

```{python}
# Default (Rust MAP): best for dense FE / well-connected graphs
fit = pf.feols("Y ~ X1 | f1", data=df)

# Rust MAP – explicit
fit = pf.feols("Y ~ X1 | f1", data=df, demeaner=MapDemeaner(backend="rust"))

# Numba MAP (requires pip install pyfixest[numba])
fit = pf.feols("Y ~ X1 | f1", data=df, demeaner=MapDemeaner(backend="numba"))

# `within` LSMR backend – best for sparse multi-way FE structures.
# Defaults to the additive Schwarz preconditioner.
fit = pf.feols("Y ~ X1 | f1 + f2", data=df, demeaner=LsmrDemeaner())

# `within` LSMR with the additive Schwarz preconditioner - the default;
# best for sparse / weakly connected fixed effects.
fit = pf.feols(
    "Y ~ X1 | f1 + f2",
    data=df,
    demeaner=LsmrDemeaner(preconditioner="additive"),
)

# `within` LSMR with the diagonal (Jacobi) preconditioner - cheaper to set up,
# often a good choice for denser / well-connected fixed effects.
fit = pf.feols(
    "Y ~ X1 | f1 + f2",
    data=df,
    demeaner=LsmrDemeaner(preconditioner="diagonal"),
)

# you can also turn off the preconditioner; this is almost never recommended
# we only offer it for benchmarking / users who are curious to learn how much
# the preconditioner helps
fit = pf.feols(
    "Y ~ X1 | f1 + f2",
    data=df,
    demeaner=LsmrDemeaner(preconditioner="off"),
)
```

The `MapDemeaner` accepts `fixef_tol` and `fixef_maxiter` arguments to control
convergence. `LsmrDemeaner` uses `fixef_atol` and `fixef_btol`. For the
`within` backend the two tolerances are collapsed to
`max(fixef_atol, fixef_btol)` (the solver takes a single tolerance);
`torch` and `cupy` use both independently (SciPy LSMR convention).

```{python}
fit = pf.feols(
    "Y ~ X1 | f1",
    data=df,
    demeaner=MapDemeaner(backend="rust", fixef_tol=1e-10, fixef_maxiter=50_000),
)
```

## Caching and Pickling the `within` Preconditioner

The `within` LSMR backend builds an additive Schwarz preconditioner from the
fixed-effect design before the iterative solve for two or more fixed-effect
factors. For single fixed effects, PyFixest falls back to closed form demeaning
(via the MAP algo, which converges in one iteration) and no
preconditioner is computed or applied. For large multi-way FE problems, setting up
the preconditioner can be a meaningful fraction of total runtime, so it is worth
reusing the preconditioner across solves on the same design.

### Within a session

Every fit produced with `LsmrDemeaner(backend="within")` exposes the
preconditioner used during demeaning via the `preconditioner` attribute:

```{python}
fit = pf.feols("Y ~ X1 | f1 + f2", data=df, demeaner=LsmrDemeaner())
pre = fit.preconditioner
print(pre)
```

You can pass that instance back through `LsmrDemeaner(preconditioner=...)` on a later
fit to skip the setup phase:

```{python}
fit_reused = pf.feols(
    "Y ~ X1 | f1 + f2",
    data=df,
    demeaner=LsmrDemeaner(preconditioner=pre),
)

pf.etable([fit, fit_reused], digits = 6)
```

For GLMs that rely on iterated weighted least squares (IWLS, in `fepois`, `feglm`), either the user-provided preconditioner or the preconditioner built in the first iteration is used across all IWLS iterations. Because the working weights change across iterations, the preconditioner is "stale" with respect to the changing problem. Reusing the preconditioner does not change the target solution — when LSMR converges, it converges to the correct demeaned values — but it can take more iterations per IWLS step, and on poorly conditioned problems may fail to converge within `fixef_maxiter`. The tradeoff is paying that potential extra iteration cost in exchange for skipping the costly setup phase on every IWLS iteration.

### Across sessions

`Preconditioner` instances are pickleable, so the setup phase can be
amortised across Python sessions:

```{python}
import pickle
import tempfile, os

# session 1: build and save
fit = pf.feols("Y ~ X1 | f1 + f2", data=df, demeaner=LsmrDemeaner())
tmp = os.path.join(tempfile.gettempdir(), "precond.pkl")
with open(tmp, "wb") as f:
    pickle.dump(fit.preconditioner, f)

# session 2: load and reuse
with open(tmp, "rb") as f:
    pre = pickle.load(f)

fit2 = pf.feols(
    "Y ~ X1 | f1 + f2",
    data=df,
    demeaner=LsmrDemeaner(preconditioner=pre),
)

pf.etable([fit, fit2], digits = 6)
```

::: {.callout-warning}
`Preconditioner` instances are tied to specific fixed effect structures. The `within` crate validates that the preconditioner's DOF count matches the new design- a dimension mismatch raises `ValueError`. It does *not* check whether the FE *structure* matches when the dimensions happen to coincide; reusing on a different-but-same-size FE design will still run but may converge more slowly or fail to converge. It is the user's responsibility to ensure that the preconditioner is reused on the same FE structure.
:::
