PyFixest exposes several demeaner backends via the `demeaner=` argument. All backends produce numerically equivalent results; the choice only affects **speed**.

```{python}
import pyfixest as pf
from pyfixest.demeaners import LsmrDemeaner, MapDemeaner, WithinDemeaner

df = pf.get_data()
```

## Quick Reference

For a deeper discussion on when to use which backend, see [When Are Fixed Effects Problems Difficult?](../explanation/difficult-fixed-effects.md).


| Backend | When to use |
|---|---|
| `MapDemeaner(backend="numba")` | Default. Works well for dense fixed effects / well connected graphs. |
| `MapDemeaner(backend="rust")` | Identical algorithm to `MapDemeaner(backend="numba")`, but implemented in Rust. Despite being Rust, performance should be roughly identical to numba. |
| `WithinDemeaner()` | Best for **sparse** fixed-effect structures (e.g. matched employer-employee data). Supports additive Schwarz with CG or GMRES, multiplicative Schwarz with GMRES, and unpreconditioned solves via `preconditioner="off"`. |
| `MapDemeaner(backend="jax")` | MAP on the GPU via JAX (requires `pip install pyfixest[jax]`). |
| `LsmrDemeaner(backend="torch", device="cuda")` | GPU acceleration via PyTorch. Experimental. |
| `LsmrDemeaner(backend="cupy", device="cpu")` | LSMR solver on CPU via SciPy. |
| `LsmrDemeaner(backend = "cupy", device="cuda")` | LSMR solver on CUDA via CuPy. |

## Usage

To select a demeaner backend, simply pass an instance of the desired demeaner to the `demeaner` argument of `feols`:
```{python}
# Default (numba MAP): best for dense FE / well-connected graphs
fit = pf.feols("Y ~ X1 | f1", data=df)

# Rust MAP: roughly same performance as numba
fit = pf.feols("Y ~ X1 | f1", data=df, demeaner=MapDemeaner(backend="rust"))

# `within` backend - best for sparse multi-way FE structures
fit = pf.feols("Y ~ X1 | f1 + f2", data=df, demeaner=WithinDemeaner())

# Additive Schwarz + GMRES
fit = pf.feols(
    "Y ~ X1 | f1 + f2",
    data=df,
    demeaner=WithinDemeaner(
        krylov="gmres",
        preconditioner="additive",
        gmres_restart=30,
    ),
)

# Multiplicative Schwarz + GMRES
fit = pf.feols(
    "Y ~ X1 | f1 + f2",
    data=df,
    demeaner=WithinDemeaner(
        krylov="gmres",
        preconditioner="multiplicative",
        gmres_restart=50,
    ),
)
```

The `MapDemeaner` and `WithinDemeaner` backends accept `fixef_tol` and `fixef_maxiter` arguments to control convergence. `WithinDemeaner` additionally accepts `krylov`, `preconditioner`, and `gmres_restart` (used only with `krylov="gmres"`). The `preconditioner` argument accepts `"additive"`, `"multiplicative"`, and `"off"`; `"off"` matches `within.Preconditioner.Off`. For compatibility with `scipy`, the `LsmrDemeaner` has two arguments to control the convergence tolerance - `fixef_atol` and `fixef_btol`.

```{python}
fit = pf.feols(
    "Y ~ X1 | f1",
    data=df,
    demeaner=MapDemeaner(backend="rust", fixef_tol=1e-10, fixef_maxiter=50_000),
)
```
