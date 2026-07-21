# LsmrDemeaner

``` python
LsmrDemeaner(
    fixef_maxiter=1000,
    backend='within',
    precision='float64',
    device='auto',
    fixef_atol=1e-08,
    fixef_btol=1e-08,
    warn_on_cpu_fallback=True,
    preconditioner='auto',
    local_size=None,
)
```

Sparse LSMR demeaner.

Solves the demeaning problem as a single sparse least squares system with LSMR instead of alternating projections. See [Choosing a Demeaner Backend](../how-to/demeaner-backends.llms.md) for when to use which backend.

## Examples

``` python
import pyfixest as pf

data = pf.get_data()

fit = pf.feols("Y ~ X1 | f1 + f2", data, demeaner=pf.LsmrDemeaner())
fit.tidy()
```

|             | Estimate  | Std. Error | t value    | Pr(\>\|t\|) | 2.5%   | 97.5%    |
|-------------|-----------|------------|------------|-------------|--------|----------|
| Coefficient |           |            |            |             |        |          |
| X1          | -0.919255 | 0.059997   | -15.321564 | 0.0         | -1.037 | -0.80151 |

## Notes

The ``` within`` backend takes a single tolerance, so ```fixef_atol`and`fixef_btol`are collapsed to`max(fixef_atol, fixef_btol)`for that backend. The`torch\` backend uses both tolerances independently (SciPy LSMR convention).

The `local_size` field only applies to `backend="within"`; the `torch` backend ignores it.

The ``` precision``, ```device`` , and `warn_on_cpu_fallback `` fields are only relevant for the `torch` backend. The `within` backend always runs on CPU in float64 and ignores these fields.

`preconditioner` selects the preconditioner. Supported values:

- `"auto"` (default): selects different preconditioners for different backend implementations: `"additive"` for `"within"`; `"diagonal"` for `"torch"`.
- `"off"`: disables preconditioning. Supported by `"within"`; not supported by `"torch"`.
- `"additive"`: additive Schwarz preconditioner. Only supported by the `"within"` backend.
- `"diagonal"`: diagonal (Jacobi) preconditioner. Supported by `"within"` and `"torch"`.
- A :class:`pyfixest.Preconditioner` instance: a previously built preconditioner (typically obtained via `fit.preconditioner` or pickled across sessions). Only supported by `backend='within'`; preconditioners are only computed and applied for two or more fixed-effect factors because single-factor problems run MAP as the within algo provides no benefits. Passing a preconditioner to any other backend raises `ValueError` at construction time.

If a *string* value is incompatible with the chosen backend, a `UserWarning` is emitted at solve time and the backend’s default is used. A `Preconditioner` paired with a non-`within` backend is rejected eagerly with `ValueError` because there is no sensible fallback for a prebuilt object.

## Methods

| Name | Description |
|----|----|
| [LsmrDemeaner.demean](#pyfixest.demeaners.LsmrDemeaner.demean) | Demean `x` by the fixed effects in `flist` via LSMR. |
| [LsmrDemeaner.with_tol](#pyfixest.demeaners.LsmrDemeaner.with_tol) | Overwrite LSMR tolerances (used for IWLS acceleration). |

### LsmrDemeaner.demean

``` python
demean(x, flist, weights=None, cached_preconditioner=None)
```

Demean `x` by the fixed effects in `flist` via LSMR.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| cached_preconditioner | Preconditioner or None | A preconditioner saved by the caller from an earlier within solve on the same fixed-effect design. This is separate from `self.preconditioner`: the latter is the user’s requested configuration, while `cached_preconditioner` is the model’s internal “reuse this if it still matches” handle. The cache is used only when the current request is a string preconditioner with the same variant (`"additive"` or `"diagonal"`). If the user explicitly supplied a `Preconditioner` on the demeaner, that object is passed through and the model cache is ignored. | `None` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | tuple\[np.ndarray, bool, Preconditioner \| None\] | The demeaned array, a convergence flag, and the within preconditioner actually used during the solve. The third element is `None` for non-within backends, when `preconditioner='off'` was requested, or when the single-FE MAP fallback path was taken inside `demean_within` — in those cases no preconditioner participated in the solve. Callers (e.g. the `DemeanCache`) can cache the returned instance to amortise setup across subsequent solves on the same design. |

### LsmrDemeaner.with_tol

``` python
with_tol(tol)
```

Overwrite LSMR tolerances (used for IWLS acceleration).
