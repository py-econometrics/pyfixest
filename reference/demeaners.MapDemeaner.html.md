# demeaners.MapDemeaner { #pyfixest.demeaners.MapDemeaner }

```python
demeaners.MapDemeaner(fixef_maxiter=10000, fixef_tol=1e-06, backend='rust')
```

Method of Alternating Projections (MAP) demeaner.

## Methods

| Name | Description |
| --- | --- |
| [demean](#pyfixest.demeaners.MapDemeaner.demean) | Demean `x` by the fixed effects in `flist` via MAP. |
| [with_tol](#pyfixest.demeaners.MapDemeaner.with_tol) | Override the `fixef_tol`, used for IWLS acceleration. |

### demean { #pyfixest.demeaners.MapDemeaner.demean }

```python
demeaners.MapDemeaner.demean(x, flist, weights=None, cached_preconditioner=None)
```

Demean `x` by the fixed effects in `flist` via MAP.

`cached_preconditioner` is accepted for interface uniformity with
:meth:`LsmrDemeaner.demean` and ignored: MAP does not use a
preconditioner, so the third return value is always `None`.

### with_tol { #pyfixest.demeaners.MapDemeaner.with_tol }

```python
demeaners.MapDemeaner.with_tol(tol)
```

Override the `fixef_tol`, used for IWLS acceleration.