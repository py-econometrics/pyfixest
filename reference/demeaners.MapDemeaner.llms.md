# MapDemeaner

``` python
MapDemeaner(fixef_maxiter=10000, fixef_tol=1e-06, backend='rust')
```

Method of Alternating Projections (MAP) demeaner.

The default backend. Sweeps out the fixed effects by alternately projecting on each of them until convergence. See [Choosing a Demeaner Backend](../how-to/demeaner-backends.llms.md) for when to use which backend.

## Examples

``` python
import pyfixest as pf

data = pf.get_data()

fit = pf.feols("Y ~ X1 | f1 + f2", data, demeaner=pf.MapDemeaner(fixef_tol=1e-08))
fit.tidy()
```

|             | Estimate  | Std. Error | t value    | Pr(\>\|t\|) | 2.5%   | 97.5%    |
|-------------|-----------|------------|------------|-------------|--------|----------|
| Coefficient |           |            |            |             |        |          |
| X1          | -0.919255 | 0.059997   | -15.321564 | 0.0         | -1.037 | -0.80151 |

## Methods

| Name | Description |
|----|----|
| [MapDemeaner.demean](#pyfixest.demeaners.MapDemeaner.demean) | Demean `x` by the fixed effects in `flist` via MAP. |
| [MapDemeaner.with_tol](#pyfixest.demeaners.MapDemeaner.with_tol) | Override the `fixef_tol`, used for IWLS acceleration. |

### MapDemeaner.demean

``` python
demean(x, flist, weights=None, cached_preconditioner=None)
```

Demean `x` by the fixed effects in `flist` via MAP.

`cached_preconditioner` is accepted for interface uniformity with :meth:`LsmrDemeaner.demean` and ignored: MAP does not use a preconditioner, so the third return value is always `None`.

### MapDemeaner.with_tol

``` python
with_tol(tol)
```

Override the `fixef_tol`, used for IWLS acceleration.
