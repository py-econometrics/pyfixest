# BaseDemeaner

``` python
BaseDemeaner(fixef_maxiter=10000)
```

Base configuration shared by all fixed-effects demeaners.

Holds the settings shared by all backends, currently `fixef_maxiter`. This class is not passed to an estimation function directly. Use one of the concrete demeaners instead: [MapDemeaner](../reference/demeaners.MapDemeaner.llms.md), the default, or [LsmrDemeaner](../reference/demeaners.LsmrDemeaner.llms.md). See [Choosing a Demeaner Backend](../how-to/demeaner-backends.llms.md) for a comparison.

## Examples

``` python
import pyfixest as pf

pf.MapDemeaner(), pf.LsmrDemeaner()
```

    (MapDemeaner(fixef_maxiter=10000, fixef_tol=1e-06, backend='rust'),
     LsmrDemeaner(fixef_maxiter=1000, backend='within', precision='float64', device='auto', fixef_atol=1e-08, fixef_btol=1e-08, warn_on_cpu_fallback=True, preconditioner='auto', local_size=None))
