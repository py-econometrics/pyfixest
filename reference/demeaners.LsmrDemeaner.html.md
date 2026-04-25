# demeaners.LsmrDemeaner { #pyfixest.demeaners.LsmrDemeaner }

```python
demeaners.LsmrDemeaner(
    fixef_maxiter=10000,
    backend='cupy',
    precision='float64',
    device='auto',
    fixef_atol=1e-08,
    fixef_btol=1e-08,
    warn_on_cpu_fallback=True,
    use_preconditioner=True,
)
```

Sparse LSMR demeaner for CuPy/SciPy and PyTorch backends.