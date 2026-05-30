# demeaners.WithinDemeaner { #pyfixest.demeaners.WithinDemeaner }

```python
demeaners.WithinDemeaner(
    fixef_maxiter=1000,
    fixef_tol=1e-06,
    krylov='cg',
    preconditioner='additive',
    gmres_restart=30,
)
```

Krylov-subspace demeaner implemented in Rust via the ``within`` library.