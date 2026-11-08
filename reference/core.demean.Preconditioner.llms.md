# Preconditioner

``` python
Preconditioner(data)
```

A pre-built preconditioner; pass it back to skip the factorisation.

## Attributes

| Name | Description |
|----|----|
| [Preconditioner.build_duration_seconds](#pyfixest.core.demean.Preconditioner.build_duration_seconds) | Time in seconds spent building this preconditioner. |
| [Preconditioner.config](#pyfixest.core.demean.Preconditioner.config) | Complete normalized configuration used to build this preconditioner. |
| [Preconditioner.ncols](#pyfixest.core.demean.Preconditioner.ncols) | Number of columns (DOFs). |
| [Preconditioner.nrows](#pyfixest.core.demean.Preconditioner.nrows) | Number of rows (DOFs). |

## Methods

| Name | Description |
|----|----|
| [Preconditioner.apply](#pyfixest.core.demean.Preconditioner.apply) | Apply the preconditioner: `y = M⁻¹ x`. |

### Preconditioner.apply

``` python
apply(x)
```

Apply the preconditioner: `y = M⁻¹ x`.
