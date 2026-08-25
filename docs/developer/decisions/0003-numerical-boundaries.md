# 0003 — Standalone numerics and native hot loops

Status: accepted

## Context

Numerical logic embedded in model objects is hard to test independently, while
premature low-level optimization obscures the econometrics.

## Decision

Model methods validate, orchestrate, and delegate. Standalone functions operate
on arrays and return typed result objects. Plain NumPy is the default. Move a
loop to Rust only after measurement shows that it is performance-critical and
cannot be expressed clearly through vectorized NumPy.

## Consequences

Numerical seams can be compared directly with external or reference
implementations. Native kernels retain readable reference implementations when
feasible.
