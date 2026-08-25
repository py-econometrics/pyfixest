# 0002 — Stable core and estimator add-ons

Status: accepted

## Context

Adding estimator-specific branches to shared planning and fitting code makes
existing estimators harder to reason about and increases compatibility risk for
downstream users.

## Decision

New estimators begin as standalone add-on functions and modules that compose
stable formula, data, fitting, inference, and reporting primitives. Shared core
changes require a real shared consumer, a generic contract, and maintainer
design approval.

## Consequences

Estimator-specific preparation and algorithms stay local. The generic runner
and result classes remain small, and a new estimator can evolve without
silently changing established estimators.
