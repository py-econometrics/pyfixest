# 0001 — Fixest parity and explicit deviations

Status: accepted

## Context

Pyfixest exists to provide Python users with the behavior and vocabulary of R
`fixest`. Silent differences make code migration and numerical review
difficult.

## Decision

User-facing behavior, names, and defaults mirror `fixest` wherever the
features overlap. Intentional differences require a rationale, permanent
external-reference tests, an entry in the compatibility ledger, and
human-maintainer review.

## Consequences

Fixest parity is the default during design and review. A discrepancy is treated
as an unresolved issue until evidence establishes that it is intentional.
