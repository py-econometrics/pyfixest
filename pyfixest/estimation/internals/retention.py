"""Storage policy for fitted-model state."""

from __future__ import annotations

from dataclasses import dataclass

from pyfixest.errors import MissingModelDataError


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Record the fitted-model storage options."""

    store_data: bool = True
    lean: bool = False


_STORE_DATA_ATTRIBUTES: tuple[str, ...] = ("_data", "model_matrix")

_LEAN_ATTRIBUTES: tuple[str, ...] = (
    "_data",
    "model_matrix",
    "sandwich",
    "_u_hat",
    "fitted_values",
    "working_state",
    "within_data",
    "observation_weights",
)


def omitted_attributes(policy: RetentionPolicy) -> tuple[str, ...]:
    """Name the attributes a fitted model drops under `policy`."""
    names: list[str] = []
    if not policy.store_data:
        names.extend(_STORE_DATA_ATTRIBUTES)
    if policy.lean:
        names.extend(_LEAN_ATTRIBUTES)
    return tuple(dict.fromkeys(names))


def require_retained(model, operation: str, *names: str) -> None:
    """Fail before `operation` touches attributes omitted by the retention policy."""
    policy = model.options.retention
    omitted = set(omitted_attributes(policy))
    missing = []
    for name in names:
        if hasattr(model, name):
            continue
        if name not in omitted:
            getattr(model, name)
        missing.append(name)
    if missing:
        remedy = "Refit with store_data=True and lean=False."
        if policy.store_data and policy.lean:
            remedy = "Refit with lean=False."
        raise MissingModelDataError(
            f"{operation} requires retained model state omitted by the storage "
            f"options: {', '.join(missing)}. {remedy}"
        )
