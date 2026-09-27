"""Storage policy for fitted-model state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyfixest.errors import MissingModelDataError

if TYPE_CHECKING:
    from pyfixest.estimation.models.feols_ import Feols


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


def apply_retention(model: Feols) -> None:
    """Drop the state the fitted model's storage options omit.

    The estimation functions pass this to `run_estimation`, which applies it
    to each model as soon as it is fitted. The fitting pipeline itself
    returns complete models, so internal refits keep every attribute they
    read.
    """
    model._clear_attributes()


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
