"""Retention of completed components and the resources used to produce them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pyfixest.errors import MissingModelDataError


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Choose raw-data and observation-sized retention independently."""

    store_data: bool = True
    lean: bool = False


# The stored data and the formula state built from it.
_DATA_ATTRIBUTES: tuple[str, ...] = ("_data", "model_matrix", "_cluster_df")

# Components and transitional fit products whose size depends on observations.
# Compact inference and performance measures are retained.
_OBSERVATION_ATTRIBUTES: tuple[str, ...] = (
    "within_data",
    "working_state",
    "observation_weights",
    "_scores",
    "_u_hat",
    "_Y_hat_link",
    "_Y_hat_response",
    "_X_hat",
    "_v_hat",
    "_sumFE",
    "_alpha",
    "_y_hat_null",
    "_x_final",
    "_s_final",
    "_z_final",
    "_w_final",
    "_y_final",
    "_model_spec",
    "_context",
    "_input_index",
    "_sample_positions",
)


def omitted_attributes(policy: RetentionPolicy) -> tuple[str, ...]:
    """Name the attributes a fitted model drops under `policy`."""
    names: tuple[str, ...] = ()
    if not policy.store_data or policy.lean:
        names += _DATA_ATTRIBUTES
    if policy.lean:
        names += _OBSERVATION_ATTRIBUTES
    return names


def require_retained(model, operation: str, *names: str) -> None:
    """Fail before `operation` touches attributes omitted by the retention policy."""
    missing = [name for name in names if not hasattr(model, name)]
    if missing:
        remedy = (
            "Refit with store_data=True and lean=False."
            if any(name in _DATA_ATTRIBUTES for name in missing)
            else "Refit with lean=False."
        )
        raise MissingModelDataError(
            f"{operation} requires retained {', '.join(missing)}. {remedy}"
        )


def formula_context(model_spec, context: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only context names used by materialized formula factors.

    In particular, captured stack frames often contain the input data and
    earlier fits. They must not become a second owner of discarded storage.
    Explicitly used functions remain available, including their closures.
    """
    names = {
        str(variable).split(".")[0]
        for spec in model_spec._flatten()
        for variable in spec.variables
    }
    return {name: context[name] for name in names if name in context}
