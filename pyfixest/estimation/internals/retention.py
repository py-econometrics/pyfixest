"""Retention of completed components and the resources used to produce them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

from pyfixest.core.demean import Preconditioner, WithinPreconditionerName
from pyfixest.demeaners import LsmrDemeaner


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Choose raw-data and observation-sized retention independently."""

    store_data: bool = True
    lean: bool = False


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


def apply_retention(model, *, policy: RetentionPolicy) -> None:
    """Apply storage policy recursively after the fitting lifecycle completes."""
    first_stage = getattr(model, "_model_1st_stage", None)
    if first_stage is not None:
        apply_retention(first_stage, policy=policy)

    model._store_data = policy.store_data
    model._lean = policy.lean
    # Mutable caches belong to the execution block, not to statistical results.
    cache = getattr(model, "_demean_cache", None)
    if cache is not None:
        model._preconditioner = cache.lookup_preconditioner.get(model._na_index)
    model.__dict__.pop("_demean_cache", None)
    model.__dict__.pop("_input_index", None)
    if (
        policy.lean
        and isinstance(model._demeaner, LsmrDemeaner)
        and isinstance(model._demeaner.preconditioner, Preconditioner)
    ):
        model._demeaner = replace(
            model._demeaner,
            preconditioner=cast(
                WithinPreconditionerName,
                model._demeaner.preconditioner.variant.lower(),
            ),
        )

    if not policy.store_data or policy.lean:
        model.__dict__.pop("_data", None)
        model.__dict__.pop("model_matrix", None)
        model.__dict__.pop("_cluster_df", None)

    if policy.lean:
        # These are the components and transitional fit products whose size
        # depends on observations. Compact inference/performance is retained.
        for name in (
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
            "_sample_index",
            "_preconditioner",
        ):
            model.__dict__.pop(name, None)
