from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import pandas as pd

from pyfixest.errors import PyfixestDeprecationWarning
from pyfixest.estimation.internals.model_state import VcovSpec
from pyfixest.utils.dev_utils import (
    DataFrameType,
    _find_stack_level,
    _narwhals_to_pandas,
)
from pyfixest.utils.utils import Ssc


class _AllSampleSentinel:
    """Represent the full sample in legacy fitted-model metadata."""

    def __repr__(self) -> str:
        return "all"


_ALL_SAMPLE = _AllSampleSentinel()


def _resolve_ssc(ssc: Ssc | Mapping[str, Any] | None) -> Ssc:
    """Return the small-sample options as an `Ssc`, defaulting to `ssc()`."""
    if ssc is None:
        return Ssc()
    if isinstance(ssc, Ssc):
        return ssc
    if isinstance(ssc, Mapping):
        return Ssc.from_mapping(ssc)
    raise TypeError(f"ssc must be created with pf.ssc(); got {type(ssc).__name__}.")


def _resolve_vcov(
    vcov: str | dict[str, str] | None,
    vcov_kwargs: Mapping[str, str | int] | None,
) -> VcovSpec:
    """Parse the requested variance estimator before any model is fitted."""
    return VcovSpec.from_user_input(
        "iid" if vcov is None else vcov, vcov_kwargs=vcov_kwargs
    )


def _warn_argument_without_effect(*, function: str, argument: str) -> None:
    """Warn that a deprecated argument of an estimation function is ignored."""
    warnings.warn(
        f"The `{argument}` argument of `{function}()` is deprecated, has no "
        "effect, and will be removed in a future release. Remove it from the call.",
        PyfixestDeprecationWarning,
        stacklevel=_find_stack_level(),
    )


def _bare_column_hint(name: str) -> str:
    """Return a hint when a column name carries a formula tilde, such as ``'~f1'``."""
    if name.startswith("~"):
        return f" Pass the bare column name, such as {name.lstrip('~').strip()!r}."
    return ""


def _estimation_input_checks(
    fml: str,
    data: DataFrameType,
    vcov: VcovSpec,
    weights: str | None,
    ssc: Ssc,
    fixef_rm: str,
    collin_tol: float,
    copy_data: bool,
    store_data: bool,
    lean: bool,
    weights_type: str,
    reps: int | None,
    seed: int | None,
    split: str | None,
    fsplit: str | None,
    separation_check: list[str] | None = None,
):
    if not isinstance(fml, str):
        raise TypeError(
            f"fml must be a formula string such as 'y ~ x'; got {type(fml).__name__}."
        )
    if not isinstance(data, pd.DataFrame):
        data = _narwhals_to_pandas(data)
    if not isinstance(fixef_rm, str):
        raise TypeError("fixef_rm must be a string")
    if not isinstance(collin_tol, float):
        raise TypeError("collin_tol must be a float")

    if fixef_rm not in ["none", "singleton"]:
        raise ValueError("fixef_rm must be either 'none' or 'singleton'.")
    if collin_tol <= 0 or collin_tol >= 1:
        raise ValueError(
            f"collin_tol must be greater than 0 and less than 1; got {collin_tol!r}."
        )

    if not (isinstance(weights, str) or weights is None):
        raise ValueError(
            f"weights must be a string or None but you provided weights = {weights}."
        )
    if weights is not None and weights not in data.columns:
        raise ValueError(
            f"weights must be a column in data; got weights={weights!r}."
            + _bare_column_hint(weights)
        )

    for name, value in (
        ("copy_data", copy_data),
        ("store_data", store_data),
        ("lean", lean),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be True or False; got {value!r}.")

    if weights_type not in ["aweights", "fweights"]:
        raise ValueError(
            f"""
            The `weights_type` argument must be of type `aweights`
            (for analytical / precision weights) or `fweights`
            (for frequency weights) but it is {weights_type}.
            """
        )

    if reps is not None:
        if not isinstance(reps, int):
            raise TypeError("The function argument `reps` must be of type int.")

        if reps <= 0:
            raise ValueError("The function argument `reps` must be strictly positive.")

    if seed is not None and not isinstance(seed, int):
        raise TypeError("The function argument `seed` must be of type int.")

    if split is not None and not isinstance(split, str):
        raise TypeError("The function argument split needs to be of type str.")

    if fsplit is not None and not isinstance(fsplit, str):
        raise TypeError("The function argument fsplit needs to be of type str.")

    if split is not None and fsplit is not None and split != fsplit:
        raise ValueError(
            f"Arguments split and fsplit are both specified, but not identical. "
            f"Got split={split!r} and fsplit={fsplit!r}. Supply one option or "
            "use the same column for both."
        )

    if isinstance(split, str) and split not in data.columns:
        raise KeyError(
            f"Column '{split}' not found in data. Pass an existing column to split."
            + _bare_column_hint(split)
        )

    if isinstance(fsplit, str) and fsplit not in data.columns:
        raise KeyError(
            f"Column '{fsplit}' not found in data. Pass an existing column to fsplit."
            + _bare_column_hint(fsplit)
        )

    if separation_check is not None:
        if not isinstance(separation_check, list):
            raise TypeError(
                "The function argument `separation_check` must be of type list."
            )

        if not all(x in ["fe", "ir"] for x in separation_check):
            raise ValueError(
                "The function argument `separation_check` must be a list of strings containing 'fe' and/or 'ir'."
            )

    for column in vcov.clustervar:
        if column not in data.columns:
            raise ValueError(
                f"The variable {column!r} is not in the data. "
                "Pass an existing clustering column, for example "
                f"vcov={{{vcov.vcov_type_detail!r}: '<column>'}}."
                + _bare_column_hint(column)
            )
    for option, column in (("time_id", vcov.time_id), ("panel_id", vcov.panel_id)):
        if column is not None and column not in data.columns:
            raise ValueError(
                f"The variable {column!r} is not in the data. "
                f"Pass an existing column to {option} in vcov_kwargs."
                + _bare_column_hint(column)
            )
