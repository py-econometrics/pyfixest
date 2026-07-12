"""Validate and normalize inputs shared by public estimation functions."""

from __future__ import annotations

import pandas as pd

from pyfixest.typing import (
    DataFrameType,
    QuantregVcovType,
    RegressionVcovType,
    SscConfig,
    VcovKwargs,
    WeightsType,
)
from pyfixest.utils.dev_utils import _narwhals_to_pandas


def _estimation_input_checks(
    fml: str,
    data: DataFrameType,
    vcov: RegressionVcovType | QuantregVcovType | dict[str, str] | None,
    vcov_kwargs: VcovKwargs | None,
    weights: None | str,
    ssc: SscConfig,
    fixef_rm: str,
    collin_tol: float,
    copy_data: bool,
    store_data: bool,
    lean: bool,
    weights_type: WeightsType,
    reps: int | None,
    seed: int | None,
    split: str | None,
    fsplit: str | None,
    separation_check: list[str] | None = None,
) -> None:
    if not isinstance(fml, str):
        raise TypeError(
            f"`fml` must be a string; received {type(fml).__name__}: {fml!r}. "
            "See `pyfixest/docs/pages/tutorials/formula-syntax.md` or "
            "https://pyfixest.org/formula-syntax.html."
        )
    if not isinstance(data, pd.DataFrame):
        data = _narwhals_to_pandas(data)
    if not isinstance(vcov, (str, dict, type(None))):
        raise TypeError(
            "`vcov` must be a string, a clustering dictionary, or None; "
            f"received {type(vcov).__name__}: {vcov!r}. See "
            "`pyfixest/docs/pages/tutorials/standard-errors.md` or "
            "https://pyfixest.org/standard-errors.html."
        )
    if not isinstance(fixef_rm, str):
        raise TypeError(
            f"`fixef_rm` must be a string; received {type(fixef_rm).__name__}: "
            f"{fixef_rm!r}."
        )
    if not isinstance(collin_tol, float):
        raise TypeError(
            f"`collin_tol` must be a float in (0, 1); received "
            f"{type(collin_tol).__name__}: {collin_tol!r}."
        )

    if fixef_rm not in ["none", "singleton"]:
        raise ValueError(
            f"Invalid `fixef_rm` value {fixef_rm!r}; expected 'none' or 'singleton'."
        )
    if collin_tol <= 0:
        raise ValueError(
            f"`collin_tol` must be greater than zero; received {collin_tol!r}."
        )
    if collin_tol >= 1:
        raise ValueError(
            f"`collin_tol` must be less than one; received {collin_tol!r}."
        )

    if not (isinstance(weights, str) or weights is None):
        raise TypeError(
            "`weights` must be a column name or None; received "
            f"{type(weights).__name__}: {weights!r}."
        )
    if weights is not None and weights not in data.columns:
        raise ValueError(
            f"The `weights` column {weights!r} was not found in `data`. Pass the "
            "name of an existing numeric column or set `weights=None`."
        )

    bool_args = {
        "copy_data": copy_data,
        "store_data": store_data,
        "lean": lean,
    }
    for name, arg in bool_args.items():
        if not isinstance(arg, bool):
            raise TypeError(
                f"`{name}` must be a bool; received {type(arg).__name__}: {arg!r}."
            )

    if weights_type not in ["aweights", "fweights"]:
        raise ValueError(
            f"Invalid `weights_type` value {weights_type!r}; expected 'aweights' "
            "(analytic/precision weights) or 'fweights' (frequency weights)."
        )

    if reps is not None:
        if not isinstance(reps, int):
            raise TypeError(
                f"`reps` must be an int; received {type(reps).__name__}: {reps!r}."
            )

        if reps <= 0:
            raise ValueError(f"`reps` must be strictly positive; received {reps!r}.")

    if seed is not None and not isinstance(seed, int):
        raise TypeError(
            f"`seed` must be an int or None; received {type(seed).__name__}: {seed!r}."
        )

    if split is not None and not isinstance(split, str):
        raise TypeError(
            f"`split` must be a column name or None; received "
            f"{type(split).__name__}: {split!r}."
        )

    if fsplit is not None and not isinstance(fsplit, str):
        raise TypeError(
            f"`fsplit` must be a column name or None; received "
            f"{type(fsplit).__name__}: {fsplit!r}."
        )

    if split is not None and fsplit is not None and split != fsplit:
        raise ValueError(
            "`split` and `fsplit` cannot name different columns in the same "
            f"call; received split={split!r} and fsplit={fsplit!r}. Choose "
            "`split` to fit groups only or `fsplit` to include the full sample."
        )

    if isinstance(split, str) and split not in data.columns:
        raise KeyError(f"Column '{split}' not found in data.")

    if isinstance(fsplit, str) and fsplit not in data.columns:
        raise KeyError(f"Column '{fsplit}' not found in data.")

    if separation_check is not None:
        if not isinstance(separation_check, list):
            raise TypeError(
                "`separation_check` must be a list containing 'fe' and/or 'ir'; "
                f"received {type(separation_check).__name__}: {separation_check!r}."
            )

        if not all(x in ["fe", "ir"] for x in separation_check):
            raise ValueError(
                f"Invalid `separation_check` value {separation_check!r}; expected "
                "a list containing only 'fe' and/or 'ir'."
            )

    if vcov_kwargs is not None:
        # check that dict keys are either "lag", "time_id", or "panel_id"
        if not all(key in ["lag", "time_id", "panel_id"] for key in vcov_kwargs):
            raise ValueError(
                f"Invalid `vcov_kwargs` keys {list(vcov_kwargs)!r}; accepted "
                "keys are 'lag', 'time_id', and 'panel_id'. See "
                "`pyfixest/docs/pages/tutorials/standard-errors.md` or "
                "https://pyfixest.org/standard-errors.html."
            )

        # if lag provided, check that it is an int
        if "lag" in vcov_kwargs and not isinstance(vcov_kwargs["lag"], int):
            raise TypeError(
                "`vcov_kwargs['lag']` must be an int; received "
                f"{type(vcov_kwargs['lag']).__name__}: {vcov_kwargs['lag']!r}."
            )

        if "time_id" in vcov_kwargs:
            if not isinstance(vcov_kwargs["time_id"], str):
                raise TypeError(
                    "`vcov_kwargs['time_id']` must be a column name; received "
                    f"{type(vcov_kwargs['time_id']).__name__}: "
                    f"{vcov_kwargs['time_id']!r}."
                )
            if vcov_kwargs["time_id"] not in data.columns:
                raise ValueError(
                    f"The variable '{vcov_kwargs['time_id']}' is not in the data."
                )

        if "panel_id" in vcov_kwargs:
            if not isinstance(vcov_kwargs["panel_id"], str):
                raise TypeError(
                    "`vcov_kwargs['panel_id']` must be a column name; received "
                    f"{type(vcov_kwargs['panel_id']).__name__}: "
                    f"{vcov_kwargs['panel_id']!r}."
                )
            if vcov_kwargs["panel_id"] not in data.columns:
                raise ValueError(
                    f"The variable '{vcov_kwargs['panel_id']}' is not in the data."
                )


class _AllSampleSentinel:
    """Sentinel representing the full sample in fsplit mode."""

    def __repr__(self) -> str:
        return "all"


_ALL_SAMPLE = _AllSampleSentinel()
