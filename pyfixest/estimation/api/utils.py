from __future__ import annotations

from math import isfinite
from numbers import Real

import pandas as pd

from pyfixest.utils.dev_utils import DataFrameType, _narwhals_to_pandas


def _estimation_input_checks(
    fml: str,
    data: DataFrameType,
    vcov: str | dict[str, str] | None,
    vcov_kwargs: dict[str, str | int | float] | None,
    weights: None | str,
    ssc: dict[str, str | bool],
    fixef_rm: str,
    collin_tol: float,
    copy_data: bool,
    store_data: bool,
    lean: bool,
    weights_type: str,
    use_compression: bool,
    reps: int | None,
    seed: int | None,
    split: str | None,
    fsplit: str | None,
    separation_check: list[str] | None = None,
):
    if not isinstance(fml, str):
        raise TypeError("fml must be a string")
    if not isinstance(data, pd.DataFrame):
        data = _narwhals_to_pandas(data)
    if not isinstance(vcov, (str, dict, type(None))):
        raise TypeError("vcov must be a string, dictionary, or None")
    if not isinstance(fixef_rm, str):
        raise TypeError("fixef_rm must be a string")
    if not isinstance(collin_tol, float):
        raise TypeError("collin_tol must be a float")

    if fixef_rm not in ["none", "singleton"]:
        raise ValueError("fixef_rm must be either 'none' or 'singleton'.")
    if collin_tol <= 0:
        raise ValueError("collin_tol must be greater than zero")
    if collin_tol >= 1:
        raise ValueError("collin_tol must be less than one")

    if not (isinstance(weights, str) or weights is None):
        raise ValueError(
            f"weights must be a string or None but you provided weights = {weights}."
        )
    if weights is not None:
        assert weights in data.columns, "weights must be a column in data"

    bool_args = [copy_data, store_data, lean]
    for arg in bool_args:
        if not isinstance(arg, bool):
            raise TypeError(f"The function argument {arg} must be of type bool.")

    if weights_type not in ["aweights", "fweights"]:
        raise ValueError(
            f"""
            The `weights_type` argument must be of type `aweights`
            (for analytical / precision weights) or `fweights`
            (for frequency weights) but it is {weights_type}.
            """
        )

    if not isinstance(use_compression, bool):
        raise TypeError("The function argument `use_compression` must be of type bool.")
    if use_compression and weights is not None:
        raise NotImplementedError(
            "Compressed regression is not supported with weights."
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
            f"""
                        Arguments split and fsplit are both specified, but not identical.
                        split is specified as {split}, while fsplit is specified as {fsplit}.
                        """
        )

    if isinstance(split, str) and split not in data.columns:
        raise KeyError(f"Column '{split}' not found in data.")

    if isinstance(fsplit, str) and fsplit not in data.columns:
        raise KeyError(f"Column '{fsplit}' not found in data.")

    if separation_check is not None:
        if not isinstance(separation_check, list):
            raise TypeError(
                "The function argument `separation_check` must be of type list."
            )

        if not all(x in ["fe", "ir"] for x in separation_check):
            raise ValueError(
                "The function argument `separation_check` must be a list of strings containing 'fe' and/or 'ir'."
            )

    if vcov == "conley":
        if vcov_kwargs is None:
            raise ValueError(
                "Missing required vcov_kwargs for Conley standard errors. "
                "Please provide 'lat', 'lon', and 'cutoff'."
            )

        allowed_conley_keys = ["lat", "lon", "cutoff", "distance"]
        if not all(key in allowed_conley_keys for key in vcov_kwargs):
            raise ValueError(
                "The function argument `vcov_kwargs` must be a dictionary with keys "
                "'lat', 'lon', 'cutoff', or 'distance' for Conley standard errors."
            )

        required_conley_keys = ["lat", "lon", "cutoff"]
        missing_keys = [key for key in required_conley_keys if key not in vcov_kwargs]
        if missing_keys:
            raise ValueError(
                "The function argument `vcov_kwargs` must contain 'lat', 'lon', "
                "and 'cutoff' for Conley standard errors."
            )

        for key in ["lat", "lon"]:
            if not isinstance(vcov_kwargs[key], str):
                raise TypeError(
                    "The function argument `vcov_kwargs` must be a dictionary with "
                    f"string values for '{key}' if explicitly provided."
                )
            if vcov_kwargs[key] not in data.columns:
                raise ValueError(
                    f"The variable '{vcov_kwargs[key]}' is not in the data."
                )
            if not pd.api.types.is_numeric_dtype(data[vcov_kwargs[key]]):
                name = "latitude" if key == "lat" else "longitude"
                raise ValueError(f"The {name} variable must be numeric.")

        cutoff = vcov_kwargs["cutoff"]
        if not isinstance(cutoff, Real) or isinstance(cutoff, bool):
            raise TypeError(
                "The function argument `vcov_kwargs` must be a dictionary with a "
                "numeric value for 'cutoff' if explicitly provided."
            )
        if not isfinite(float(cutoff)) or cutoff < 0:
            raise ValueError(
                "The function argument `vcov_kwargs` must contain a non-negative "
                "finite value for 'cutoff'."
            )

        if "distance" in vcov_kwargs:
            distance = vcov_kwargs["distance"]
            if not isinstance(distance, str):
                raise ValueError(
                    "The function argument `vcov_kwargs` must be a dictionary with "
                    "a string value for 'distance' if explicitly provided."
                )
            if distance not in ["triangular", "spherical"]:
                raise ValueError(
                    "The Conley distance must be either 'triangular' or 'spherical'."
                )
        return

    if vcov_kwargs is not None:
        # check that dict keys are either "lag", "time_id", or "panel_id"
        if not all(key in ["lag", "time_id", "panel_id"] for key in vcov_kwargs):
            raise ValueError(
                "The function argument `vcov_kwargs` must be a dictionary with keys 'lag', 'time_id', or 'panel_id'."
            )

        # if lag provided, check that it is an int
        if "lag" in vcov_kwargs and not isinstance(vcov_kwargs["lag"], int):
            raise ValueError(
                "The function argument `vcov_kwargs` must be a dictionary with integer values for 'lag' if explicitly provided."
            )

        if "time_id" in vcov_kwargs:
            if not isinstance(vcov_kwargs["time_id"], str):
                raise ValueError(
                    "The function argument `vcov_kwargs` must be a dictionary with string values for 'time_id' if explicitly provided."
                )
            if vcov_kwargs["time_id"] not in data.columns:
                raise ValueError(
                    f"The variable '{vcov_kwargs['time_id']}' is not in the data."
                )

        if "panel_id" in vcov_kwargs:
            if not isinstance(vcov_kwargs["panel_id"], str):
                raise ValueError(
                    "The function argument `vcov_kwargs` must be a dictionary with string values for 'panel_id' if explicitly provided."
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
