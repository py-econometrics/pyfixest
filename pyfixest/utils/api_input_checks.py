from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd

@dataclass
class EstimationInputs:
    fml: str
    data: pd.DataFrame
    vcov: Optional[Union[str, dict[str, str]]]
    weights: Optional[str]
    ssc: dict[str, Union[str, bool]]
    fixef_rm: str
    collin_tol: float
    copy_data: bool
    store_data: bool
    lean: bool
    fixef_tol: float
    weights_type: str
    use_compression: bool
    reps: Optional[int]
    seed: Optional[int]
    split: Optional[str]
    fsplit: Optional[str]
    separation_check: Optional[list[str]] = field(default=None)

    "Dataclass to store and check the arguments of the estimation functions feols, fepois, feglm etc."

    def validate(self):

        "Validate the arguments of the EstimationInputs class."

        # Step 1: Check types
        _check_type(self.fml, str, "fml")
        _check_type(self.data, pd.DataFrame, "data")
        _check_type(self.vcov, (str, dict, type(None)), "vcov")
        _check_type(self.weights, (str, type(None)), "weights")
        _check_type(self.ssc, dict, "ssc")
        _check_type(self.fixef_rm, str, "fixef_rm")
        _check_type(self.collin_tol, float, "collin_tol")
        _check_type(self.copy_data, bool, "copy_data")
        _check_type(self.store_data, bool, "store_data")
        _check_type(self.lean, bool, "lean")
        _check_type(self.fixef_tol, float, "fixef_tol")
        _check_type(self.weights_type, str, "weights_type")
        _check_type(self.use_compression, bool, "use_compression")
        _check_type(self.reps, (int, type(None)), "reps")
        _check_type(self.seed, (int, type(None)), "seed")
        _check_type(self.split, (str, type(None)), "split")
        _check_type(self.fsplit, (str, type(None)), "fsplit")
        _check_type(self.separation_check, (list, type(None)), "separation_check")

        # Step 2: Check values
        if isinstance(self.vcov, str):
            _check_value(self.vcov, ["iid", "HC1", "HC2", "HC3", "hetero"], "vcov")
        elif isinstance(self.vcov, dict):
            for key, value in self.vcov.items():
                if not isinstance(key, str):
                    raise TypeError(f"Key '{key}' in vcov must be a string.")
                if not isinstance(value, str):
                    raise TypeError(f"Value '{value}' in vcov must be a string.")
                _check_value(key, ["CRV1", "CRV3"], f"key: {key} in vcov")
                _check_value(value, self.data.columns, f"value: {value} in vcov")
        if self.weights is not None:
            _check_value(self.weights, self.data.columns, "weights")
        _check_value(self.fixef_rm, ["none", "singleton", "drop"], "fixef_rm")
        if not (0 < self.collin_tol < 1):
            raise ValueError("collin_tol must be in (0, 1).")
        if not (0 < self.fixef_tol < 1):
            raise ValueError("fixef_tol must be in (0, 1).")
        _check_value(self.weights_type, ["aweights", "fweights"], "weights_type")
        if not (0 < self.reps):
            raise ValueError("reps must be strictly positive.")
        if self.split is not None:
            _check_value(self.split, self.data.columns, "split")
        if self.fsplit is not None:
            _check_value(self.fsplit, self.data.columns, "fsplit")
        if self.split is not None and self.fsplit is not None:
            if self.split != self.fsplit:
                raise ValueError(
                    f"split and fsplit specified but not identical: {self.split} vs {self.fsplit}"
                )
        _check_value(self.separation_check, ["fe", "ir", None], "separation_check")

def _check_type(value, expected_type, name):
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Argument '{name}' must be {expected_type.__name__}, got {type(value).__name__}"
        )

def _check_value(value, valid_values, name):
    if value not in valid_values:
        raise ValueError(
            f"Argument '{name}' must be one of {valid_values}, got {value!r}"
        )
