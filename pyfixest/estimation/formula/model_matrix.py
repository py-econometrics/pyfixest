import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Optional, Union

import formulaic
import numpy as np
import pandas as pd
from formulaic.parser import DefaultFormulaParser

from pyfixest.estimation import detect_singletons
from pyfixest.estimation.formula import FORMULAIC_FEATURE_FLAG
from pyfixest.estimation.formula.factor_interaction import factor_interaction
from pyfixest.estimation.formula.parse import Formula, _Pattern
from pyfixest.estimation.formula.utils import log
from pyfixest.utils.utils import capture_context


def _factorize(series: pd.Series, encode_null: bool = False) -> np.ndarray:
    factorize: bool = not pd.api.types.is_numeric_dtype(series)
    if factorize:
        factorized, _ = pd.factorize(series, use_na_sentinel=True)
    else:
        factorized = series.to_numpy()
    if not encode_null and factorize:
        # Keep nulls (otherwise they are encoded as -1 when use_na_sentinel=True)
        factorized = np.where(factorized == -1, np.nan, factorized)
    return factorized


def _interact_fixed_effects(fixed_effects: str, data: pd.DataFrame) -> pd.DataFrame:
    fes = re.split(_Pattern.variables, fixed_effects)
    for fixed_effect in fes:
        if "^" not in fixed_effect:
            continue
        # Encode interacted fixed effects
        vars = fixed_effect.split("^")
        data[fixed_effect.replace("^", "_")] = (
            data[vars[0]]
            .astype(pd.StringDtype())
            .str.cat(
                data[vars[1:]].astype(pd.StringDtype()),
                sep="^",
                na_rep=None,  # a row containing a missing value in any of the columns (before concatenation) will have a missing value in the result
            )
        )
    return data.loc[:, [fe.replace("^", "_") for fe in fes]]


def _encode_fixed_effects(fixed_effects: str, data: pd.DataFrame) -> pd.DataFrame:
    data = _interact_fixed_effects(fixed_effects, data)
    return data.apply(_factorize, axis=0)


def _get_weights(data: pd.DataFrame, weights: str) -> pd.Series:
    if weights not in data.columns:
        raise ValueError(f"The weights column '{weights}' is not a column in the data.")
    w = data[weights]
    try:
        w = pd.to_numeric(w, errors="raise")
    except ValueError:
        raise ValueError(f"The weights column '{weights}' must be numeric.")
    if not (w.dropna() > 0.0).all():
        raise ValueError(
            f"The weights column '{weights}' must have only non-negative values."
        )
    return w


@dataclass(frozen=True, kw_only=True)
class _ModelMatrixKey:
    main: str = "fml_second_stage"
    fixed_effects: str = "fe"
    instrumental_variable: str = "fml_first_stage"
    weights: str = "weights"


class ModelMatrix:
    @property
    def dependent(self) -> pd.DataFrame:
        return self._data[self._dependent]

    @property
    def independent(self) -> pd.DataFrame:
        return self._data[self._independent]

    @property
    def fixed_effects(self) -> Optional[pd.DataFrame]:
        if self._fixed_effects is None:
            return None
        else:
            return self._data[self._fixed_effects]

    @property
    def endogenous(self) -> Optional[pd.DataFrame]:
        if self._endogenous is None:
            return None
        else:
            return self._data[self._endogenous]

    @property
    def instruments(self) -> Optional[pd.DataFrame]:
        if self._instruments is None:
            return None
        else:
            return self._data[self._instruments]

    @property
    def weights(self) -> Optional[pd.DataFrame]:
        if self._weights is None:
            return None
        else:
            return self._data[self._weights]

    @property
    def model_spec(self) -> formulaic.ModelSpec:
        return self._model_spec

    def __init__(
        self,
        model_matrix: formulaic.ModelMatrix,
        drop_rows: set[int],
        drop_singletons: bool = True,
    ) -> None:
        self._model_spec = model_matrix.model_spec
        self._collect_columns(model_matrix)
        self._collect_data(model_matrix)
        self._process(dropped_rows=drop_rows, drop_singletons=drop_singletons)

    def _collect_columns(self, model_matrix: formulaic.ModelMatrix) -> None:
        mapping: dict[str, tuple[str, str | None]] = {
            "_dependent": (_ModelMatrixKey.main, "lhs"),
            "_independent": (_ModelMatrixKey.main, "rhs"),
            "_fixed_effects": (_ModelMatrixKey.fixed_effects, None),
            "_endogenous": (_ModelMatrixKey.instrumental_variable, "lhs"),
            "_instruments": (_ModelMatrixKey.instrumental_variable, "rhs"),
            "_weights": (_ModelMatrixKey.weights, None),
        }
        for attribute, (key1, key2) in mapping.items():
            try:
                columns = (
                    model_matrix[key1].columns
                    if key2 is None
                    else model_matrix[key1][key2].columns
                )
            except KeyError:
                columns = None
            setattr(self, attribute, columns)

    def _collect_data(self, model_matrix: formulaic.ModelMatrix) -> None:
        data: list[pd.DataFrame] = list(model_matrix._flatten())
        if not all(data[0].index.identical(other.index) for other in data[1:]):
            raise ValueError("All design matrix data must have the same index.")
        self._data = pd.concat(data, ignore_index=False, axis=1)

    def _process(self, dropped_rows: set[int], drop_singletons: bool = False) -> None:
        # Drop rows with non-finite values
        is_infinite = ~np.isfinite(self._data).all(axis=1)
        if is_infinite.any():
            dropped_rows |= set(self._data.index[is_infinite])
            self._data.drop(self._data.index[is_infinite], inplace=True)
        if drop_singletons and self.fixed_effects is not None:
            # Drop singletons
            is_singleton = detect_singletons(self.fixed_effects.astype("int32").values)
            if is_singleton.any():
                dropped_rows |= set(self._data.index[is_singleton])
                self._data.drop(self._data.index[is_singleton], inplace=True)
        if self.fixed_effects is not None:
            # Intercept not meaningful in the presence of fixed effects
            self._independent = self._independent.drop("Intercept", errors="ignore")
            self._instruments = self._instruments.drop("Intercept", errors="ignore")

        self.na_index_str = ",".join(str(i) for i in dropped_rows)


def get(
    formula: Formula,
    data: pd.DataFrame,
    weights: str | None = None,
    drop_singletons: bool = False,
    ensure_full_rank: bool = True,
    context: Union[int, Mapping[str, Any]] = 0,
) -> ModelMatrix:
    """

    Parameters
    ----------
    formula: Formula
    data: pd.DataFrame
    weights: str or None
    drop_singletons: bool
    ensure_full_rank: bool
    context : int or Mapping[str, Any]
        A dictionary containing additional context variables to be used by
        formulaic during the creation of the model matrix. This can include
        custom factorization functions, transformations, or any other
        variables that need to be available in the formula environment.

    Returns
    -------
    ModelMatrix

    """
    # Process input data
    data.reset_index(drop=True, inplace=True)  # Sanitise index
    n_observations: Final[int] = data.shape[0]
    # Collate kwargs to be passed to formulaic.Formula
    formula_kwargs: dict[str, str] = {
        _ModelMatrixKey.main: formula.fml_second_stage
    }  # Main formula
    if formula.fixed_effects is not None:
        # Encode fixed effects as integers to prevent categorical encoding
        # This is because fixed effects are partialled out in the demeaning step and not directly estimated
        encoded_fixed_effects = _encode_fixed_effects(formula.fixed_effects, data)
        data[encoded_fixed_effects.columns] = encoded_fixed_effects
        formula_kwargs.update(
            {
                _ModelMatrixKey.fixed_effects: f"{'+'.join(encoded_fixed_effects.columns)}-1"
            }
        )
    if formula.fml_first_stage is not None:
        # Instrumental variable
        formula_kwargs.update(
            {_ModelMatrixKey.instrumental_variable: formula.fml_first_stage}
        )
    if weights is not None:
        data[weights] = _get_weights(data, weights)
        formula_kwargs.update({_ModelMatrixKey.weights: f"{weights}-1"})
    model_matrix = formulaic.Formula(
        formula_kwargs,
        _parser=DefaultFormulaParser(feature_flags=FORMULAIC_FEATURE_FLAG),
    ).get_model_matrix(
        data=data,
        ensure_full_rank=ensure_full_rank,
        na_action="drop",
        output="pandas",
        context={
            "log": log,  # custom log settings infinite to nan
            "i": factor_interaction,  # fixest::i()-style syntax
        }
        | {**capture_context(context)},
    )
    drop_rows: set[int] = set(range(n_observations)).difference(
        model_matrix[_ModelMatrixKey.main]["lhs"].index
    )
    return ModelMatrix(
        model_matrix, drop_rows=drop_rows, drop_singletons=drop_singletons
    )
