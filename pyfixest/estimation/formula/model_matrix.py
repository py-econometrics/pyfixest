import re
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Union

import formulaic
import numpy as np
import pandas as pd
from formulaic.parser import DefaultFormulaParser

from pyfixest.estimation import detect_singletons
from pyfixest.estimation.formula import FORMULAIC_FEATURE_FLAG
from pyfixest.estimation.formula.parse import Formula, _Pattern
from pyfixest.utils.utils import capture_context


def _factorize(series: pd.Series) -> np.ndarray:
    return pd.factorize(series, use_na_sentinel=True)[0].astype("int32")


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


def _encode_fixed_effects(
    fixed_effects: str, data: pd.DataFrame, dropna: bool = True
) -> pd.DataFrame:
    data = _interact_fixed_effects(fixed_effects, data)
    if dropna:
        data.dropna(how="any", axis=0, inplace=True)
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


@dataclass(kw_only=True, frozen=True)
class ModelMatrix:
    """A model matrix."""

    dependent: pd.DataFrame
    independent: pd.DataFrame
    model_spec: formulaic.ModelSpec
    fixed_effects: pd.DataFrame = None
    endogenous: pd.DataFrame = None
    instruments: pd.DataFrame = None
    weights: pd.DataFrame = None


def get(
    formula: Formula,
    data: pd.DataFrame,
    weights: str | None = None,
    drop_singletons: bool = False,
    context: Union[int, Mapping[str, Any]] = 0,
) -> ModelMatrix:
    """

    Parameters
    ----------
    formula: Formula
    data: pd.DataFrame
    weights: str or None
    drop_singletons: bool
    context : int or Mapping[str, Any]
        A dictionary containing additional context variables to be used by
        formulaic during the creation of the model matrix. This can include
        custom factorization functions, transformations, or any other
        variables that need to be available in the formula environment.

    Returns
    -------
    ModelMatrix

    """
    # Set infinite to null
    numeric_columns = data.select_dtypes(include="number").columns
    data[numeric_columns] = data[numeric_columns].where(
        np.isfinite(data[numeric_columns]),
        pd.NA,
    )
    formula_kwargs: dict[str, str] = {_ModelMatrixKey.main: formula.fml_second_stage}
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
        na_action="drop",
        ensure_full_rank=False,
        output="pandas",
        context={**capture_context(context)},
    )
    fixed_effects = (
        model_matrix[_ModelMatrixKey.fixed_effects].astype("int32")
        if formula.fixed_effects is not None
        else None
    )
    if drop_singletons and fixed_effects is not None:
        is_singleton = detect_singletons(fixed_effects.values)
        if is_singleton.any():
            warnings.warn(
                f"{is_singleton.sum()} singleton fixed effect(s) detected. These observations are dropped from the model."
            )
            for model in model_matrix:
                if isinstance(model, formulaic.ModelMatrices):
                    for m in model:
                        m.drop(m.index[is_singleton], inplace=True)
                else:
                    model.drop(model.index[is_singleton], inplace=True)
    return ModelMatrix(
        dependent=model_matrix[_ModelMatrixKey.main]["lhs"],
        independent=model_matrix[_ModelMatrixKey.main]["rhs"],
        model_spec=model_matrix.model_spec,
        fixed_effects=fixed_effects,
        endogenous=model_matrix[_ModelMatrixKey.instrumental_variable]["lhs"]
        if formula.fml_first_stage is not None
        else None,
        instruments=model_matrix[_ModelMatrixKey.instrumental_variable]["rhs"]
        if formula.fml_first_stage is not None
        else None,
        weights=model_matrix[_ModelMatrixKey.weights] if weights is not None else None,
    )
