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


def _factorize(series: pd.Series) -> np.ndarray:
    factorized, _ = pd.factorize(series, use_na_sentinel=True)
    # use_sentinel=True replaces np.nan with -1, so we revert to np.nan
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
    """
    A wrapper around formulaic.ModelMatrix for the specification of PyFixest models.

    This class organizes and processes model matrices for econometric estimation,
    extracting dependent and independent variables, fixed effects, instrumental
    variables, and weights. It handles missing data, singleton observations,
    and ensures proper formatting for estimation procedures.

    Attributes
    ----------
    dependent : pd.DataFrame
        The dependent variable(s) (left-hand side of the main equation).
    independent : pd.DataFrame
        The independent variable(s) (right-hand side of the main equation).
    fixed_effects : pd.DataFrame or None
        Fixed effects variables, encoded as integers.
    endogenous : pd.DataFrame or None
        Endogenous variables in instrumental variable specifications.
    instruments : pd.DataFrame or None
        Instrumental variables for IV estimation.
    weights : pd.DataFrame or None
        Observation weights for weighted estimation.
    model_spec : formulaic.ModelSpec
        The underlying formulaic model specification.
    na_index_str : str
        Comma-separated string of row indices that were dropped.
    """

    @property
    def dependent(self) -> pd.DataFrame:
        """
        Get the dependent variable(s) from the model.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the dependent variable(s) (left-hand side
            of the main equation).
        """
        return self._data.loc[:, self._dependent]

    @property
    def independent(self) -> pd.DataFrame:
        """
        Get the independent variable(s) from the model.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the independent variable(s) (right-hand side
            of the main equation). Intercept columns are excluded when fixed
            effects are present.
        """
        return self._data.loc[:, self._independent]

    @property
    def fixed_effects(self) -> Optional[pd.DataFrame]:
        """
        Get the fixed effects variables from the model.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing the fixed effects variables encoded as integers,
            or None if no fixed effects are specified in the model.
        """
        if self._fixed_effects is None:
            return None
        else:
            return self._data.loc[:, self._fixed_effects]

    @property
    def endogenous(self) -> Optional[pd.DataFrame]:
        """
        Get the endogenous variable(s) for instrumental variable estimation.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing the endogenous variable(s) (left-hand side
            of the first-stage equation in IV estimation), or None if not
            using instrumental variables.
        """
        if self._endogenous is None:
            return None
        else:
            return self._data.loc[:, self._endogenous]

    @property
    def instruments(self) -> Optional[pd.DataFrame]:
        """
        Get the instrumental variable(s) for IV estimation.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing the instrumental variable(s) (right-hand side
            of the first-stage equation in IV estimation), or None if not
            using instrumental variables. Intercept columns are excluded when
            fixed effects are present.
        """
        if self._instruments is None:
            return None
        else:
            return self._data.loc[:, self._instruments]

    @property
    def weights(self) -> Optional[pd.DataFrame]:
        """
        Get the observation weights for weighted estimation.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing the observation weights (must be non-negative
            numeric values), or None if no weights are specified.
        """
        if self._weights is None:
            return None
        else:
            return self._data.loc[:, self._weights]

    @property
    def model_spec(self) -> formulaic.ModelSpec:
        """
        Get the underlying formulaic model specification.

        Returns
        -------
        formulaic.ModelSpec
            The formulaic ModelSpec object containing metadata about the
            model structure and transformations.
        """
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
        # Extract dependent and independent variables (always present)
        self._dependent = model_matrix[_ModelMatrixKey.main]["lhs"].columns.tolist()
        self._independent = model_matrix[_ModelMatrixKey.main]["rhs"].columns.tolist()
        # Extract fixed effects (optional)
        try:
            self._fixed_effects = model_matrix[
                _ModelMatrixKey.fixed_effects
            ].columns.tolist()
        except KeyError:
            self._fixed_effects = None
        # Extract endogenous variables
        try:
            self._endogenous = model_matrix[_ModelMatrixKey.instrumental_variable][
                "lhs"
            ].columns.tolist()
        except KeyError:
            self._endogenous = None
        # Extract instruments
        try:
            self._instruments = model_matrix[_ModelMatrixKey.instrumental_variable][
                "rhs"
            ].columns.tolist()
        except KeyError:
            self._instruments = None
        # Extract weights (optional)
        try:
            self._weights = model_matrix[_ModelMatrixKey.weights].columns.tolist()
        except KeyError:
            self._weights = None

    def _collect_data(self, model_matrix: formulaic.ModelMatrix) -> None:
        datas: list[pd.DataFrame] = list(model_matrix._flatten())
        if not all(datas[0].index.identical(other.index) for other in datas[1:]):
            raise ValueError("All design matrix data must have the same index.")
        data = pd.concat(datas, ignore_index=False, axis=1)
        self._data = data.loc[:, ~data.columns.duplicated()]

    def _process(self, dropped_rows: set[int], drop_singletons: bool = False) -> None:
        if self.dependent.shape[1] != 1:
            # If the dependent variable is not numeric, formulaic's contrast encoding kicks in
            # creating multiple columns for the dependent variable
            # TODO: Make this check more explicit?
            raise TypeError("The dependent variable must be numeric.")
        if self.endogenous is not None and self.endogenous.shape[1] != 1:
            raise TypeError("The endogenous variable must be numeric.")
        # Drop rows with non-finite values
        is_infinite = pd.Series(
            ~np.isfinite(self._data).all(axis=1), index=self._data.index
        )
        if is_infinite.any():
            infinite_indices = is_infinite[is_infinite].index.tolist()
            dropped_rows |= set(infinite_indices)
            self._data.drop(infinite_indices, inplace=True)
            warnings.warn(
                f"{is_infinite.sum()} rows with infinite values dropped from the model.",
            )
        if drop_singletons and self.fixed_effects is not None:
            # Drop singletons
            is_singleton = pd.Series(
                detect_singletons(self.fixed_effects.astype("int32").to_numpy()),
                index=self._data.index,
            )
            if is_singleton.any():
                singleton_indices = self._data[is_singleton].index.tolist()
                dropped_rows |= set(singleton_indices)
                self._data.drop(singleton_indices, inplace=True)
                warnings.warn(
                    f"{is_singleton.sum()} singleton fixed effect(s) dropped from the model."
                )
        if self.fixed_effects is not None:
            # Intercept not meaningful in the presence of fixed effects
            self._independent = [col for col in self._independent if col != "Intercept"]
            if self._instruments is not None:
                self._instruments = [
                    col for col in self._instruments if col != "Intercept"
                ]

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
        fixed_effects = _interact_fixed_effects(
            fixed_effects=formula.fixed_effects, data=data
        )
        data[fixed_effects.columns] = fixed_effects
        formula_kwargs.update(
            {
                _ModelMatrixKey.fixed_effects: f"{'+'.join(f'__fixed_effect__({fe})' for fe in fixed_effects.columns)}-1"
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
            "__fixed_effect__": _factorize,
        }
        | {**capture_context(context)},
    )
    drop_rows: set[int] = set(range(n_observations)).difference(
        model_matrix[_ModelMatrixKey.main]["lhs"].index
    )
    return ModelMatrix(
        model_matrix, drop_rows=drop_rows, drop_singletons=drop_singletons
    )
