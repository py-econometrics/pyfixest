import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import formulaic
import numpy as np
import pandas as pd
from formulaic.parser import DefaultFormulaParser
from numpy.typing import NDArray

from pyfixest.core.detect_singletons import detect_singletons
from pyfixest.estimation.formula import FORMULAIC_FEATURE_FLAG
from pyfixest.estimation.formula.factor_interaction import factor_interaction
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.formula.utils import (
    _encode_fixed_effects,
    _factorize,
    _get_weights,
    log,
)
from pyfixest.utils.utils import capture_context


@dataclass(frozen=True, kw_only=True)
class _ModelMatrixKey:
    main: str = "second_stage"
    fixed_effects: str = "fe"
    instrumental_variable: str = "first_stage"
    weights: str = "weights"
    offset: str = "offset"


class ModelMatrix:
    """
    A wrapper around formulaic.ModelMatrix for the specification of PyFixest models.

    This class organizes and processes model matrices for econometric estimation,
    extracting dependent and independent variables, fixed effects, instrumental
    variables, and weights. It handles missing data, singleton observations,
    and ensures proper formatting for estimation procedures.

    An internal API. Instances are built by the `prepare_model_matrix` step of
    the fit pipeline from a materialized `formulaic.ModelMatrix` and are not
    constructed directly. There is therefore no standalone example. Formulas are
    written as strings and passed to
    [feols()](/reference/estimation.api.feols.feols.qmd). See the
    [formula syntax tutorial](/tutorials/formula-syntax.qmd) for the syntax.

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
    na_index : frozenset[int]
        Indices of rows that were dropped.
    """

    def __init__(
        self,
        model_matrix: formulaic.ModelMatrix,
        drop_rows: frozenset[int],
        drop_singletons: bool = True,
        drop_intercept: bool = False,
    ) -> None:
        self._drop_intercept = drop_intercept
        self._model_spec = model_matrix.model_spec
        self._na_index = drop_rows
        self._collect_columns(model_matrix)
        self._collect_data(model_matrix)
        self._process(drop_singletons=drop_singletons)

    @staticmethod
    def _get_columns(mm: formulaic.ModelMatrix, *keys: str) -> list[str] | None:
        """Extract column names by traversing nested keys, or None if missing."""
        try:
            result = mm
            for k in keys:
                result = result[k]
            return result.columns.tolist()
        except KeyError:
            return None

    def _collect_columns(self, model_matrix: formulaic.ModelMatrix) -> None:
        self._dependent = self._get_columns(model_matrix, _ModelMatrixKey.main, "lhs")
        self._independent = self._get_columns(model_matrix, _ModelMatrixKey.main, "rhs")
        self._fixed_effects = self._get_columns(
            model_matrix, _ModelMatrixKey.fixed_effects
        )
        self._endogenous = self._get_columns(
            model_matrix, _ModelMatrixKey.instrumental_variable, "lhs"
        )
        self._instruments = self._get_columns(
            model_matrix, _ModelMatrixKey.instrumental_variable, "rhs"
        )
        self._weights = self._get_columns(model_matrix, _ModelMatrixKey.weights)
        self._offset = self._get_columns(model_matrix, _ModelMatrixKey.offset)

    def _collect_data(self, model_matrix: formulaic.ModelMatrix) -> None:
        datas: list[pd.DataFrame] = list(model_matrix._flatten())
        if not all(datas[0].index.identical(other.index) for other in datas[1:]):
            raise ValueError("All design matrix data must have the same index.")
        data = pd.concat(datas, ignore_index=False, axis=1)
        self._data = data.loc[:, ~data.columns.duplicated()]

    def _process(self, drop_singletons: bool = False) -> None:
        if self._dependent is None or len(self._dependent) != 1:
            # If the dependent variable is not numeric, formulaic's contrast encoding kicks in
            # creating multiple columns for the dependent variable
            # TODO: Make this check more explicit?
            raise TypeError("The dependent variable must be numeric.")
        if self._endogenous is not None and len(self._endogenous) != 1:
            raise TypeError("The endogenous variable must be numeric.")
        # integer and boolean columns are finite by construction
        maybe_infinite = self._data.select_dtypes(exclude=["integer", "bool"])
        self._drop(
            ~np.isfinite(maybe_infinite.to_numpy()).all(axis=1),
            "rows with infinite values",
        )
        if self._fixed_effects is not None:
            # Ensure fixed effects are `int32`
            self._data[self._fixed_effects] = self._data[self._fixed_effects].astype(
                "int32"
            )
        if self._fixed_effects is not None or self._drop_intercept:
            if self._independent is not None:
                self._independent = [
                    col for col in self._independent if col != "Intercept"
                ]
            if self._instruments is not None:
                self._instruments = [
                    col for col in self._instruments if col != "Intercept"
                ]
        # Drop singletons if specified
        if drop_singletons and self._fixed_effects is not None:
            fixed_effects = self._data.loc[:, self._fixed_effects]
            self._drop(
                detect_singletons(fixed_effects.to_numpy()),
                "singleton fixed effect(s)",
            )

    def _drop(self, is_dropped: NDArray[np.bool_], reason: str) -> None:
        """Drop the masked rows from `self._data` and add their labels to `na_index`.

        `reason` completes the warning "{n} {reason} dropped from the model."
        """
        n_dropped = int(is_dropped.sum())
        if not n_dropped:
            return
        self._na_index |= frozenset(self._data.index[is_dropped].tolist())
        self._data = self._data.loc[~is_dropped]
        warnings.warn(f"{n_dropped} {reason} dropped from the model.")

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
        cols = self._dependent or []
        return self._data[cols]

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
        cols = self._independent or []
        return self._data[cols]

    @property
    def fixed_effects(self) -> pd.DataFrame | None:
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
    def endogenous(self) -> pd.DataFrame | None:
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
    def instruments(self) -> pd.DataFrame | None:
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
    def weights(self) -> pd.DataFrame | None:
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
    def offset(self) -> pd.DataFrame | None:
        """
        Get the offset variable for GLM estimation (currently supported only for Fepois).

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing the offset variable (added to the linear
            predictor with a fixed coefficient of 1), or None if no offset
            is specified.
        """
        if self._offset is None:
            return None
        else:
            return self._data.loc[:, self._offset]

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

    @property
    def na_index(self) -> frozenset[int]:
        """Integer positions of rows dropped in model matrix creation."""
        return self._na_index


def create_model_matrix(
    formula: Formula,
    data: pd.DataFrame,
    weights: str | None = None,
    offset: str | None = None,
    drop_singletons: bool = False,
    drop_intercept: bool = False,
    ensure_full_rank: bool = True,
    context: int | Mapping[str, Any] = 0,
) -> ModelMatrix:
    """
    Create a ModelMatrix from a formula and data.

    This function constructs model matrices for econometric estimation by parsing
    formulas and extracting the necessary components (dependent/independent variables,
    fixed effects, instruments, weights) from the provided data.

    Parameters
    ----------
    formula : Formula
        A Formula object specifying the model structure, including dependent and
        independent variables, fixed effects, and instrumental variables.
    data : pd.DataFrame
        The input data containing all variables referenced in the formula.
        The index will be reset during processing.
    weights : str or None, default=None
        Column name in data to use as observation weights. Weights must be
        non-negative numeric values. If None, no weighting is applied.
    offset : str or None, default=None
        Column name in data to use as an offset (added to the linear predictor
        with a fixed coefficient of 1). Rows with NaN in the offset column are
        dropped together with NaN rows in the rest of the formula.
    drop_singletons : bool, default=False
        If True, observations that are singletons in any fixed effect category
        are dropped from the model.
    drop_intercept : bool, default=False
        If True, the intercept column is removed from the independent variables
        and instruments matrices. The intercept is always removed when fixed
        effects are present, regardless of this parameter.
    ensure_full_rank : bool, default=True
        If True, formulaic will ensure the design matrix is full rank by
        dropping collinear columns.
    context : int or Mapping[str, Any], default=0
        Additional context variables for formulaic during model matrix creation.
        Can be an integer (stack frame depth) or a dictionary of variables to
        make available in the formula environment (e.g., custom transformations).

    Returns
    -------
    ModelMatrix
        A ModelMatrix object containing the processed dependent and independent
        variables, fixed effects, instruments, weights, and metadata about
        dropped observations.

    """
    # Process input data
    data.reset_index(drop=True, inplace=True)  # Sanitise index
    n_observations: Final[int] = data.shape[0]
    formula_formulaic = _get_formulaic_formula(
        formula=formula, data=data, weights=weights, offset=offset
    )
    model_matrix = formula_formulaic.get_model_matrix(
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
    drop_rows = _dropped_rows(
        kept=model_matrix[_ModelMatrixKey.main]["lhs"].index,
        n_observations=n_observations,
    )
    return ModelMatrix(
        model_matrix,
        drop_rows=drop_rows,
        drop_singletons=drop_singletons,
        drop_intercept=drop_intercept,
    )


def _dropped_rows(kept: pd.Index, n_observations: int) -> frozenset[int]:
    """Row labels in `range(n_observations)` that `kept` does not contain.

    `create_model_matrix` resets the data index beforehand, so row labels and
    row positions coincide and the complement can be taken with a mask.
    """
    is_kept = np.zeros(n_observations, dtype=bool)
    is_kept[kept.to_numpy()] = True
    return frozenset(np.flatnonzero(~is_kept).tolist())


def _get_formulaic_formula(
    formula: Formula,
    data: pd.DataFrame,
    weights: str | None = None,
    offset: str | None = None,
) -> formulaic.Formula:
    # Collate kwargs to be passed to formulaic.Formula
    formula_kwargs: dict[str, str] = {_ModelMatrixKey.main: formula.second_stage}
    if formula.fixed_effects is not None:
        fixed_effects_formula = _encode_fixed_effects(
            fixed_effects=formula.fixed_effects, data=data
        )
        formula_kwargs.update({_ModelMatrixKey.fixed_effects: fixed_effects_formula})
    if formula.first_stage is not None:
        formula_kwargs.update(
            {_ModelMatrixKey.instrumental_variable: formula.first_stage}
        )
    if weights is not None:
        data[weights] = _get_weights(data, weights)
        formula_kwargs.update({_ModelMatrixKey.weights: f"{weights}-1"})
    if offset is not None:
        if offset not in data.columns:
            raise ValueError(f"Offset variable '{offset}' not found in data.")
        try:
            data[offset] = pd.to_numeric(data[offset], errors="raise")
        except ValueError:
            raise ValueError(f"The offset column '{offset}' must be numeric.")
        formula_kwargs.update({_ModelMatrixKey.offset: f"{offset}-1"})
    formula_formulaic = formulaic.Formula(
        formula_kwargs,
        _parser=DefaultFormulaParser(feature_flags=FORMULAIC_FEATURE_FLAG),
    )
    return formula_formulaic
