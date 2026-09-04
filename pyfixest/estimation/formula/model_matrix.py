from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, TypeAlias, cast

import formulaic
import numpy as np
import pandas as pd
from formulaic.parser import DefaultFormulaParser
from numpy.typing import NDArray

from pyfixest.core.detect_singletons import detect_singletons
from pyfixest.estimation.formula import FORMULAIC_FEATURE_FLAG, FORMULAIC_TRANSFORMS
from pyfixest.estimation.formula.formulaic_compat import (
    AnyModelMatrix,
    flatten_model_matrix,
    model_spec_lhs,
)
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.formula.utils import _get_weights
from pyfixest.utils.utils import capture_context

_ModelSpecMapping: TypeAlias = Mapping[str, formulaic.ModelSpec]


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

    Once constructed, the instance is the formula state a fitted model retains
    and is treated as read-only. Estimator-level row filters such as GLM
    separation call `without_rows`, which returns a filtered copy instead of
    mutating the instance.

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
    model_spec : Mapping[str, formulaic.ModelSpec]
        The underlying formulaic model specifications keyed by role.
    na_index : frozenset[int]
        Indices of rows that were dropped.
    """

    def __init__(
        self,
        model_matrix: AnyModelMatrix,
        drop_rows: frozenset[int],
        drop_singletons: bool = True,
        drop_intercept: bool = False,
    ) -> None:
        self._drop_intercept = drop_intercept
        self._model_spec = cast(_ModelSpecMapping, model_matrix.model_spec)
        self._na_index = drop_rows
        self._collect_columns(model_matrix)
        self._collect_data(model_matrix)
        self._process(drop_singletons=drop_singletons)

    @staticmethod
    def _get_columns(mm: AnyModelMatrix, *keys: str) -> list[str] | None:
        """Extract column names by traversing nested keys, or None if missing."""
        try:
            result = mm
            for k in keys:
                result = result[k]
            return result.columns.tolist()
        except KeyError:
            return None

    def _collect_columns(self, model_matrix: AnyModelMatrix) -> None:
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

    def _collect_data(self, model_matrix: AnyModelMatrix) -> None:
        datas = flatten_model_matrix(model_matrix)
        if not all(datas[0].index.identical(other.index) for other in datas[1:]):
            raise ValueError("All design matrix data must have the same index.")
        data = pd.concat(datas, ignore_index=False, axis=1)
        self._data = data.loc[:, ~data.columns.duplicated()]

    def _process(self, drop_singletons: bool = False) -> None:
        if model_spec_lhs(self.model_spec[_ModelMatrixKey.main]).factor_contrasts:
            raise TypeError("The dependent variable must be numeric.")
        elif self._dependent is None or len(self._dependent) != 1:
            raise TypeError("The model must contain exactly one dependent variable.")

        if self._endogenous is not None:
            if model_spec_lhs(
                self.model_spec[_ModelMatrixKey.instrumental_variable]
            ).factor_contrasts:
                raise TypeError("The endogenous variable must be numeric.")
            elif len(self._endogenous) != 1:
                raise TypeError(
                    "The model must contain exactly one endogenous variable."
                )

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

        if self._offset is not None:
            if self.model_spec[_ModelMatrixKey.offset].factor_contrasts:
                raise TypeError("The offset must be numeric.")
            elif len(self._offset) != 1:
                raise ValueError("The offset must evaluate to exactly one column.")

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
        self._na_index = self._na_index.union(self._data.index[is_dropped].tolist())
        self._data = self._data.loc[~is_dropped]
        warnings.warn(f"{n_dropped} {reason} dropped from the model.")

    def without_rows(self, rows: list[int]) -> ModelMatrix:
        """Return a copy without ``rows``; they are recorded in ``na_index``.

        Estimator-level filters such as GLM separation run after construction and
        call this instead of mutating the retained instance, which is left
        unchanged.
        """
        if not rows:
            return self
        filtered = copy.copy(self)
        filtered._data = self._data.drop(index=rows)
        filtered._na_index = self._na_index.union(rows)
        return filtered

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
        Get the evaluated offset for GLM estimation.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing the evaluated offset expression, which is
            added to the linear predictor with a fixed coefficient of 1, or
            None if no offset is specified.
        """
        if self._offset is None:
            return None
        else:
            return self._data.loc[:, self._offset]

    @property
    def model_spec(self) -> _ModelSpecMapping:
        """
        Get the underlying formulaic model specification.

        Returns
        -------
        Mapping[str, formulaic.ModelSpec]
            Formulaic specifications keyed by model-matrix role.
        """
        return self._model_spec

    @property
    def na_index(self) -> frozenset[int]:
        """Integer positions of dropped rows, including ``without_rows`` drops."""
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
        Formulaic expression that evaluates to one numeric offset column. The
        offset is added to the linear predictor with a fixed coefficient of 1.
        Rows with missing offset values are dropped together with missing rows
        in the rest of the formula.
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

    Examples
    --------
    ```{python}
    import pyfixest as pf
    from pyfixest.estimation.formula.model_matrix import create_model_matrix
    from pyfixest.estimation.formula.parse import Formula

    data = pf.get_data()
    formula = Formula.parse("Y ~ X1 + f1 + f2")[0]
    model_matrix = create_model_matrix(formula=formula, data=data)
    ```
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
        context=FORMULAIC_TRANSFORMS | {**capture_context(context)},
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
    if formula.is_fixed_effects:
        formula_kwargs.update(
            {_ModelMatrixKey.fixed_effects: f"{formula.fixed_effects_wrapped} - 1"}
        )
    if formula.is_instrumental_variable:
        formula_kwargs.update(
            {_ModelMatrixKey.instrumental_variable: formula.first_stage}
        )
    if weights is not None:
        data[weights] = _get_weights(data, weights)
        formula_kwargs.update({_ModelMatrixKey.weights: f"{weights}-1"})
    if offset is not None:
        formula_kwargs[_ModelMatrixKey.offset] = f"{offset} - 1"
    formula_formulaic = formulaic.Formula(
        formula_kwargs,
        _parser=DefaultFormulaParser(
            feature_flags=FORMULAIC_FEATURE_FLAG,
            # When FEs are present, include_intercept=True so that spans_intercept=True
            # terms (like i()) receive reduced_rank=True from formulaic, causing them to
            # drop the first level (matching R/fixest). The intercept column is removed
            # afterwards in ModelMatrix._process(). Without this, i() would receive
            # reduced_rank=False and generate all levels; the post-hoc collinearity check
            # would then drop the last level instead of the first, mismatching R.
            include_intercept=formula.is_fixed_effects,
        ),
    )

    return formula_formulaic
