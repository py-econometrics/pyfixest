"""Materialize regression roles with the Rust formula engine.

The adapter relies on behavior of the pinned ``formula`` revision that its
public API does not document. ``tests/test_formula_engine.py`` checks each
assumption, so a revision that changes one fails there first:

- A missing input, an unseen category, or an unseen fixed-effect level yields
  null in every column of the affected term, after both ``fit`` and
  ``transform``. Row validity is therefore the absence of nulls, provided
  every term emits a column.
- ``transform`` reuses the fixed-effect codes assigned by ``fit``.
- A term's fitted state depends only on its own input columns, so equal labels
  fitted on equal input data hold equal values across specifications.
- ``Model.formula`` renders a canonical formula: top-level ``+`` separates
  terms, quoted names keep their backticks, and categorical columns are
  labelled ``<canonical C() call>[<level>]``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import formula
import numpy as np
import pandas as pd
import polars as pl

from pyfixest.errors import IVIdentificationError
from pyfixest.estimation.formula.linear import (
    LinearFormula,
    bare_name,
    categorical_calls,
    rhs_terms,
    split_top_level,
)
from pyfixest.estimation.formula.model_matrix import ModelMatrix
from pyfixest.estimation.formula.parse import FixedEffectSpecification
from pyfixest.estimation.formula.transforms.fixed_effects_encoding import (
    FIXED_EFFECT_ENCODING_STATE_KEY,
)
from pyfixest.estimation.formula.utils import _get_weights
from pyfixest.estimation.internals.model_state import DroppedRowCounts


def to_polars(data: pd.DataFrame) -> pl.DataFrame:
    """Convert pandas input without carrying its index into formula evaluation."""
    # Pandas also permits numeric categories, which Polars categorical does not.
    converted = data.copy(deep=False)
    for column in data.select_dtypes(include="category"):
        dtype = data[column].cat.categories.dtype
        if pd.api.types.is_integer_dtype(dtype):
            converted[column] = data[column].astype("Int64")
        elif pd.api.types.is_bool_dtype(dtype):
            converted[column] = data[column].astype("boolean").astype("Int64")
        else:
            converted[column] = data[column].astype(dtype)
    # Native numeric expressions require a primitive numeric dtype.
    for column in data.select_dtypes(include="bool"):
        converted[column] = data[column].astype("Int64")
    return pl.from_pandas(converted, include_index=False)


def _numeric_frame(model: formula.Model) -> pd.DataFrame:
    """Preserve missing indicator values as numeric NaNs when crossing to pandas."""
    frame = model.frame()
    indicators = [name for name, dtype in frame.schema.items() if dtype == pl.Boolean]
    return frame.with_columns(pl.col(indicators).cast(pl.Float64)).to_pandas()


def _normalized_specification(
    main: formula.Model, first: formula.Model | None
) -> LinearFormula:
    """Use native canonical terms for model labels and duplicate-step detection."""
    parts = split_top_level(main.formula, "|")
    dependent, rhs = split_top_level(parts[0], "~")
    fixed_effects = parts[1] if len(parts) == 2 else None
    no_intercept = bool(re.search(r"\s*-\s*1$", rhs))
    terms = split_top_level(re.sub(r"\s*-\s*1$", "", rhs), "+")
    endogenous = instruments = None
    if first is not None:
        endogenous, first_rhs = split_top_level(
            split_top_level(first.formula, "|")[0], "~"
        )
        terms = [term for term in terms if term != endogenous]
        instrument_terms = split_top_level(re.sub(r"\s*-\s*1$", "", first_rhs), "+")
        instruments = (
            " + ".join(
                term for term in instrument_terms if term not in terms and term != "1"
            )
            or "1"
        )
        first_no_intercept = bool(re.search(r"\s*-\s*1$", first_rhs))
        if fixed_effects is None and first_no_intercept != no_intercept:
            instruments = (
                f"{instruments} - 1" if first_no_intercept else f"1 + {instruments}"
            )
    exogenous = " + ".join(term for term in terms if term != "1")
    if "1" in terms and fixed_effects is None:
        exogenous = f"1 + {exogenous}" if exogenous else "1"
    if no_intercept and fixed_effects is None:
        exogenous = f"{exogenous} - 1" if exogenous else "-1"
    return LinearFormula(
        dependent, exogenous or "1", fixed_effects, endogenous, instruments
    )


@dataclass(frozen=True)
class FittedRole:
    """Retain one native role so prediction reuses its fitted transforms."""

    model: formula.Model | None
    column_names: tuple[str, ...]
    factor_contrasts: dict[str, Any] = field(default_factory=dict)
    transform_state: dict[str, Any] = field(default_factory=dict)
    renames: dict[str, str] = field(default_factory=dict)
    bare_columns: frozenset[str] = frozenset()

    def __deepcopy__(self, memo):
        """Share read-only fitted state; native transforms return new models."""
        return self

    def get_model_matrix(
        self,
        data: pd.DataFrame,
        *,
        na_action: str = "drop",
        context: Any = None,
        output: str = "pandas",
    ) -> pd.DataFrame:
        """Transform new rows with the training model and preserve row positions."""
        if context:
            raise NotImplementedError("The formula engine does not support context.")
        if output != "pandas":
            raise ValueError("Fitted formula roles produce pandas frames.")
        result = (
            pd.DataFrame(index=data.index)
            if self.model is None
            else _numeric_frame(self.model.transform(to_polars(data)))
        )
        result.index = data.index
        result = result.rename(columns=self.renames)
        return result.dropna() if na_action == "drop" else result


@dataclass(frozen=True)
class FittedStage:
    """Fitted response and design roles belonging to one regression equation."""

    lhs: FittedRole
    rhs: FittedRole


@dataclass(frozen=True)
class FixedEffectLevels:
    """Decode native group codes for public fixed-effect estimates."""

    name: str
    values: pd.DataFrame

    @property
    def code_index(self) -> pd.Index:
        """Allocate coefficient storage for the native integer-code range."""
        return pd.RangeIndex(
            int(self.values.index.max()) + 1 if len(self.values) else 0
        )

    def decode(self, *, codes: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return original grouping values in the requested code order."""
        return tuple(
            self.values.loc[codes, column].to_numpy() for column in self.values
        )


def _role(
    model: formula.Model, role: formula.Role, data: pd.DataFrame
) -> tuple[FittedRole, pd.DataFrame]:
    try:
        selected = model.select(role)
    except KeyError:
        return FittedRole(None, ()), pd.DataFrame(index=data.index)
    frame = _numeric_frame(selected)
    frame.index = data.index
    bare_columns: frozenset[str] = frozenset()
    if role != formula.Role.FixedEffect:
        equation = split_top_level(model.formula, "|")[0]
        side = split_top_level(equation, "~")[
            0 if role == formula.Role.Dependent else 1
        ]
        calls = categorical_calls(side)
        if role == formula.Role.Dependent and calls:
            raise TypeError("The dependent and endogenous variables must be numeric.")
        for call in calls:
            # Nulls mark invalid rows only in emitted columns. A C() term with
            # no column could not mark missing or unseen categories.
            if not any(f"{call}[" in column for column in frame.columns):
                raise ValueError(
                    f"`{call}` encodes no columns: the estimation data contain no "
                    "level other than its reference. Remove the term or choose "
                    "another reference."
                )
        bare_columns = frozenset(
            name for term in rhs_terms(side) if (name := bare_name(term)) is not None
        )
    if role != formula.Role.FixedEffect:
        for column in frame:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                raise TypeError(
                    f"Formula column `{column}` must be numeric; use C() to encode categories."
                )
            if column in data and isinstance(data[column].dtype, pd.CategoricalDtype):
                raise TypeError(
                    f"Use C({column}) explicitly for a categorical regressor."
                )
    return FittedRole(selected, tuple(frame.columns), bare_columns=bare_columns), frame


class LinearModelMatrix(ModelMatrix):
    """Apply pyfixest's sample and solver contracts to native formula frames."""

    def __init__(
        self,
        *,
        specification: LinearFormula,
        data: pd.DataFrame,
        weights: str | None,
        drop_singletons: bool,
        drop_intercept: bool,
    ) -> None:
        from formulaic.parser.types import Factor, Term

        data = data.copy(deep=False)
        data.index = pd.RangeIndex(len(data))
        for role_name, source in (
            ("dependent", specification.dependent),
            ("endogenous", specification.endogenous),
        ):
            if source is None:
                continue
            name = source.strip("`").replace("``", "`")
            if name in data and (
                isinstance(data[name].dtype, pd.CategoricalDtype)
                or not pd.api.types.is_numeric_dtype(data[name])
            ):
                raise TypeError(f"The {role_name} variable must be numeric.")
        native_data = to_polars(data)
        models = formula.parse(specification.second_stage).fit(native_data)
        if len(models) != 1:
            raise NotImplementedError(
                "Expand stepwise specifications before constructing a model matrix."
            )
        main = models[0]
        lhs, dependent = _role(main, formula.Role.Dependent, data)
        rhs, independent = _role(main, formula.Role.Independent, data)
        fe_spec, fixed_effects = _role(main, formula.Role.FixedEffect, data)
        if len(fixed_effects.columns):
            renames = {}
            for index, column in enumerate(fixed_effects):
                name = f"__formula_fe_{index}"
                while name in data or name in independent or name in dependent:
                    name += "_"
                renames[column] = name
            fixed_effects = fixed_effects.rename(columns=renames)
            fe_spec = FittedRole(
                fe_spec.model, tuple(fixed_effects.columns), renames=renames
            )
        self._model_spec: dict[str, FittedStage | FittedRole] = {
            "second_stage": FittedStage(lhs, rhs)
        }
        frames = [dependent, independent]
        self._dependent_column_names = list(dependent.columns)
        self._independent_column_names = list(independent.columns)
        self._endogenous_column_names = self._instruments_column_names = None
        self._weights_column_names = self._offset_column_names = None
        self._fixed_effects_column_names = list(fixed_effects.columns) or None
        self._fixed_effect_specification_by_column = {}
        self._slope_column_by_term = {}
        self._drop_intercept = drop_intercept
        if len(fixed_effects.columns):
            # The normalized native formula preserves quoted lookup names.
            fe_source = split_top_level(main.formula, "|")[-1]
            terms = split_top_level(fe_source, "+")
            for column, term in zip(fixed_effects.columns, terms, strict=True):
                names = [
                    part.strip().strip("`").replace("``", "`")
                    for part in split_top_level(term, ":")
                ]
                description = FixedEffectSpecification(
                    levels=Term(
                        [
                            Factor(name, eval_method=Factor.EvalMethod.LOOKUP)
                            for name in names
                        ]
                    ),
                    intercept=True,
                )
                self._fixed_effect_specification_by_column[column] = description
                codes = fixed_effects[column]
                first_level = codes.notna() & ~codes.duplicated()
                levels = data.loc[first_level, names].copy()
                levels.index = codes.loc[first_level].astype("int64")
                fe_spec.transform_state[column] = {
                    FIXED_EFFECT_ENCODING_STATE_KEY: FixedEffectLevels(
                        description.name, levels
                    )
                }
            self._model_spec["fe"] = fe_spec
            frames.append(fixed_effects)
        self._fixed_effect_specifications = tuple(
            self._fixed_effect_specification_by_column.values()
        )
        first = None
        if specification.is_instrumental_variable:
            first = formula.parse(specification.as_first_stage().second_stage).fit(
                native_data
            )[0]
            endog_spec, endogenous = _role(first, formula.Role.Dependent, data)
            instrument_spec, instruments = _role(first, formula.Role.Independent, data)
            self._model_spec["first_stage"] = FittedStage(endog_spec, instrument_spec)
            self._endogenous_column_names = list(endogenous.columns)
            self._instruments_column_names = list(instruments.columns)
            exogenous_columns = set(independent.columns) - set(endogenous.columns)
            if not set(instruments.columns) - exogenous_columns:
                raise IVIdentificationError(
                    "The instruments add no explanatory power: all instruments are covariates."
                )
            frames.extend([endogenous, instruments])
        self.specification = _normalized_specification(main, first)
        if weights is not None:
            weight_frame = _get_weights(data, weights).to_frame()
            self._weights_column_names = [weights]
            frames.append(weight_frame)
        joined = pd.concat(frames, axis=1)
        missing = joined.isna().any(axis=1)
        self._data = joined.loc[~missing, ~joined.columns.duplicated()]
        self._dropped_row_index = frozenset(data.index[missing].tolist())
        self._dropped_by_stage = DroppedRowCounts(missing=int(missing.sum()))
        self._process(drop_singletons=drop_singletons)

    def column_keys(self, role):
        """Key columns by label so different specifications share demeaned columns.

        Models sharing a demeaning cache transform the same input rows, and a
        native term's state depends only on its own inputs, so a label fixes
        the column's values. An input column and an expression can share a
        label, e.g. a data column named ``log(x)``; the key keeps them apart.
        """
        frame = getattr(self, role)
        if frame is None:
            return ()
        stage, side = {
            "dependent": ("second_stage", "lhs"),
            "independent": ("second_stage", "rhs"),
            "endogenous": ("first_stage", "lhs"),
            "instruments": ("first_stage", "rhs"),
        }[role]
        bare_columns = getattr(self._model_spec[stage], side).bare_columns
        return tuple(
            ("column" if column in bare_columns else "expression", column)
            for column in frame
        )
