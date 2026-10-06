from __future__ import annotations

import hashlib
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from formulaic.parser.types import Factor, Term
from formulaic.transforms import TRANSFORMS
from formulaic.transforms.contrasts import C, TreatmentContrasts
from formulaic.utils.layered_mapping import LayeredMapping
from formulaic.utils.stateful_transforms import stateful_eval, stateful_transform
from formulaic.utils.variables import Variable, get_required_variables

from pyfixest.utils.dev_utils import _find_stack_level

FIXED_EFFECT_ENCODING: Final[str] = "__fixed_effect_encoding__"


@dataclass(kw_only=True)
class _FixedEffectContrasts(TreatmentContrasts):
    """Native treatment coding with the parsed FE label in dummy names."""

    variable: str

    @TreatmentContrasts.override
    def get_factor_format(
        self, levels: Sequence[Hashable], reduced_rank: bool = True
    ) -> str:
        # Formulaic calls str.format() on this template; labels are literal text.
        label = self.variable.replace("{", "{{").replace("}", "}}")
        return label + ("[T.{field}]" if reduced_rank else "[{field}]")


@dataclass(frozen=True, slots=True)
class FixedEffectEncoding:
    """Observed factor levels and combinations in fitted FE-code order."""

    term: Term
    levels: tuple[pd.Index, ...]
    combinations: pd.RangeIndex | pd.MultiIndex

    @property
    def variable(self) -> str:
        """Original factor labels used in fixed-effect output and diagnostics."""
        return ":".join(str(factor) for factor in self.term.factors)

    @classmethod
    def fit(
        cls, *, term: Term, values: tuple[pd.Series, ...]
    ) -> tuple[FixedEffectEncoding, pd.Series]:
        """Learn indexes and return training codes without repeating lookups."""
        levels = tuple(_sorted_levels(column=column) for column in values)
        per_factor = [
            level.get_indexer(column)
            for level, column in zip(levels, values, strict=True)
        ]
        if len(values) == 1:
            # The sorted level index already defines the complete code order.
            combinations = pd.RangeIndex(len(levels[0]))
            codes = per_factor[0].astype(np.float64)
        else:
            valid = np.all(np.column_stack(per_factor) >= 0, axis=1)
            observed = pd.MultiIndex.from_arrays(per_factor)
            combinations = observed[valid].unique().sort_values()
            codes = combinations.get_indexer(observed).astype(np.float64)
        codes[codes < 0] = np.nan
        return (
            cls(term=term, levels=levels, combinations=combinations),
            pd.Series(codes, index=values[0].index),
        )

    def encode(self, *, values: tuple[pd.Series, ...]) -> pd.Series:
        """Use the fitted indexes for both training and prediction matching."""
        per_factor = [
            level.get_indexer(column)
            for level, column in zip(self.levels, values, strict=True)
        ]
        if len(values) == 1:
            codes = per_factor[0].astype(np.float64)
        else:
            codes = self.combinations.get_indexer(
                pd.MultiIndex.from_arrays(per_factor)
            ).astype(np.float64)
        codes[codes < 0] = np.nan
        return pd.Series(codes, index=values[0].index)

    def decoded_values(self, *, codes: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return original level values for the requested fitted FE codes."""
        if len(self.levels) == 1:
            return (self.levels[0].to_numpy()[codes],)
        combinations = self.combinations[codes]
        return tuple(
            level.to_numpy()[
                combinations.get_level_values(position).to_numpy(dtype=np.int64)
            ]
            for position, level in enumerate(self.levels)
        )


def _sorted_levels(*, column: pd.Series) -> pd.Index:
    """Retain groupby ordering, including the declared categorical order."""
    if isinstance(column.dtype, pd.CategoricalDtype):
        categories = column.cat.categories
        return categories[categories.isin(column.dropna().unique())]
    # pandas factorization uses the same safe sorting as groupby for mixtures
    # of numbers and strings, without constructing any unobserved combinations.
    _, levels = pd.factorize(column, sort=True)
    return pd.Index(levels)


def wrap_fixed_effect(term: Term, *, dummies: bool = False) -> Term:
    """Carry original factors in metadata; source contains only a stable ID.

    Numeric and dummy encodings use the same state key in separate
    materializations so fixed-effect recovery reuses the fitted group codes.
    """
    identity = repr(
        [(factor.expr, factor.eval_method.value) for factor in term.factors]
    )
    identifier = int(hashlib.sha256(identity.encode()).hexdigest(), 16)
    return Term(
        [
            Factor(
                f"__fixed_effect__({identifier})",
                eval_method=Factor.EvalMethod.PYTHON,
                metadata={"term": term, "dummies": dummies},
            )
        ]
    )


def fixed_effect_context(
    *, terms: Iterable[Term], data: pd.DataFrame, context: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind dependency reporting to this materialization's original FE terms.

    Formulaic calls dependency hooks before injecting transform metadata.
    Keep the ID-to-term map in this call's context, rather than a global
    registry. Formulaic assigns data/context sources to the returned variables.
    """
    original_terms = {}
    for term in terms:
        (factor,) = term.factors
        original_terms[factor.expr] = factor.metadata["term"]
    if not original_terms:
        return dict(context)

    evaluation_context = LayeredMapping(
        LayeredMapping(data, name="data"),
        LayeredMapping(context, name="context"),
        LayeredMapping(TRANSFORMS, name="transforms"),
    )

    def with_context(transform):
        @wraps(transform)
        def evaluate(*args, **kwargs):
            # Dependency hooks evaluate nested arguments without stateful_eval.
            # Supply their context here, using temporary transform state.
            kwargs.setdefault("_context", evaluation_context)
            return transform(*args, **kwargs)

        return evaluate

    dependency_context = evaluation_context.with_layers(
        {
            name: with_context(transform)
            for name, transform in (TRANSFORMS | dict(context)).items()
            if getattr(transform, "__is_stateful_transform__", False)
        }
    )

    def required_variables(identifier):
        original = original_terms[f"__fixed_effect__({identifier})"]
        variables = set()
        for factor in original.factors:
            if factor.eval_method is Factor.EvalMethod.LOOKUP:
                variables.add(Variable(factor.expr, roles={Variable.Role.VALUE}))
            elif factor.eval_method is Factor.EvalMethod.PYTHON:
                variables.update(
                    get_required_variables(factor.expr, dependency_context)
                )
        return variables

    @wraps(encode_fixed_effects)
    def encode(*args, **kwargs):
        return encode_fixed_effects(*args, **kwargs)

    cast(Any, encode).get_required_variables = required_variables
    return dict(context) | {"__fixed_effect__": encode}


@stateful_transform
def encode_fixed_effects(
    identifier, _state=None, _metadata=None, _spec=None, _context: Any = None
):
    """Evaluate parsed factors and code their observed level combinations.

    LOOKUP names never enter Python source. PYTHON factors retain their own
    source and persistent nested transform state, including prediction state.
    """
    state = cast(MutableMapping[str, Any], _state)
    metadata = cast(Mapping[str, Any], _metadata)
    term: Term = metadata["term"]
    factor_states = state.setdefault("factors", {})
    values = evaluate_fixed_effect_factors(
        term=term, context=_context, factor_states=factor_states, spec=_spec
    )
    if FIXED_EFFECT_ENCODING not in state:
        encoding, codes = FixedEffectEncoding.fit(term=term, values=values)
        state[FIXED_EFFECT_ENCODING] = encoding
    else:
        encoding = cast(FixedEffectEncoding, state[FIXED_EFFECT_ENCODING])
        codes = encoding.encode(values=values)
        unseen = codes.isna().to_numpy() & np.all(
            np.column_stack([column.notna().to_numpy() for column in values]), axis=1
        )
        if unseen.any():
            missing = pd.concat(values, axis=1).loc[unseen].drop_duplicates()
            warnings.warn(
                f"{missing.shape[0]} unseen level(s) for fixed effect "
                f"`{encoding.variable}`: {missing.iloc[:20]}\n"
                "Predictions for affected observations will be NaN",
                UserWarning,
                stacklevel=_find_stack_level(),
            )
    if metadata["dummies"]:
        return C(codes, contrasts=_FixedEffectContrasts(variable=encoding.variable))
    return codes


def evaluate_fixed_effect_factors(
    *, term: Term, context: Any, factor_states: MutableMapping[str, Any], spec: Any
) -> tuple[pd.Series, ...]:
    """Evaluate FE components positionally, using their fitted transform state."""
    columns = []
    index = next(iter(context.data.values())).index
    for factor in term.factors:
        if factor.eval_method is Factor.EvalMethod.LOOKUP:
            values = context[factor.expr]
        elif factor.eval_method is Factor.EvalMethod.PYTHON:
            values = stateful_eval(
                factor.expr,
                context,
                {factor.expr: factor.metadata},
                factor_states.setdefault(factor.expr, {}),
                spec,
            )
        else:
            raise ValueError(
                f"Fixed effect `{factor}` must be a lookup or Python expression."
            )
        values = getattr(values, "__wrapped__", values)
        if not isinstance(values, pd.Series):
            values = pd.Series(values, index=index, name=factor.expr)
        elif not values.index.equals(index):
            values = values.reindex(index)
        columns.append(values.rename(str(factor)))
    return tuple(columns)
