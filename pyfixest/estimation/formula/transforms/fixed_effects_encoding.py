from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from formulaic.parser.types import Factor, Term
from formulaic.transforms import TRANSFORMS
from formulaic.utils.layered_mapping import LayeredMapping
from formulaic.utils.stateful_transforms import stateful_eval, stateful_transform
from formulaic.utils.variables import Variable, get_required_variables

from pyfixest.errors import FixedEffectEvaluationError
from pyfixest.estimation.formula.utils import term_key
from pyfixest.utils.dev_utils import _find_stack_level

FIXED_EFFECT_ENCODING: Final[str] = "__fixed_effect_encoding__"
_FIXED_EFFECT_TRANSFORM: Final[str] = "__fixed_effect__"


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
        """Learn the level mappings and return the training codes."""
        # Sorted factorization preserves declared categorical order and safely
        # orders mixed numbers/strings, returning codes and observed levels together.
        factorized = [pd.factorize(column, sort=True) for column in values]
        per_factor = [codes for codes, _ in factorized]
        levels = tuple(pd.Index(levels) for _, levels in factorized)
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


def wrap_fixed_effect(term: Term) -> Term:
    """Carry original factors in metadata; source contains only a stable ID.

    Original lookup names remain metadata and never become executable Python
    source.
    """
    identifier = int(
        hashlib.sha256(json.dumps(term_key(term=term)).encode()).hexdigest(), 16
    )
    return Term(
        [
            Factor(
                f"{_FIXED_EFFECT_TRANSFORM}({identifier})",
                eval_method=Factor.EvalMethod.PYTHON,
                metadata={"term": term, "identifier": identifier},
            )
        ]
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class FixedEffectContext:
    """Per-materialization bindings for the FE encoder and dependency hook.

    `make()` prepares the bindings, and `register()` exposes them to Formulaic.
    Formulaic invokes the dependency hook and encoder during materialization.
    """

    context: Mapping[str, Any]
    terms_by_id: Mapping[int, Term]
    dependency_context: Mapping[str, Any]

    @classmethod
    def make(
        cls, *, terms: Iterable[Term], data: pd.DataFrame, context: Mapping[str, Any]
    ) -> FixedEffectContext:
        """Prepare dependency bindings without evaluating the FE factors."""
        terms_by_id = {}
        for term in terms:
            (factor,) = term.factors
            terms_by_id[factor.metadata["identifier"]] = factor.metadata["term"]
        if not terms_by_id:
            return cls(context=context, terms_by_id={}, dependency_context={})

        evaluation_context = LayeredMapping(
            LayeredMapping(data, name="data"),
            LayeredMapping(context, name="context"),
            LayeredMapping(TRANSFORMS, name="transforms"),
        )
        dependency_context = evaluation_context.with_layers(
            {
                name: wraps(transform)(
                    partial(
                        _evaluate_transform_with_context,
                        transform=transform,
                        context=evaluation_context,
                    )
                )
                for name, transform in (TRANSFORMS | dict(context)).items()
                if getattr(transform, "__is_stateful_transform__", False)
            }
        )
        return cls(
            context=context,
            terms_by_id=terms_by_id,
            dependency_context=dependency_context,
        )

    def register(self) -> dict[str, Any]:
        """Return a Formulaic context with the FE encoder and hook registered."""
        if not self.terms_by_id:
            return dict(self.context)
        return dict(self.context) | {
            _FIXED_EFFECT_TRANSFORM: stateful_transform(
                encode_fixed_effects, get_required_variables=self.get_required_variables
            )
        }

    def get_required_variables(self, identifier: int) -> set[Variable]:
        """Report original FE dependencies when Formulaic invokes the hook.

        Formulaic passes the wrapper's integer argument to this hook, without
        factor metadata. The local ID map supplies the original parsed term;
        Formulaic assigns sources to the variables returned here.
        """
        original = self.terms_by_id[identifier]
        variables = set()
        for factor in original.factors:
            if factor.eval_method is Factor.EvalMethod.LOOKUP:
                variables.add(Variable(factor.expr, roles={Variable.Role.VALUE}))
            elif factor.eval_method is Factor.EvalMethod.PYTHON:
                variables.update(
                    get_required_variables(factor.expr, self.dependency_context)
                )
        return variables


def _evaluate_transform_with_context(
    *args, transform: Callable[..., Any], context: Mapping[str, Any], **kwargs
):
    """Evaluate a nested transform with context during dependency discovery."""
    # Dependency hooks evaluate nested arguments without stateful_eval.
    # Supply their context here, using temporary transform state.
    kwargs.setdefault("_context", context)
    return transform(*args, **kwargs)


def encode_fixed_effects(
    identifier, _state=None, _metadata=None, _spec=None, _context: Any = None
):
    """Evaluate parsed factors and code their observed level combinations.

    The identifier distinguishes Formulaic transform-state keys and is resolved
    by the dependency hook. Evaluation reads the parsed term from `_metadata`.

    LOOKUP names never enter Python source. PYTHON factors retain their own
    source and persistent nested transform state, including prediction state.
    """
    state = cast(MutableMapping[str, Any], _state)
    metadata = cast(Mapping[str, Any], _metadata)
    term: Term = metadata["term"]
    try:
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
                np.column_stack([column.notna().to_numpy() for column in values]),
                axis=1,
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
    except FixedEffectEvaluationError:
        raise
    except Exception as exc:
        raise FixedEffectEvaluationError(
            f"Unable to encode fixed effect `{term}`. [{type(exc).__name__}: {exc}]"
        ) from exc
    return codes


def evaluate_fixed_effect_factors(
    *, term: Term, context: Any, factor_states: MutableMapping[str, Any], spec: Any
) -> tuple[pd.Series, ...]:
    """Evaluate FE components positionally, using their fitted transform state."""
    columns = []
    index = next(iter(context.data.values())).index
    for factor in term.factors:
        if factor.eval_method not in (
            Factor.EvalMethod.LOOKUP,
            Factor.EvalMethod.PYTHON,
        ):
            raise FixedEffectEvaluationError(
                f"Fixed effect `{term}`: factor `{factor}` must be a lookup "
                "or Python expression."
            )
        try:
            if factor.eval_method is Factor.EvalMethod.LOOKUP:
                values = context[factor.expr]
            else:
                values = stateful_eval(
                    factor.expr,
                    context,
                    {factor.expr: factor.metadata},
                    factor_states.setdefault(factor.expr, {}),
                    spec,
                )
        except Exception as exc:
            raise FixedEffectEvaluationError(
                f"Unable to evaluate fixed effect `{term}`: factor `{factor}` failed. "
                f"[{type(exc).__name__}: {exc}]"
            ) from exc
        values = getattr(values, "__wrapped__", values)
        if not isinstance(values, pd.Series):
            values = pd.Series(values, index=index, name=factor.expr)
        elif not values.index.equals(index):
            values = values.reindex(index)
        columns.append(values.rename(str(factor)))
    return tuple(columns)
