from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pyfixest.estimation.api.utils import _ALL_SAMPLE, _AllSampleSentinel
from pyfixest.estimation.config import EstimationConfig
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.models.fegaussian_ import Fegaussian
from pyfixest.estimation.models.feiv_ import Feiv
from pyfixest.estimation.models.felogit_ import Felogit
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.estimation.models.feprobit_ import Feprobit
from pyfixest.estimation.quantreg.quantreg_ import Quantreg
from pyfixest.estimation.quantreg.QuantregMulti import QuantregMulti


@dataclass(frozen=True)
class ModelEntry:
    """One row in the model registry.

    `model_cls` is the class to instantiate. `needs` lists the
    extra option groups that class's constructor accepts on top of
    the base kwargs - for example `"iwls"` for GLMs, or
    `"quantreg"` for quantile regression. The planner reads this
    to decide which config fields to thread through.
    """

    model_cls: type
    needs: frozenset[str] = field(default_factory=frozenset)


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "feols": ModelEntry(Feols, frozenset({"demeaner"})),
    "fepois": ModelEntry(
        Fepois,
        frozenset({"demeaner", "iwls", "separation_check", "offset"}),
    ),
    "feglm-logit": ModelEntry(
        Felogit,
        frozenset({"demeaner", "iwls", "separation_check", "accelerate"}),
    ),
    "feglm-probit": ModelEntry(
        Feprobit,
        frozenset({"demeaner", "iwls", "separation_check", "accelerate"}),
    ),
    "feglm-gaussian": ModelEntry(
        Fegaussian,
        frozenset({"demeaner", "iwls", "separation_check", "accelerate"}),
    ),
    "quantreg": ModelEntry(Quantreg, frozenset({"quantreg"})),
    "quantreg_multi": ModelEntry(
        QuantregMulti, frozenset({"quantreg", "quantreg_multi"})
    ),
}


def _resolve_model_class(method: str, is_iv: bool) -> type:
    """Pick the model class to instantiate for this method.

    The only special case is `feols` with an IV formula, which
    dispatches to `Feiv`. Everything else just looks up the
    method in the registry — IV isn't supported there, so we
    ignore `is_iv`.
    """
    if method == "feols" and is_iv:
        return Feiv
    return MODEL_REGISTRY[method].model_cls


def _drop_singletons(fixef_rm: str) -> bool:
    return fixef_rm == "singleton"


@dataclass(frozen=True)
class ModelSpec:
    """A single model to fit, with everything the runner needs to do it.

    `model_kwargs` holds every constructor argument that doesn't
    change over the course of the run. The cache dicts
    (`lookup_demeaned_data` and, for non-quantreg methods,
    `lookup_preconditioner`) are deliberately *not* in here —
    the runner injects them at fit time so that specs sharing the
    same `cache_key` can share the cache.
    """

    method: str
    model_cls: type
    formula: FixestFormula
    fixef_key: str | None
    sample_split_value: Any
    model_kwargs: dict[str, Any]

    @property
    def cache_key(self) -> tuple[Any, str | None]:
        """Specs with the same key can share demean / preconditioner caches."""
        return (self.sample_split_value, self.fixef_key)


def build_all_splits(
    *,
    run_full: bool,
    run_split: bool,
    splitvar: str | None,
    data: pd.DataFrame,
) -> list[Any]:
    """List the sample-split values in the order the runner will visit them.

    The full sample comes first if requested, followed by the
    sorted unique values of the split column. The order matches
    what `FixestMulti` did before the refactor, which keeps
    cache blocks contiguous downstream.
    """
    all_splits: list[str | int | float | _AllSampleSentinel] = []
    if run_full:
        all_splits.append(_ALL_SAMPLE)
    if run_split:
        assert splitvar is not None
        all_splits.extend(
            data[splitvar].dropna().drop_duplicates().sort_values().tolist()
        )
    return all_splits


def expand_specs(
    *,
    config: EstimationConfig,
    formula_dict: Mapping[str | None, list[FixestFormula]],
    data: pd.DataFrame,
    splits: list[Any],
    is_iv: bool,
    splitvar: str | None,
    captured_context: Mapping[str, Any],
) -> list[ModelSpec]:
    """Build one `ModelSpec` per model the user's call expands into.

    We iterate splits, then fixef keys, then formulas. The order is
    by design: specs that share a cache key
    end up next to each other in the list, so the runner can
    reuse its demean and preconditioner caches across them and
    drop them as soon as the cache key changes.
    """
    model_cls = _resolve_model_class(config.method, is_iv)
    needs = MODEL_REGISTRY[config.method].needs

    ssc_dict = dict(config.ssc_dict) if config.ssc_dict else {}
    drop_singletons = _drop_singletons(config.fixef_rm)

    specs: list[ModelSpec] = []
    for sample_split_value in splits:
        for fixef_key in formula_dict:
            for formula in formula_dict[fixef_key]:
                model_kwargs = _build_model_kwargs(
                    config=config,
                    needs=needs,
                    formula=formula,
                    data=data,
                    ssc_dict=ssc_dict,
                    drop_singletons=drop_singletons,
                    sample_split_value=sample_split_value,
                    splitvar=splitvar,
                    captured_context=captured_context,
                )
                specs.append(
                    ModelSpec(
                        method=config.method,
                        model_cls=model_cls,
                        formula=formula,
                        fixef_key=fixef_key,
                        sample_split_value=sample_split_value,
                        model_kwargs=model_kwargs,
                    )
                )
    return specs


def _build_model_kwargs(
    *,
    config: EstimationConfig,
    needs: frozenset[str],
    formula: FixestFormula,
    data: pd.DataFrame,
    ssc_dict: dict[str, Any],
    drop_singletons: bool,
    sample_split_value: Any,
    splitvar: str | None,
    captured_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose the static constructor kwargs for one model.

    The cache dicts (`lookup_demeaned_data`, `lookup_preconditioner`)
    are intentionally *not* set here — they're injected per
    cache-block by the runner.
    """
    kwargs: dict[str, Any] = {
        "FixestFormula": formula,
        "data": data,
        "ssc_dict": ssc_dict,
        "drop_singletons": drop_singletons,
        "drop_intercept": config.drop_intercept,
        "weights": config.weights,
        "weights_type": config.weights_type,
        "solver": config.solver,
        "collin_tol": config.collin_tol,
        "store_data": config.store_data,
        "copy_data": config.copy_data,
        "lean": config.lean,
        "context": captured_context,
        "sample_split_value": sample_split_value,
        "sample_split_var": splitvar,
    }

    if "demeaner" in needs:
        kwargs["demeaner"] = config.demeaner

    if "iwls" in needs:
        kwargs["tol"] = config.iwls_tol
        kwargs["maxiter"] = config.iwls_maxiter

    if "separation_check" in needs:
        kwargs["separation_check"] = config.separation_check

    if "offset" in needs:
        kwargs["offset"] = config.offset

    if "accelerate" in needs:
        kwargs["accelerate"] = config.accelerate

    if "quantreg" in needs:
        kwargs["quantile"] = config.quantile
        kwargs["method"] = config.quantreg_method
        kwargs["quantile_tol"] = config.quantile_tol
        kwargs["quantile_maxiter"] = config.quantile_maxiter
        kwargs["seed"] = config.seed

    if "quantreg_multi" in needs:
        kwargs["multi_method"] = config.quantreg_multi_method

    return kwargs
