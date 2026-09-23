from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import MapDemeaner
from pyfixest.estimation.api.utils import _ALL_SAMPLE, _AllSampleSentinel
from pyfixest.estimation.config import EstimationConfig
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.literals import WeightsTypeOptions
from pyfixest.estimation.internals.model_state import (
    EstimationOptions,
    GlmEstimationOptions,
    QuantregEstimationOptions,
)
from pyfixest.estimation.internals.vcov_utils import _get_vcov_type
from pyfixest.estimation.models.fegaussian_ import Fegaussian
from pyfixest.estimation.models.feiv_ import Feiv
from pyfixest.estimation.models.felogit_ import Felogit
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.estimation.models.feprobit_ import Feprobit
from pyfixest.estimation.protocols import FittedModel, ModelFactory
from pyfixest.estimation.quantreg.quantreg_ import Quantreg
from pyfixest.estimation.quantreg.QuantregMulti import QuantregMulti
from pyfixest.utils.utils import Ssc


@dataclass(frozen=True)
class ModelEntry:
    """One row in the model registry.

    `model_cls` is the class to instantiate and `options_cls` the
    `EstimationOptions` flavour its constructor takes, which decides
    which `EstimationConfig` fields the planner reads. The remaining
    flags record the per-class wiring that the options class alone does
    not express.
    """

    model_cls: ModelFactory
    options_cls: type[EstimationOptions] = EstimationOptions
    # Quantile regression does not absorb fixed effects, so it neither
    # demeans nor shares the runner's preconditioner cache.
    accepts_preconditioner: bool = True
    # `QuantregMulti` fans one call out over several quantiles and needs
    # the quantile list and the process algorithm on top of the options.
    fits_quantile_process: bool = False


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "feols": ModelEntry(Feols),
    "fepois": ModelEntry(Fepois, options_cls=GlmEstimationOptions),
    "feglm-logit": ModelEntry(Felogit, options_cls=GlmEstimationOptions),
    "feglm-probit": ModelEntry(Feprobit, options_cls=GlmEstimationOptions),
    "feglm-gaussian": ModelEntry(Fegaussian, options_cls=GlmEstimationOptions),
    "quantreg": ModelEntry(
        Quantreg,
        options_cls=QuantregEstimationOptions,
        accepts_preconditioner=False,
    ),
    "quantreg_multi": ModelEntry(
        QuantregMulti,
        options_cls=QuantregEstimationOptions,
        accepts_preconditioner=False,
        fits_quantile_process=True,
    ),
}


def _resolve_model_class(method: str, is_iv: bool) -> ModelFactory:
    """Pick the model class to instantiate for this method.

    The only special case is `feols` with an IV formula, which
    dispatches to `Feiv`. Everything else just looks up the
    method in the registry — IV isn't supported there, so we
    ignore `is_iv`.
    """
    if method == "feols" and is_iv:
        return Feiv
    return MODEL_REGISTRY[method].model_cls


@dataclass(frozen=True)
class ParsedFormula:
    """Stores the results from formula parsing = everything the runner needs to know.

    `formula_dict` keys by the fixed-effects formula string (or
    `None` when no FE) and maps to the list of `FixestFormula`
    objects for that block.
    `is_iv` is true when any formula has a
    first stage.
    `is_multiple_estimation` if multiple estimation syntax is used.
    """

    formula_dict: dict[str | None, list[FixestFormula]]
    is_iv: bool
    is_multiple_estimation: bool


def parse_formula(config: EstimationConfig) -> ParsedFormula:
    """Parse the config's `fml` string into a `ParsedFormula`.

    Pure: same `(fml, split, fsplit, quantile)` always produce the
    same parse. `is_multiple_estimation` reflects formula
    expansion *and* sample-split / multi-quantile fan-out.
    """
    run_split = config.split is not None or config.fsplit is not None
    formula_dictionary = FixestFormula.parse_to_dict(config.fml)
    is_multiple_estimation = (
        sum(len(v) for v in formula_dictionary.values()) > 1
        or run_split
        or (isinstance(config.quantile, list) and len(config.quantile) > 1)
    )
    is_iv = any(
        f.is_instrumental_variable
        for formulas in formula_dictionary.values()
        for f in formulas
    )
    return ParsedFormula(
        formula_dict=formula_dictionary,
        is_iv=is_iv,
        is_multiple_estimation=is_multiple_estimation,
    )


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
    model_cls: ModelFactory
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
    entry = MODEL_REGISTRY[config.method]

    options = _build_options(
        config=config, entry=entry, captured_context=captured_context
    )

    specs: list[ModelSpec] = []
    for sample_split_value in splits:
        for fixef_key in formula_dict:
            for formula in formula_dict[fixef_key]:
                model_kwargs = _build_model_kwargs(
                    config=config,
                    entry=entry,
                    formula=formula,
                    data=data,
                    options=options,
                    sample_split_value=sample_split_value,
                    splitvar=splitvar,
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


def _build_options(
    *,
    config: EstimationConfig,
    entry: ModelEntry,
    captured_context: Mapping[str, Any],
) -> EstimationOptions:
    """Turn the estimation request into the options value the model is built with.

    This is the single boundary between `EstimationConfig`, which records
    what the user asked for, and the frozen `options` a fitted model
    publishes. `entry.options_cls` decides which estimator-specific fields
    are filled; the shared fields are the same for every model class.
    """
    shared: dict[str, Any] = {
        "ssc": config.ssc if config.ssc is not None else Ssc(),
        "drop_singletons": _drop_singletons(config.fixef_rm),
        "drop_intercept": config.drop_intercept,
        "weights": config.weights,
        # validated at the API boundary (estimation/api/utils.py)
        "weights_type": cast(WeightsTypeOptions, config.weights_type),
        # only `fepois` reads an offset; the other APIs never set one
        "offset": config.offset,
        "collin_tol": config.collin_tol,
        "solver": config.solver,
        "demeaner": config.demeaner if config.demeaner is not None else MapDemeaner(),
        "store_data": config.store_data,
        "copy_data": config.copy_data,
        "lean": config.lean,
        "context": captured_context,
    }

    options_cls = entry.options_cls
    if issubclass(options_cls, GlmEstimationOptions):
        return GlmEstimationOptions(
            **shared,
            maxiter=config.iwls_maxiter,
            tol=config.iwls_tol,
            separation_check=config.separation_check,
            accelerate=config.accelerate,
        )
    if issubclass(options_cls, QuantregEstimationOptions):
        quantile = config.quantile
        return QuantregEstimationOptions(
            **shared,
            # `quantile` is validated at the API boundary
            # (estimation/api/quantreg.py); the quantile process carries the
            # first requested quantile and gives each child fit its own via
            # `dataclasses.replace`
            quantile=cast(
                float, quantile[0] if isinstance(quantile, list) else quantile
            ),
            method=config.quantreg_method,
            quantile_tol=config.quantile_tol,
            quantile_maxiter=config.quantile_maxiter,
            seed=config.seed,
        )
    return EstimationOptions(**shared)


def _build_model_kwargs(
    *,
    config: EstimationConfig,
    entry: ModelEntry,
    formula: FixestFormula,
    data: pd.DataFrame,
    options: EstimationOptions,
    sample_split_value: Any,
    splitvar: str | None,
) -> dict[str, Any]:
    """Compose the static constructor kwargs for one model.

    The cache dicts (`lookup_demeaned_data`, `lookup_preconditioner`)
    are intentionally *not* set here — they're injected per
    cache-block by the runner.
    """
    kwargs: dict[str, Any] = {
        "FixestFormula": formula,
        "data": data,
        "options": options,
        "sample_split_value": sample_split_value,
        "sample_split_var": splitvar,
    }

    if entry.fits_quantile_process:
        # the fan-out itself is not an option of any single fit
        kwargs["quantile"] = config.quantile
        kwargs["multi_method"] = config.quantreg_multi_method

    return kwargs


def fit_one(
    spec: ModelSpec,
    *,
    lookup_demeaned_data: dict[frozenset[int], DemeanedData],
    lookup_preconditioner: dict[frozenset[int], Preconditioner],
    vcov: str | dict[str, str] | None,
    vcov_kwargs: dict[str, str | int] | None,
) -> FittedModel:
    """Run the full fit pipeline for one model spec.

    Constructs the model class, runs prepare → fit → vcov → inference,
    and clears large attributes. The two per-cache-block dicts are
    injected here so they're shared across every spec in the block.

    Returns the fitted model.
    """
    model_kwargs = dict(spec.model_kwargs)
    model_kwargs["lookup_demeaned_data"] = lookup_demeaned_data
    if MODEL_REGISTRY[spec.method].accepts_preconditioner:
        model_kwargs["lookup_preconditioner"] = lookup_preconditioner

    FIT: FittedModel = spec.model_cls(**model_kwargs)

    FIT.prepare_model_matrix()
    FIT._validate_response()
    FIT.get_fit()
    # if X is empty: no inference (empty X only as shorthand for demeaning)
    if not FIT.X_is_empty:
        vcov_type = _get_vcov_type(vcov)
        # vcov() reads the model's retained estimation data when data is None
        FIT.vcov(vcov=vcov_type, vcov_kwargs=vcov_kwargs)

        FIT.get_inference()
        FIT._finalize_fit()
    # delete large attributes
    FIT._clear_attributes()

    return FIT
