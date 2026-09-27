from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.estimation.config import EstimationConfig, QuantileProcess
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.literals import EstimationMethod
from pyfixest.estimation.internals.model_state import (
    EstimationOptions,
    GlmEstimationOptions,
    QuantregEstimationOptions,
    SampleSplit,
    VcovSpec,
)
from pyfixest.estimation.models.fegaussian_ import Fegaussian
from pyfixest.estimation.models.feiv_ import Feiv
from pyfixest.estimation.models.felogit_ import Felogit
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.estimation.models.feprobit_ import Feprobit
from pyfixest.estimation.protocols import FittedModel, ModelFactory
from pyfixest.estimation.quantreg.quantreg_ import Quantreg
from pyfixest.estimation.quantreg.QuantregMulti import QuantregMulti


@dataclass(frozen=True)
class ModelEntry:
    """How to build the model for one estimation method.

    Attributes
    ----------
    model_cls
        The model class to create, e.g. `Feols`.
    options_cls
        The type of options the model expects, e.g. `GlmEstimationOptions`
        for GLMs.
    iv_model_cls
        The model class to use instead when the formula has an IV part.
    quantile_process_model_cls
        The model class to use instead when the call fits a quantile process.
    accepts_preconditioner
        Whether the model can reuse the runner's shared preconditioner cache.
    """

    model_cls: ModelFactory
    options_cls: type[EstimationOptions] = EstimationOptions
    iv_model_cls: ModelFactory | None = None
    quantile_process_model_cls: ModelFactory | None = None
    # Quantile regression does not absorb fixed effects, so it neither
    # demeans nor shares the runner's preconditioner cache.
    accepts_preconditioner: bool = True


MODEL_REGISTRY: dict[EstimationMethod, ModelEntry] = {
    "feols": ModelEntry(Feols, iv_model_cls=Feiv),
    "fepois": ModelEntry(Fepois, options_cls=GlmEstimationOptions),
    "feglm-logit": ModelEntry(Felogit, options_cls=GlmEstimationOptions),
    "feglm-probit": ModelEntry(Feprobit, options_cls=GlmEstimationOptions),
    "feglm-gaussian": ModelEntry(Fegaussian, options_cls=GlmEstimationOptions),
    "quantreg": ModelEntry(
        Quantreg,
        options_cls=QuantregEstimationOptions,
        quantile_process_model_cls=QuantregMulti,
        accepts_preconditioner=False,
    ),
}


def _resolve_model_class(
    method: EstimationMethod, *, is_iv: bool, fits_quantile_process: bool
) -> ModelFactory:
    """Pick the model class to instantiate for this method.

    IV formulas dispatch to the entry's `iv_model_cls`. Methods without one
    reject IV formulas in their estimation function, so `is_iv` is ignored.
    A quantile process dispatches to the entry's `quantile_process_model_cls`;
    a method without one cannot fit it.
    """
    entry = MODEL_REGISTRY[method]
    if fits_quantile_process:
        if entry.quantile_process_model_cls is None:
            raise TypeError(f"{method!r} models cannot fit a quantile process.")
        return entry.quantile_process_model_cls
    if is_iv and entry.iv_model_cls is not None:
        return entry.iv_model_cls
    return entry.model_cls


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

    Pure: same `(fml, split, fsplit, quantile_process)` always produce
    the same parse. `is_multiple_estimation` reflects formula
    expansion *and* sample-split / multi-quantile fan-out.
    """
    run_split = config.split is not None or config.fsplit is not None
    formula_dictionary = FixestFormula.parse_to_dict(config.fml)
    process = config.quantile_process
    is_multiple_estimation = (
        sum(len(v) for v in formula_dictionary.values()) > 1
        or run_split
        or (process is not None and len(process.quantiles) > 1)
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


@dataclass(frozen=True)
class ModelSpec:
    """A single model to fit, with everything the runner needs to do it.

    The fields are the constructor inputs that don't change over the
    course of the run; `fit_one` passes them to `model_cls`. The cache
    dicts (`lookup_demeaned_data` and, for non-quantreg methods,
    `lookup_preconditioner`) are deliberately *not* in here — the runner
    injects them at fit time so that specs sharing the same `cache_key`
    can share the cache.
    """

    method: EstimationMethod
    model_cls: ModelFactory
    formula: FixestFormula
    fixef_key: str | None
    data: pd.DataFrame
    options: EstimationOptions
    sample_split: SampleSplit | None
    quantile_process: QuantileProcess | None = None

    @property
    def cache_key(self) -> tuple[SampleSplit | None, str | None]:
        """Specs with the same key can share demean / preconditioner caches."""
        return (self.sample_split, self.fixef_key)


def build_all_splits(
    *,
    run_full: bool,
    run_split: bool,
    splitvar: str | None,
    data: pd.DataFrame,
) -> list[SampleSplit | None]:
    """List the sample splits in the order the runner will visit them.

    An unsplit estimation has the single split ``None``. Otherwise the
    full sample comes first if requested, followed by the sorted unique
    values of the split column. The order matches what `FixestMulti` did
    before the refactor, which keeps cache blocks contiguous downstream.
    """
    if not run_split:
        return [None]
    assert splitvar is not None
    all_splits: list[SampleSplit | None] = []
    if run_full:
        all_splits.append(SampleSplit(var=splitvar, value=None))
    all_splits.extend(
        SampleSplit(var=splitvar, value=value)
        for value in data[splitvar].dropna().drop_duplicates().sort_values().tolist()
    )
    return all_splits


def expand_specs(
    *,
    config: EstimationConfig,
    formula_dict: Mapping[str | None, list[FixestFormula]],
    data: pd.DataFrame,
    splits: list[SampleSplit | None],
    is_iv: bool,
) -> list[ModelSpec]:
    """Build one `ModelSpec` per model the user's call expands into.

    We iterate splits, then fixef keys, then formulas. The order is
    by design: specs that share a cache key
    end up next to each other in the list, so the runner can
    reuse its demean and preconditioner caches across them and
    drop them as soon as the cache key changes. Every spec shares
    the one options value the estimation function built.
    """
    entry = MODEL_REGISTRY[config.method]
    if not isinstance(config.options, entry.options_cls):
        raise TypeError(
            f"{config.method!r} models take {entry.options_cls.__name__}; "
            f"got {type(config.options).__name__}."
        )
    model_cls = _resolve_model_class(
        config.method,
        is_iv=is_iv,
        fits_quantile_process=config.quantile_process is not None,
    )

    return [
        ModelSpec(
            method=config.method,
            model_cls=model_cls,
            formula=formula,
            fixef_key=fixef_key,
            data=data,
            options=config.options,
            sample_split=sample_split,
            quantile_process=config.quantile_process,
        )
        for sample_split in splits
        for fixef_key in formula_dict
        for formula in formula_dict[fixef_key]
    ]


def fit_one(
    spec: ModelSpec,
    *,
    lookup_demeaned_data: dict[frozenset[int], DemeanedData],
    lookup_preconditioner: dict[frozenset[int], Preconditioner],
    vcov: VcovSpec,
) -> FittedModel:
    """Run the full fit pipeline for one model spec.

    Constructs the model class, runs prepare → fit → vcov → inference,
    and clears large attributes. `vcov` was parsed at the API boundary;
    the model rejects an estimator it does not support before fitting.
    The two per-cache-block dicts are injected here so they're shared
    across every spec in the block.

    Returns the fitted model.
    """
    entry = MODEL_REGISTRY[spec.method]
    model_kwargs: dict[str, Any] = {
        "FixestFormula": spec.formula,
        "data": spec.data,
        "options": spec.options,
        "sample_split": spec.sample_split,
        "lookup_demeaned_data": lookup_demeaned_data,
    }
    if entry.accepts_preconditioner:
        model_kwargs["lookup_preconditioner"] = lookup_preconditioner
    if spec.quantile_process is not None:
        # the fan-out itself is not an option of any single fit
        model_kwargs["quantile"] = spec.quantile_process.quantiles
        model_kwargs["multi_method"] = spec.quantile_process.multi_method

    FIT: FittedModel = spec.model_cls(**model_kwargs)

    FIT.prepare_model_matrix()
    FIT._validate_response()
    # if X is empty: no inference (empty X only as shorthand for demeaning)
    if not FIT._X_is_empty:
        FIT._check_vcov_support(vcov)
    FIT.get_fit()
    if not FIT._X_is_empty:
        FIT._vcov_from_spec(vcov)
        FIT.get_inference()
        FIT._finalize_fit()
    # delete large attributes
    FIT._clear_attributes()

    return FIT
