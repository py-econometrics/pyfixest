"""Estimation plan and runner.

Splits multiple estimation into three explicit pieces:

- ``ModelSpec``: everything needed to fit ONE model (formula, model class,
  sample split, constructor options).
- ``expand_specs``: pure function expanding a parsed formula dictionary,
  sample splits, and estimation options into an ordered ``list[ModelSpec]``.
- ``fit_one``: runs the fit pipeline (``prepare_model_matrix -> get_fit ->
  vcov -> inference -> performance -> clear``) for a single spec — one
  place, one order, for all model types.

``MODEL_REGISTRY`` maps each estimation method to its model class and the
extra constructor arguments it consumes, replacing the per-method if-chains
that previously lived in ``FixestMulti._estimate_all_models``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pyfixest.estimation.api.utils import _ALL_SAMPLE, _AllSampleSentinel
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.vcov_utils import _get_vcov_type
from pyfixest.estimation.models.fegaussian_ import Fegaussian
from pyfixest.estimation.models.feiv_ import Feiv
from pyfixest.estimation.models.felogit_ import Felogit
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.feols_compressed_ import FeolsCompressed
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.estimation.models.feprobit_ import Feprobit
from pyfixest.estimation.quantreg.quantreg_ import Quantreg
from pyfixest.estimation.quantreg.QuantregMulti import QuantregMulti


@dataclass(frozen=True)
class ModelEntry:
    """Registry entry: the model class for a method and the option groups
    its constructor consumes (on top of the shared arguments)."""

    model_cls: type
    needs: frozenset[str] = frozenset()


MODEL_REGISTRY: dict[str, ModelEntry] = {
    # "feols" resolves to Feiv when the formula has a first stage,
    # see _resolve_model_class
    "feols": ModelEntry(Feols, frozenset({"demeaner"})),
    "fepois": ModelEntry(Fepois, frozenset({"demeaner", "iwls", "offset"})),
    "feglm-logit": ModelEntry(Felogit, frozenset({"demeaner", "iwls", "accelerate"})),
    "feglm-probit": ModelEntry(Feprobit, frozenset({"demeaner", "iwls", "accelerate"})),
    "feglm-gaussian": ModelEntry(
        Fegaussian, frozenset({"demeaner", "iwls", "accelerate"})
    ),
    "compression": ModelEntry(FeolsCompressed, frozenset({"compression"})),
    "quantreg": ModelEntry(Quantreg, frozenset({"quantreg"})),
    "quantreg_multi": ModelEntry(
        QuantregMulti, frozenset({"quantreg", "quantreg_multi"})
    ),
}


def _resolve_model_class(method: str, is_iv: bool) -> type:
    "Resolve the model class for a method; feols dispatches on is_iv."
    if method == "feols" and is_iv:
        return Feiv
    return MODEL_REGISTRY[method].model_cls


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to fit ONE model.

    ``model_kwargs`` holds the constructor arguments except ``data`` and
    ``lookup_demeaned_data``, which the runner injects (the data is shared,
    the demeaning cache is shared per ``cache_key`` block).
    """

    method: str
    model_cls: type
    formula: FixestFormula
    fixef_key: str
    sample_split_var: str | None
    sample_split_value: Any
    model_kwargs: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def cache_key(self) -> tuple[Any, str]:
        """Models with the same key share one demeaning cache block
        (same sample split and same fixed effects)."""
        return (self.sample_split_value, self.fixef_key)


def expand_specs(
    *,
    formula_dict: dict[str, list[FixestFormula]],
    method: str,
    is_iv: bool,
    data: pd.DataFrame,
    splitvar: str | None,
    run_full: bool,
    run_split: bool,
    ssc_dict: dict[str, str | bool],
    drop_singletons: bool,
    drop_intercept: bool,
    weights: str | None,
    weights_type: str,
    solver: str,
    collin_tol: float,
    store_data: bool,
    copy_data: bool,
    lean: bool,
    context: Any,
    demeaner: Any = None,
    iwls_tol: float = 1e-08,
    iwls_maxiter: int = 25,
    separation_check: list[str] | None = None,
    offset: str | None = None,
    accelerate: bool = True,
    quantile: float | list[float] | None = None,
    quantreg_method: str = "fn",
    quantile_tol: float = 1e-06,
    quantile_maxiter: int | None = None,
    quantreg_multi_method: str = "cfm1",
    reps: int | None = None,
    seed: int | None = None,
) -> list[ModelSpec]:
    """Expand formulas, sample splits, and options into an ordered model plan.

    Pure: reads ``data`` only to enumerate the sample-split values. The
    returned order matches the legacy triple loop
    (splits x fixef keys x formulas), which also defines the demean-cache
    block boundaries via ``ModelSpec.cache_key``.
    """
    entry = MODEL_REGISTRY[method]

    all_splits: list[str | int | float | _AllSampleSentinel] = []
    if run_full:
        all_splits.append(_ALL_SAMPLE)
    if run_split:
        all_splits.extend(
            data[splitvar].dropna().drop_duplicates().sort_values().tolist()
        )

    specs: list[ModelSpec] = []
    for sample_split_value in all_splits:
        for fixef_key, formulas in formula_dict.items():
            for formula in formulas:
                model_kwargs: dict[str, Any] = {
                    "FixestFormula": formula,
                    "ssc_dict": ssc_dict,
                    "drop_singletons": drop_singletons,
                    "drop_intercept": drop_intercept,
                    "weights": weights,
                    "weights_type": weights_type,
                    "solver": solver,
                    "collin_tol": collin_tol,
                    "store_data": store_data,
                    "copy_data": copy_data,
                    "lean": lean,
                    "context": context,
                    "sample_split_value": sample_split_value,
                    "sample_split_var": splitvar,
                }

                if "demeaner" in entry.needs:
                    model_kwargs["demeaner"] = demeaner
                if "iwls" in entry.needs:
                    model_kwargs.update(
                        {
                            "separation_check": separation_check,
                            "tol": iwls_tol,
                            "maxiter": iwls_maxiter,
                        }
                    )
                if "offset" in entry.needs:
                    model_kwargs["offset"] = offset
                if "accelerate" in entry.needs:
                    model_kwargs["accelerate"] = accelerate
                if "quantreg" in entry.needs:
                    model_kwargs.update(
                        {
                            "quantile": quantile,
                            "method": quantreg_method,
                            "quantile_tol": quantile_tol,
                            "quantile_maxiter": quantile_maxiter,
                            "seed": seed,
                        }
                    )
                if "quantreg_multi" in entry.needs:
                    model_kwargs["multi_method"] = quantreg_multi_method
                if "compression" in entry.needs:
                    model_kwargs.update({"reps": reps, "seed": seed})

                specs.append(
                    ModelSpec(
                        method=method,
                        model_cls=_resolve_model_class(method, is_iv),
                        formula=formula,
                        fixef_key=fixef_key,
                        sample_split_var=splitvar,
                        sample_split_value=sample_split_value,
                        model_kwargs=model_kwargs,
                    )
                )

    return specs


def fit_one(
    spec: ModelSpec,
    data: pd.DataFrame,
    lookup_demeaned_data: dict[frozenset[int], pd.DataFrame],
    vcov: str | dict[str, str] | None,
    vcov_kwargs: dict[str, Any] | None = None,
):
    """Construct and fit a single model from a spec.

    Owns the fit pipeline: ``prepare_model_matrix -> get_fit -> vcov ->
    get_inference -> get_performance -> wald_test -> first_stage ->
    _clear_attributes`` — one place, one order, for all model types.
    """
    fit = spec.model_cls(
        data=data,
        lookup_demeaned_data=lookup_demeaned_data,
        **spec.model_kwargs,
    )

    fit.prepare_model_matrix()
    if isinstance(fit, (Felogit, Feprobit, Fegaussian)):
        fit._check_dependent_variable()
    fit.get_fit()

    # if X is empty: no inference (empty X only as shorthand for demeaning)
    if not fit._X_is_empty:
        vcov_type = _get_vcov_type(vcov)
        fit.vcov(
            vcov=vcov_type,
            vcov_kwargs=vcov_kwargs,
            data=fit._data
            if not isinstance(fit, QuantregMulti)
            else fit.all_quantregs[fit.quantiles[0]]._data,
        )  #  a little hacky, but works

        fit.get_inference()
        if spec.method == "feols" and not fit._is_iv:
            fit.get_performance()
            if isinstance(fit, Feols):
                fit.wald_test()
        if isinstance(fit, Feiv):
            fit.first_stage()

    # delete large attributes
    fit._clear_attributes()

    return fit
