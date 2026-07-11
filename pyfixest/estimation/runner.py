"""Run single and multiple estimation from parsed plans."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.estimation.config import EstimationConfig
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.models.feiv_ import Feiv
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.estimation.plan_ import (
    ParsedFormula,
    build_all_splits,
    expand_specs,
    fit_one,
)
from pyfixest.estimation.quantreg.QuantregMulti import QuantregMulti
from pyfixest.utils.dev_utils import _narwhals_to_pandas
from pyfixest.utils.utils import capture_context


def _prepare_data(config: EstimationConfig) -> pd.DataFrame:
    """Convert input data to pandas with a clean RangeIndex.

    Reindexing is required because formulaic's model matrix starts from 0:N
    and downstream `dropna()` calls would otherwise produce mis-aligned indices.
    """
    data = _narwhals_to_pandas(config.data)
    if config.copy_data:
        data = data.copy()
    data.reset_index(drop=True, inplace=True)
    return data


def _split_plan(config: EstimationConfig) -> tuple[bool, bool, str | None]:
    """Derive `(run_full, run_split, splitvar)` from the split/fsplit options."""
    split = config.split
    fsplit = config.fsplit
    run_split = split is not None or fsplit is not None
    run_full = not (split and not fsplit)
    splitvar: str | None = (split or fsplit) if run_split else None
    return run_full, run_split, splitvar


def run_estimation(
    config: EstimationConfig,
    parsed: ParsedFormula,
) -> Feols | Fepois | Feiv | FixestMulti:
    """Fit every spec the user's call expands into; unwrap when a single model was asked for.

    Prepares the runtime inputs (data, context, split plan), builds a
    `FixestMulti` results container, fits models based on the planner's
    specs block-by-block (sharing the demean / preconditioner cache within
    each `cache_key` block), and returns either the multi-object or the
    single fitted model.
    """
    data = _prepare_data(config)
    context: Mapping[str, Any] = capture_context(config.context)
    run_full, run_split, splitvar = _split_plan(config)

    fixest = FixestMulti(config=config, parsed=parsed, data=data, context=context)

    all_splits = build_all_splits(
        run_full=run_full,
        run_split=run_split,
        splitvar=splitvar,
        data=data,
    )

    specs = expand_specs(
        config=config,
        formula_dict=parsed.formula_dict,
        data=data,
        splits=all_splits,
        is_iv=parsed.is_iv,
        splitvar=splitvar,
        captured_context=context,
    )

    _NO_CACHE_KEY: Any = object()
    prev_cache_key: Any = _NO_CACHE_KEY
    lookup_demeaned_data: dict[frozenset[int], pd.DataFrame] = {}
    lookup_preconditioner: dict[frozenset[int], Preconditioner] = {}

    for spec in specs:
        if spec.cache_key != prev_cache_key:
            lookup_demeaned_data = {}
            lookup_preconditioner = {}
            prev_cache_key = spec.cache_key

        FIT = fit_one(
            spec,
            lookup_demeaned_data=lookup_demeaned_data,
            lookup_preconditioner=lookup_preconditioner,
            vcov=config.vcov,
            vcov_kwargs=config.vcov_kwargs,
        )

        if isinstance(FIT, QuantregMulti):
            for q_model in FIT.all_quantregs.values():
                fixest.all_fitted_models[q_model._model_name] = q_model
        else:
            fixest.all_fitted_models[FIT._model_name] = FIT

    if parsed.is_multiple_estimation:
        return fixest
    return fixest.fetch_model(0, print_fml=False)
