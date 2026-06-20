from __future__ import annotations

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


def run_estimation(
    config: EstimationConfig,
    parsed: ParsedFormula,
) -> Feols | Fepois | Feiv | FixestMulti:
    """Fit every spec the user's call expands into; unwrap when a single model was asked for.

    Builds a `FixestMulti` results container, fits models based on the planner's
    specs block-by-block (sharing the demean / preconditioner cache within
    each `cache_key` block), and returns either the multi-object or the
    single fitted model.
    """
    fixest = FixestMulti(config, parsed)

    all_splits = build_all_splits(
        run_full=fixest._run_full,
        run_split=fixest._run_split,
        splitvar=fixest._splitvar,
        data=fixest._data,
    )

    specs = expand_specs(
        config=config,
        formula_dict=parsed.formula_dict,
        data=fixest._data,
        splits=all_splits,
        is_iv=parsed.is_iv,
        splitvar=fixest._splitvar,
        captured_context=fixest._context,
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
