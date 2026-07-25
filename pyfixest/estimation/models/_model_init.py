from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.api.utils import _AllSampleSentinel
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.literals import SolverOptions


@dataclass(frozen=True)
class ModelInit:
    """Constructor arguments shared by every model class.

    Every model class takes one of these as its first argument and adds only
    its own extras as keyword arguments, so an option that applies to all
    estimators is declared once here instead of in each `__init__`. The
    planner builds it in `plan_._build_model_init`; the runner fills the two
    cache fields per cache block via `dataclasses.replace`.

    Attributes
    ----------
    FixestFormula : FixestFormula
        The parsed formula for this model.
    data : pandas.DataFrame
        The estimation data, before the sample split is applied.
    ssc_dict : dict[str, str | bool]
        Small-sample-correction options.
    drop_singletons : bool
        Whether to drop singleton fixed effects.
    drop_intercept : bool
        Whether to drop the intercept.
    weights : str | None
        Name of the weights column, or None for an unweighted fit.
    weights_type : str | None
        Either "aweights" or "fweights".
    collin_tol : float
        Tolerance for the collinearity check.
    lookup_demeaned_data : dict[frozenset[int], pandas.DataFrame]
        Demeaned columns cached across models sharing a cache key.
    lookup_preconditioner : dict[frozenset[int], Preconditioner] | None
        Within-preconditioner cached across models sharing a cache key.
    solver : SolverOptions
        Linear solver used for the fit.
    demeaner : AnyDemeaner | None
        Resolved demeaner configuration. None selects `MapDemeaner`.
    store_data : bool
        Whether to keep `_data` on the fitted model.
    copy_data : bool
        Whether to copy the input data before estimation.
    lean : bool
        Whether to strip large attributes after fitting.
    context : int | Mapping[str, Any]
        Evaluation scope handed to formulaic.
    sample_split_var : str | None
        Column the sample is split on, or None.
    sample_split_value : str | int | float | _AllSampleSentinel | None
        Value of `sample_split_var` this model is fitted on.
    """

    FixestFormula: FixestFormula
    data: pd.DataFrame
    ssc_dict: dict[str, str | bool]
    drop_singletons: bool
    drop_intercept: bool
    weights: str | None
    weights_type: str | None
    collin_tol: float
    lookup_demeaned_data: dict[frozenset[int], pd.DataFrame] = field(
        default_factory=dict
    )
    lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None
    solver: SolverOptions = "np.linalg.solve"
    demeaner: AnyDemeaner | None = None
    store_data: bool = True
    copy_data: bool = True
    lean: bool = False
    context: int | Mapping[str, Any] = 0
    sample_split_var: str | None = None
    sample_split_value: str | int | float | _AllSampleSentinel | None = None
