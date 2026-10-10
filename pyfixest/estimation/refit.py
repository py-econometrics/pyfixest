"""Refit a fitted model's estimator on other data.

Leave-one-cluster-out covariances (CRV3), randomization inference, the causal
cluster variance, and the IV first stage all rerun a fitted model's estimator.
`refit` is their single entry point, so every refit replays the estimation
options of the fit it starts from.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from pyfixest.core.demean import Preconditioner, within_preconditioner_name
from pyfixest.demeaners import AnyDemeaner, LsmrDemeaner
from pyfixest.estimation.config import EstimationConfig
from pyfixest.estimation.internals.retention import require_retained

if TYPE_CHECKING:
    import pandas as pd

    from pyfixest.estimation.internals.model_state import VcovSpec
    from pyfixest.estimation.models.feols_ import Feols


def refit(
    fit: Feols,
    *,
    data: pd.DataFrame,
    fml: str | None = None,
    vcov: VcovSpec,
    same_sample: bool = False,
) -> Feols:
    """Rerun the estimator of `fit` on `data` with its estimation options.

    The refit replays `fit.options` with `copy_data=False`, never modifies
    `data`, and returns a complete fitted model: the caller applies no
    storage options to it.

    With `same_sample`, `data` holds the rows of the fit's sample, with potentially
    some columns changed (e.g. for randomization inference, the IV first stage).
    The refit then reuses the fit's preconditioner and drops no singletons, and it
    raises if the rows of `data` do not match the initial fit. Otherwise `data`
    can be any other data (e.g. sample splits for CRV3, causal cluster variance).
    In this case, preconditioners are rebuilt from the new data, from which we drop
    separated observations etc.

    Parameters
    ----------
    fit : Feols
        The fitted model whose estimator and options are replayed.
    data : pd.DataFrame
        The data to fit, with the columns of the fit's sample.
    fml : str, optional
        The formula to fit; defaults to the fit's.
    vcov : VcovSpec
        The covariance estimator of the refit.
    same_sample : bool, optional
        Whether `data` has the rows of the fit's sample, so the refit must
        keep every row. Defaults to False.

    Returns
    -------
    Feols
        The fitted model, of the same estimator as `fit`.
    """
    # lazy loading to avoid circular import
    from pyfixest.estimation.models.feols_ import Feols
    from pyfixest.estimation.plan_ import estimation_method_of, parse_formula
    from pyfixest.estimation.runner import run_estimation

    require_retained(fit, "refit", "_data")
    # `run_estimation` resets the index, so compare it before fitting
    if same_sample and not data.index.equals(fit._data.index):
        raise ValueError(
            "`same_sample=True` requires `data` to have the same index as "
            "the data the model was fit on."
        )
    if same_sample:
        options = replace(
            fit.options,
            # do not drop singletons again (they are already dropped)
            drop_singletons=False,
            # reuse the preconditioner
            demeaner=_with_preconditioner(fit.options.demeaner, fit.preconditioner),
        )
    else:
        options = replace(
            fit.options,
            demeaner=_without_prebuilt_preconditioner(fit.options.demeaner),
        )
    config = EstimationConfig(
        method=estimation_method_of(type(fit)),
        data=data.copy(deep=False),
        fml=fit.model.formula if fml is None else fml,
        # the shallow copy above keeps `data` unchanged without a deep copy
        options=replace(options, copy_data=False),
        vcov=vcov,
    )
    # the caller reads the refit in full or throws it away
    result = run_estimation(config, parse_formula(config), apply_retention=False)
    if not isinstance(result, Feols):
        raise TypeError(f"A refit must return a single model, not {result!r}.")
    if same_sample and result.sample_info.n_obs != fit.sample_info.n_obs:
        raise ValueError(
            f"A refit on the fit's sample kept {result.sample_info.n_obs} of its "
            f"{fit.sample_info.n_obs} observations."
        )
    return result


def _with_preconditioner(
    demeaner: AnyDemeaner, preconditioner: Preconditioner | None
) -> AnyDemeaner:
    """Pass the fit's preconditioner to an LSMR demeaner on the same sample.

    A refit starts with an empty demean cache, so the fit's preconditioner is
    reused only when it is passed in. It depends on the fixed-effect codes,
    the weights, and the rows, which a same-sample refit keeps.
    """
    if isinstance(demeaner, LsmrDemeaner) and preconditioner is not None:
        return replace(demeaner, preconditioner=preconditioner)
    return demeaner


def _without_prebuilt_preconditioner(demeaner: AnyDemeaner) -> AnyDemeaner:
    """Replace a prebuilt LSMR `Preconditioner` by the name of its configuration.

    A prebuilt preconditioner is tied to the fixed-effect design it was built
    on, so a refit on another sample must build its own.
    """
    if not (
        isinstance(demeaner, LsmrDemeaner)
        and isinstance(demeaner.preconditioner, Preconditioner)
    ):
        return demeaner
    return replace(
        demeaner,
        preconditioner=within_preconditioner_name(demeaner.preconditioner),
    )
