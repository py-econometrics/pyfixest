"""Refit a fitted model's estimator on other data.

Leave-one-cluster-out covariances (CRV3), randomization inference, the causal
cluster variance, and the IV first stage all rerun a fitted model's estimator.
`refit` is their single entry point, so every refit replays the estimation
options of the fit it starts from.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner, LsmrDemeaner, LsmrPreconditioner
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

    With `same_sample`, `data` holds the rows of the fit's sample, with some
    columns changed (randomization inference, the IV first stage). The refit
    then reuses the fit's preconditioner and drops no singletons, and it
    raises if `data` has other rows or the refit loses one. Otherwise `data`
    is any other data (CRV3, causal cluster variance): a prebuilt
    preconditioner belongs to the fit's sample and is rebuilt, and NAs and
    singletons are dropped as in the fit.

    Parameters
    ----------
    fit : Feols
        The fitted model whose estimator and options are replayed.
    data : pd.DataFrame
        The data to fit, with the columns of the fit's sample.
    fml : str, optional
        The formula to fit; defaults to the fit's. It must keep the fit's
        fixed effects.
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
            "A refit with `same_sample=True` needs the rows of the fit's "
            "sample; `data` has another index."
        )
    if same_sample:
        options = replace(
            fit.options,
            drop_singletons=False,
            demeaner=_with_preconditioner(fit.options.demeaner, fit.preconditioner),
        )
    else:
        options = replace(
            fit.options,
            demeaner=_without_prebuilt_preconditioner(fit.options.demeaner),
        )
    config = EstimationConfig(
        method=estimation_method_of(type(fit)),
        # a shallow copy absorbs the runner's in-place index reset; the
        # model copies (or copy-on-write isolates) the frame before any
        # other write
        data=data.copy(deep=False),
        fml=fit.model.formula if fml is None else fml,
        # the shallow copy above keeps `data` intact without a deep copy
        options=replace(options, copy_data=False),
        vcov=vcov,
    )
    parsed = parse_formula(config)
    # keyed like `Formula.parse_to_dict`: the fixed effects, or None
    fit_formula = fit.model.fixest_formula
    fixed_effects = (
        str(fit_formula.fixed_effects) if fit_formula.is_fixed_effects else None
    )
    if list(parsed.formula_dict) != [fixed_effects]:
        raise ValueError(
            f"A refit must keep the fixed effects of the fit ({fixed_effects}); "
            f"got the formula {config.fml!r}."
        )

    # the caller reads the refit in full or throws it away
    result = run_estimation(config, parsed, apply_retention=False)
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
    """Replace a prebuilt LSMR `Preconditioner` by the name of its variant.

    A prebuilt preconditioner is tied to the fixed-effect design it was built
    on, so a refit on another sample must build its own. Variants without a
    public name fall back to ``"auto"``, the `LsmrDemeaner` default.
    """
    if not (
        isinstance(demeaner, LsmrDemeaner)
        and isinstance(demeaner.preconditioner, Preconditioner)
    ):
        return demeaner
    variant = demeaner.preconditioner.variant.lower()
    preconditioner: LsmrPreconditioner = (
        "additive"
        if variant == "additive"
        else "diagonal"
        if variant == "diagonal"
        else "auto"
    )
    return replace(demeaner, preconditioner=preconditioner)
