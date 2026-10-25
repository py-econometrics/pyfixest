from __future__ import annotations

import warnings
from functools import partial
from typing import Protocol

import numpy as np
import pandas as pd
from within import Effect

from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.internals.collinearity import drop_multicollinear_variables
from pyfixest.estimation.internals.demean_ import DemeanCache
from pyfixest.estimation.internals.solvers import solve_ols
from pyfixest.utils.dev_utils import _find_stack_level


def check_for_separation(
    *,
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: pd.DataFrame,
    effects: np.ndarray | list[Effect],
    outcome_bounds: tuple[float, ...],
    demeaner: AnyDemeaner,
    collin_tol: float,
    methods: list[str] | None = None,
) -> list[int]:
    """
    Check for separation.

    An observation is separated when its outcome sits at a bound that the
    inverse link attains only as the linear predictor diverges, e.g. zero for
    Poisson and zero or one for binary models. For details, see Correia,
    Guimarães and Zylkin (2021), http://arxiv.org/abs/1903.01633.

    Parameters
    ----------
    Y : pd.DataFrame
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    fe : pd.DataFrame
        Codes of the fixed effects that have level intercepts. Slope-only
        effects are excluded, because they cannot diverge to fit a level of
        outcomes at the bound.
    effects : np.ndarray or list[within.Effect]
        All fixed effects in the form the demeaner consumes.
    outcome_bounds : tuple[float, ...]
        Outcome values the family's inverse link attains only in the limit.
    demeaner : AnyDemeaner
        Demeaner configuration used by the estimated model.
    collin_tol : float
        Collinearity tolerance used by the estimated model.
    methods: list[str], optional
        Methods used to check for separation. One of fixed effects ("fe") or
        iterative rectifier ("ir"). Executes all methods by default.

    Returns
    -------
    list
        List of indices of observations that are removed due to separation.
    """
    valid_methods: dict[str, _SeparationMethod] = {
        "fe": partial(_check_for_separation_fe, fe=fe),
        "ir": partial(
            _check_for_separation_ir,
            effects=effects,
            demeaner=demeaner,
            collin_tol=collin_tol,
        ),
    }
    if methods is None:
        methods = list(valid_methods)

    invalid_methods = [method for method in methods if method not in valid_methods]
    if invalid_methods:
        raise ValueError(
            f"Invalid separation method. Expecting {list(valid_methods)}. Received {invalid_methods}"
        )

    separation_na: set[int] = set()
    for method in methods:
        separation_na = separation_na.union(
            valid_methods[method](Y=Y, X=X, outcome_bounds=outcome_bounds)
        )

    if separation_na:
        warnings.warn(
            f"{len(separation_na)!s} observations removed because of separation.",
            UserWarning,
            stacklevel=_find_stack_level(),
        )

    return list(separation_na)


class _SeparationMethod(Protocol):
    def __call__(
        self,
        *,
        Y: pd.DataFrame,
        X: pd.DataFrame,
        outcome_bounds: tuple[float, ...],
    ) -> set[int]: ...


def _check_for_separation_fe(
    *,
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: pd.DataFrame,
    outcome_bounds: tuple[float, ...],
) -> set[int]:
    """
    Check for separation using the "fe" check.

    Drops fixed-effect levels whose retained outcomes all sit at one outcome
    bound. Dropping a level can leave a level of another fixed effect at a
    bound, so the check repeats until no level is dropped.

    Parameters
    ----------
    Y : pd.DataFrame
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    fe : pd.DataFrame
        Fixed effects.
    outcome_bounds : tuple[float, ...]
        Outcome values the family's inverse link attains only in the limit.

    Returns
    -------
    set
        Set of indices of separated observations.
    """
    dependent = Y.iloc[:, 0].to_numpy()
    retained = np.ones(dependent.shape[0], dtype=bool)
    dropped_level = True
    while dropped_level:
        dropped_level = False
        for bound in outcome_bounds:
            interior = dependent != bound
            for column in fe.columns:
                levels = fe[column].to_numpy()
                interior_levels = np.unique(levels[retained & interior])
                separated = retained & ~np.isin(levels, interior_levels)
                if separated.any():
                    retained &= ~separated
                    dropped_level = True

    return set(Y.index[~retained])


def _check_for_separation_ir(
    *,
    Y: pd.DataFrame,
    X: pd.DataFrame,
    effects: np.ndarray | list[Effect],
    outcome_bounds: tuple[float, ...],
    demeaner: AnyDemeaner,
    collin_tol: float,
    tol: float = 1e-4,
    maxiter: int = 100,
) -> set[int]:
    """
    Check for separation using the "iterative rectifier" algorithm
    proposed by Correia et al. (2021). For details see http://arxiv.org/abs/1903.01633.

    The rectifier is one-sided: it detects separation at a single outcome
    bound, so `feglm()` skips it for logit and probit.

    The inputs contain the outer model's evaluated columns and retained rows.
    Auxiliary weighted projections preserve that sample and the absorbed FE
    span, which includes a constant even without an explicit intercept column.
    Collinear covariates are selected once after weighted FE absorption.

    Parameters
    ----------
    Y : pd.DataFrame
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    effects : np.ndarray or list[within.Effect]
        Fixed effects in the form the demeaner consumes.
    outcome_bounds : tuple[float, ...]
        The single outcome value the family's inverse link attains only in the
        limit.
    demeaner : AnyDemeaner
        Demeaner configuration used by the estimated model.
    collin_tol : float
        Collinearity tolerance used by the estimated model.
    tol : float
        Tolerance to detect separated observation. Defaults to 1e-4.
    maxiter : int
        Maximum number of iterations. Defaults to 100.

    Returns
    -------
    set
        Set of indices of separated observations.
    """
    (bound,) = outcome_bounds
    dependent = Y.iloc[:, 0]
    is_interior = dependent != bound
    if is_interior.all():
        return set()

    # Project on the original materialized design. Stateful terms and FE codes
    # must not be reevaluated after the outer fit filters its rows.
    response = (~is_interior).to_numpy(dtype=np.float64)[:, None]
    weights = np.where(is_interior, is_interior.sum() / tol**2, 1.0)
    cache = DemeanCache()
    sample = frozenset()
    design = X.to_numpy(dtype=np.float64)
    has_regressors = design.shape[1] > 0
    if has_regressors:
        design = cache.demean_array(
            x=design,
            flist=effects,
            weights=weights,
            na_index=sample,
            demeaner=demeaner,
        )
        design, _ = drop_multicollinear_variables(
            X=design,
            names=X.columns.tolist(),
            collin_tol=collin_tol,
        )
    hessian = design.T @ (weights[:, None] * design)

    iteration = 0
    has_converged = False
    while iteration < maxiter:
        iteration += 1
        # regress U on X
        # TODO: check acceleration in ppmlhdfe's implementation: https://github.com/sergiocorreia/ppmlhdfe/blob/master/src/ppmlhdfe_separation_relu.mata#L135
        response_demeaned = cache.demean_array(
            x=response,
            flist=effects,
            weights=weights,
            na_index=sample,
            demeaner=demeaner,
        )
        if has_regressors:
            beta = solve_ols(
                tZX=hessian,
                tZY=design.T @ (weights[:, None] * response_demeaned),
                solver="scipy.linalg.solve",
            )
            residuals = response_demeaned[:, 0] - design @ beta
        else:
            residuals = response_demeaned[:, 0]
        # Within residuals equal full-model residuals, including the FE fit.
        Uhat = pd.Series(response[:, 0] - residuals, index=dependent.index)
        # update when within tolerance of zero
        # need to be more strict below zero to avoid false positives
        within_zero = (Uhat > -0.1 * tol) & (Uhat < tol)
        Uhat.where(~(is_interior | within_zero.fillna(True)), 0, inplace=True)
        if (Uhat >= 0).all():
            # all separated observations have been identified
            has_converged = True
            break
        # rectified linear unit (ReLU)
        response[~is_interior.to_numpy(), 0] = np.fmax(Uhat[~is_interior], 0)

    separation_na: set[int] = set()
    if has_converged:
        separation_na = set(dependent[Uhat > 0].index)
    else:
        warnings.warn(
            "iterative rectivier separation check: maximum number of iterations reached before convergence",
            RuntimeWarning,
            stacklevel=_find_stack_level(),
        )

    return separation_na
