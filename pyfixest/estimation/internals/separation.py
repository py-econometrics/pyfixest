from __future__ import annotations

import warnings
from functools import partial
from typing import Protocol

import numpy as np
import pandas as pd

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
    demeaner: AnyDemeaner,
    methods: list[str] | None = None,
) -> list[int]:
    """
    Check for separation.

    Check for separation of Poisson Regression. For details, see the ppmlhdfe
    documentation on separation checks.

    Parameters
    ----------
    Y : pd.DataFrame
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    fe : pd.DataFrame
        Fixed effects.
    demeaner : AnyDemeaner
        Demeaner configuration used by the estimated model.
    methods: list[str], optional
        Methods used to check for separation. One of fixed effects ("fe") or
        iterative rectifier ("ir"). Executes all methods by default.

    Returns
    -------
    list
        List of indices of observations that are removed due to separation.
    """
    valid_methods: dict[str, _SeparationMethod] = {
        "fe": _check_for_separation_fe,
        "ir": partial(_check_for_separation_ir, demeaner=demeaner),
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
        separation_na = separation_na.union(valid_methods[method](Y=Y, X=X, fe=fe))

    if separation_na:
        warnings.warn(
            f"{len(separation_na)!s} observations removed because of separation.",
            UserWarning,
            stacklevel=_find_stack_level(),
        )

    return list(separation_na)


class _SeparationMethod(Protocol):
    def __call__(
        self, *, Y: pd.DataFrame, X: pd.DataFrame, fe: pd.DataFrame
    ) -> set[int]: ...


def _check_for_separation_fe(
    *, Y: pd.DataFrame, X: pd.DataFrame, fe: pd.DataFrame
) -> set[int]:
    """
    Check for separation using the "fe" check.

    Parameters
    ----------
    Y : pd.DataFrame
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    fe : pd.DataFrame
        Fixed effects.

    Returns
    -------
    set
        Set of indices of separated observations.
    """
    separation_na: set[int] = set()
    if fe is not None and not (Y > 0).all(axis=0).all():
        Y_help = (Y.iloc[:, 0] > 0).astype(int)

        # loop over all elements of fe
        for x in fe.columns:
            ctab = pd.crosstab(Y_help, fe[x])
            null_column = ctab.xs(0)
            # sep_candidate if
            # fixed effect level has only observations with Y > 0
            sep_candidate = ((ctab > 0).sum(axis=0).to_numpy() == 1) & (
                null_column > 0
            ).to_numpy().flatten()
            # droplist: list of levels to drop
            droplist = ctab.xs(0)[sep_candidate].index.tolist()

            # dropset: list of indices to drop
            if len(droplist) > 0:
                fe_in_droplist = fe[x].isin(droplist)
                dropset = set(fe[x][fe_in_droplist].index)
                separation_na = separation_na.union(dropset)

    return separation_na


def _check_for_separation_ir(
    *,
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: pd.DataFrame,
    demeaner: AnyDemeaner,
    tol: float = 1e-4,
    maxiter: int = 100,
) -> set[int]:
    """
    Check for separation using the "iterative rectifier" algorithm
    proposed by Correia et al. (2021). For details see http://arxiv.org/abs/1903.01633.

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
    fe : pd.DataFrame
        Fixed effects.
    demeaner : AnyDemeaner
        Demeaner configuration used by the estimated model.
    tol : float
        Tolerance to detect separated observation. Defaults to 1e-4.
    maxiter : int
        Maximum number of iterations. Defaults to 100.

    Returns
    -------
    set
        Set of indices of separated observations.
    """
    dependent = Y.iloc[:, 0]
    is_interior = dependent > 0
    if is_interior.all():
        return set()

    # Project on the original materialized design. Stateful terms and FE codes
    # must not be reevaluated after the outer fit filters its rows.
    response = (dependent == 0).to_numpy(dtype=np.float64)[:, None]
    weights = np.where(is_interior, is_interior.sum() / tol**2, 1.0)
    fixed_effects = fe.to_numpy()
    cache = DemeanCache()
    sample = frozenset()
    design = X.to_numpy(dtype=np.float64)
    if design.shape[1]:
        design = cache.demean_array(
            x=design,
            flist=fixed_effects,
            weights=weights,
            na_index=sample,
            demeaner=demeaner,
        )
        design, _ = drop_multicollinear_variables(
            X=design,
            names=X.columns.tolist(),
            collin_tol=1e-9,
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
            flist=fixed_effects,
            weights=weights,
            na_index=sample,
            demeaner=demeaner,
        )
        if design.shape[1]:
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
