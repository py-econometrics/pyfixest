from __future__ import annotations

import warnings
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import pandas as pd

from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.config import EstimationConfig
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.internals.model_state import EstimationOptions, VcovSpec
from pyfixest.utils.dev_utils import _find_stack_level
from pyfixest.utils.utils import ssc

if TYPE_CHECKING:
    from pyfixest.estimation.models.feols_ import Feols


def check_for_separation(
    fml: Formula,
    data: pd.DataFrame,
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: pd.DataFrame,
    demeaner: AnyDemeaner,
    methods: list[str] | None = None,
    context: Mapping[str, Any] | None = None,
) -> list[int]:
    """
    Check for separation.

    Check for separation of Poisson Regression. For details, see the ppmlhdfe
    documentation on separation checks.

    Parameters
    ----------
    fml : Formula
        The formula used for estimation.
    data : pd.DataFrame
        The data used for estimation.
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
    context : Mapping[str, Any], optional
        Captured formula context forwarded to iterative-rectifier refits.

    Returns
    -------
    list
        List of indices of observations that are removed due to separation.
    """
    valid_methods: dict[str, _SeparationMethod] = {
        "fe": _check_for_separation_fe,
        "ir": partial(_check_for_separation_ir, demeaner=demeaner, context=context),
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
            valid_methods[method](fml=fml, data=data, Y=Y, X=X, fe=fe)
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
        fml: Formula,
        data: pd.DataFrame,
        Y: pd.DataFrame,
        X: pd.DataFrame,
        fe: pd.DataFrame,
    ) -> set[int]:
        """
        Check for separation.

        Parameters
        ----------
        fml : Formula
            The formula used for estimation.
        data : pd.DataFrame
            The data used for estimation.
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
        ...


def _check_for_separation_fe(
    fml: Formula, data: pd.DataFrame, Y: pd.DataFrame, X: pd.DataFrame, fe: pd.DataFrame
) -> set[int]:
    """
    Check for separation using the "fe" check.

    Parameters
    ----------
    fml : Formula
        The formula used for estimation.
    data : pd.DataFrame
        The data used for estimation.
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
    fml: Formula,
    data: pd.DataFrame,
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: pd.DataFrame,
    demeaner: AnyDemeaner,
    tol: float = 1e-4,
    maxiter: int = 100,
    context: Mapping[str, Any] | None = None,
) -> set[int]:
    """
    Check for separation using the "iterative rectifier" algorithm
    proposed by Correia et al. (2021). For details see http://arxiv.org/abs/1903.01633.

    Parameters
    ----------
    fml : Formula
        The formula used for estimation.
    data : pd.DataFrame
        The data used for estimation.
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
    context : Mapping[str, Any], optional
        Captured formula context used to evaluate the auxiliary formula.

    Returns
    -------
    set
        Set of indices of separated observations.
    """
    # lazy load to avoid circular import
    from pyfixest.estimation.plan_ import parse_formula
    from pyfixest.estimation.runner import run_estimation

    # initialize
    separation_na: set[int] = set()
    tmp_suffix = "_separationTmp"
    name_dependent_separation = "U"
    while name_dependent_separation in data.columns:
        name_dependent_separation += tmp_suffix
    name_weights = "omega"
    while name_weights in data.columns:
        name_weights += tmp_suffix

    fml_separation = fml.with_dependent(name=name_dependent_separation)

    dependent = Y.iloc[:, 0]
    is_interior = dependent > 0
    if is_interior.all():
        # no boundary sample, can exit
        return separation_na

    # initialize variables
    tmp = data.copy(deep=False)
    tmp[name_dependent_separation] = (dependent == 0).astype(float)
    # weights
    N0 = (dependent > 0).sum()
    K = N0 / tol**2
    tmp[name_weights] = np.where(dependent > 0, K, 1)

    # Auxiliary weighted OLS uses the public feols defaults and the outer
    # fit's demeaner and evaluation context. The parsed plan is reused across
    # iterations; the GLM itself has not been fitted yet, so refit cannot run.
    config = EstimationConfig(
        method="feols",
        data=tmp,
        fml=fml_separation.formula,
        options=EstimationOptions(
            ssc=ssc(),
            drop_singletons=True,
            drop_intercept=False,
            weights=name_weights,
            weights_type="aweights",
            offset=None,
            collin_tol=1e-9,
            solver="scipy.linalg.solve",
            demeaner=demeaner,
            store_data=True,
            copy_data=True,
            lean=False,
            context=context if context is not None else {},
        ),
        vcov=VcovSpec(vcov_type="iid", vcov_type_detail="iid"),
    )
    parsed = parse_formula(config, formula=fml_separation)

    iteration = 0
    has_converged = False
    while iteration < maxiter:
        iteration += 1
        # regress U on X
        # TODO: check acceleration in ppmlhdfe's implementation: https://github.com/sergiocorreia/ppmlhdfe/blob/master/src/ppmlhdfe_separation_relu.mata#L135
        fitted = cast(
            "Feols",
            run_estimation(config, parsed, apply_retention=False),
        )
        # The inner fit resets its index; predictions retain tmp's row order.
        Uhat = pd.Series(fitted.predict(), index=tmp.index)
        # update when within tolerance of zero
        # need to be more strict below zero to avoid false positives
        within_zero = (Uhat > -0.1 * tol) & (Uhat < tol)
        Uhat.where(~(is_interior | within_zero.fillna(True)), 0, inplace=True)
        if (Uhat >= 0).all():
            # all separated observations have been identified
            has_converged = True
            break
        tmp.loc[~is_interior, name_dependent_separation] = np.fmax(
            Uhat[~is_interior], 0
        )  # rectified linear unit (ReLU)

    if has_converged:
        separation_na = set(dependent[Uhat > 0].index)
    else:
        warnings.warn(
            "iterative rectivier separation check: maximum number of iterations reached before convergence",
            RuntimeWarning,
            stacklevel=_find_stack_level(),
        )

    return separation_na
