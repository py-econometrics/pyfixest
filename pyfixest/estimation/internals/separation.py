"""Detect separated observations in fixed-effects models."""

from __future__ import annotations

import re
import warnings
from importlib import import_module
from typing import Protocol

import numpy as np
import pandas as pd


def check_for_separation(
    fml: str,
    data: pd.DataFrame,
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: pd.DataFrame,
    methods: list[str] | None = None,
) -> list[int]:
    """
    Check for separation.

    Check for separation of Poisson Regression. For details, see the ppmlhdfe
    documentation on separation checks.

    Parameters
    ----------
    fml : str
        The formula used for estimation.
    data : pd.DataFrame
        The data used for estimation.
    Y : pd.DataFrame
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    fe : pd.DataFrame
        Fixed effects.
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
        "ir": _check_for_separation_ir,
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
            f"{len(separation_na)!s} observations removed because of separation."
        )

    return list(separation_na)


class _SeparationMethod(Protocol):
    def __call__(
        self,
        fml: str,
        data: pd.DataFrame,
        Y: pd.DataFrame,
        X: pd.DataFrame,
        fe: pd.DataFrame,
    ) -> set[int]:
        """
        Check for separation.

        Parameters
        ----------
        fml : str
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
    fml: str, data: pd.DataFrame, Y: pd.DataFrame, X: pd.DataFrame, fe: pd.DataFrame
) -> set[int]:
    """
    Check for separation using the "fe" check.

    Parameters
    ----------
    fml : str
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
    fml: str,
    data: pd.DataFrame,
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: pd.DataFrame,
    tol: float = 1e-4,
    maxiter: int = 100,
) -> set[int]:
    """
    Check for separation using the "iterative rectifier" algorithm
    proposed by Correia et al. (2021). For details see http://arxiv.org/abs/1903.01633.

    Parameters
    ----------
    fml : str
        The formula used for estimation.
    data : pd.DataFrame
        The data used for estimation.
    Y : pd.DataFrame
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    fe : pd.DataFrame
        Fixed effects.
    tol : float
        Tolerance to detect separated observation. Defaults to 1e-4.
    maxiter : int
        Maximum number of iterations. Defaults to 100.

    Returns
    -------
    set
        Set of indices of separated observations.
    """
    # lazy load to avoid circular import
    fixest_module = import_module("pyfixest.estimation")
    feols = fixest_module.feols
    # initialize
    separation_na: set[int] = set()
    tmp_suffix = "_separationTmp"
    # build formula
    name_dependent, rest = re.split(r"\s*~\s*", fml, maxsplit=1)
    name_dependent_separation = "U"
    if name_dependent_separation in data.columns:
        name_dependent_separation += tmp_suffix

    fml_separation = f"{name_dependent_separation} ~ {rest}"

    dependent: pd.Series = data[name_dependent]
    is_interior = dependent > 0
    if is_interior.all():
        # no boundary sample, can exit
        return separation_na

    # initialize variables
    tmp: pd.DataFrame = pd.DataFrame(index=data.index)
    tmp["U"] = (dependent == 0).astype(float).rename("U")
    # weights
    N0 = (dependent > 0).sum()
    K = N0 / tol**2
    tmp["omega"] = pd.Series(
        np.where(dependent > 0, K, 1), name="omega", index=data.index
    )
    # combine data
    # TODO: avoid create new object?
    tmp = data.join(tmp, how="left", validate="one_to_one", rsuffix=tmp_suffix)
    # TODO: need to ensure that join doesn't create duplicated columns
    # assert not tmp.columns.duplicated().any()

    iteration = 0
    has_converged = False
    while iteration < maxiter:
        iteration += 1
        # regress U on X
        # TODO: check acceleration in ppmlhdfe's implementation: https://github.com/sergiocorreia/ppmlhdfe/blob/master/src/ppmlhdfe_separation_relu.mata#L135
        fitted = feols(fml_separation, data=tmp, weights="omega")
        tmp["Uhat"] = pd.Series(
            data=fitted.predict(), index=fitted._data.index, name="Uhat"
        )
        Uhat = tmp["Uhat"]
        # update when within tolerance of zero
        # need to be more strict below zero to avoid false positives
        within_zero = (Uhat > -0.1 * tol) & (Uhat < tol)
        Uhat.where(~(is_interior | within_zero.fillna(True)), 0, inplace=True)
        if (Uhat >= 0).all():
            # all separated observations have been identified
            has_converged = True
            break
        tmp.loc[~is_interior, "U"] = np.fmax(
            Uhat[~is_interior], 0
        )  # rectified linear unit (ReLU)

    if has_converged:
        separation_na = set(dependent[Uhat > 0].index)
    else:
        warnings.warn(
            "iterative rectivier separation check: maximum number of iterations reached before convergence"
        )

    return separation_na
