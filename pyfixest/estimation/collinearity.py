"""Multicollinearity detection utilities."""

import warnings
from typing import Callable, Optional

import numpy as np


def _drop_multicollinear_variables_chol(
    X_demeaned: np.ndarray,
    coefnames: list[str],
    collin_tol: float,
    backend_func: Callable,
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """
    Check for multicollinearity in the design matrices X and Z.

    Parameters
    ----------
    X_demeaned : numpy.ndarray
        A demeaned matrix.
    coefnames : list[str]
        The names of the coefficients.
    collin_tol : float
        The tolerance level for the multicollinearity check.
    backend_func: Callable
        Which backend function to use for the multicollinearity check.

    Returns
    -------
    X_demeaned : numpy.ndarray
        X_demeaned excluding multicollinear variables.
    coefnames : list[str]
        The names of the coefficients, excluding those identified as collinear.
    collin_vars : list[str]
        The collinear variables identified during the check.
    collin_index : numpy.ndarray
        Logical array, where True indicates that the variable is collinear.
    """
    # TODO: avoid doing this computation twice, e.g. compute tXXinv here as fixest does

    tXX = X_demeaned.T @ X_demeaned
    id_excl, n_excl, all_removed = backend_func(tXX, collin_tol)

    collin_vars = []
    collin_index = []

    if all_removed:
        raise ValueError(
            """
            All variables are collinear. Maybe your model specification introduces multicollinearity? If not, please reach out to the package authors!.
            """
        )

    names_array = np.array(coefnames)
    if n_excl > 0:
        collin_vars = names_array[id_excl].tolist()
        if len(collin_vars) > 5:
            indent = "    "
            formatted_collinear_vars = (
                f"\n{indent}" + f"\n{indent}".join(collin_vars[:5]) + f"\n{indent}..."
            )
        else:
            formatted_collinear_vars = str(collin_vars)

        warnings.warn(
            f"""
            {len(collin_vars)} variables dropped due to multicollinearity.
            The following variables are dropped: {formatted_collinear_vars}.
            """
        )

        X_demeaned = np.delete(X_demeaned, id_excl, axis=1)
        if X_demeaned.ndim == 2 and X_demeaned.shape[1] == 0:
            raise ValueError(
                """
                All variables are collinear. Please check your model specification.
                """
            )

        names_array = np.delete(names_array, id_excl)
        collin_index = id_excl.tolist()

    return X_demeaned, list(names_array), collin_vars, collin_index


def _drop_multicollinear_variables_var(
    X_demeaned: np.ndarray,
    coefnames: list[str],
    X_raw_sumsq: Optional[np.ndarray],
    collin_tol_var: float,
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """
    Detect variables absorbed by fixed effects via variance ratio.

    Computes rho_i = ||x_tilde_i||^2 / ||x_i||^2 for each column.
    Columns with rho_i < collin_tol_var are flagged as absorbed.

    Parameters
    ----------
    X_demeaned : numpy.ndarray
        The demeaned design matrix.
    coefnames : list[str]
        The names of the coefficients.
    X_raw_sumsq : numpy.ndarray or None
        Squared column norms of X before demeaning.
    collin_tol_var : float
        Tolerance for the variance ratio check.

    Returns
    -------
    X_demeaned : numpy.ndarray
        The design matrix after removing absorbed variables.
    coefnames : list[str]
        Coefficient names after removing absorbed variables.
    collin_vars : list[str]
        Names of absorbed variables.
    collin_index : list[int]
        Indices of absorbed variables.
    """
    if X_raw_sumsq is None or X_demeaned.shape[1] == 0:
        return X_demeaned, coefnames, [], []

    demeaned_norms = (X_demeaned**2).sum(axis=0)
    ratios = demeaned_norms / X_raw_sumsq
    absorbed_mask = ratios < collin_tol_var
    if not absorbed_mask.any():
        return X_demeaned, coefnames, [], []

    collin_index = np.where(absorbed_mask)[0]
    names_array = np.array(coefnames)
    collin_vars = names_array[collin_index].tolist()

    warnings.warn(
        f"""
        {len(collin_vars)} variables dropped (absorbed by fixed effects).
        The following variables are dropped: {collin_vars}.
        """
    )

    X_demeaned = np.delete(X_demeaned, collin_index, axis=1)
    coefnames = np.delete(names_array, collin_index).tolist()

    return X_demeaned, coefnames, collin_vars, collin_index.tolist()


def drop_multicollinear_variables(
    X_demeaned: np.ndarray,
    coefnames: list[str],
    collin_tol: float,
    backend_func: Callable,
    X_raw_sumsq: Optional[np.ndarray],
    collin_tol_var: float,
    has_fixef: bool,
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """
    Run Cholesky + variance ratio collinearity checks.

    Parameters
    ----------
    X_demeaned : numpy.ndarray
        The demeaned design matrix.
    coefnames : list[str]
        The names of the coefficients.
    collin_tol : float
        Tolerance for the Cholesky multicollinearity check.
    backend_func : Callable
        Backend function for the Cholesky check.
    X_raw_sumsq : numpy.ndarray or None
        Squared column norms of X before demeaning.
    collin_tol_var : float
        Tolerance for the variance ratio check.
    has_fixef : bool
        Whether the model has fixed effects.

    Returns
    -------
    X_demeaned : numpy.ndarray
        The design matrix after removing collinear variables.
    coefnames : list[str]
        Coefficient names after removing collinear variables.
    collin_vars : list[str]
        Names of all removed variables.
    collin_index : list[int]
        Indices of removed variables (relative to the original input columns).
    """
    N = X_demeaned.shape[1]
    collin_vars = []
    collin_index = []

    if N > 0:
        (X_demeaned, coefnames, chol_vars, chol_idx) = (
            _drop_multicollinear_variables_chol(
                X_demeaned, coefnames, collin_tol, backend_func
            )
        )
        collin_vars.extend(chol_vars)
        collin_index.extend(chol_idx)

    if has_fixef and collin_tol_var > 0 and X_demeaned.shape[1] > 0:
        if chol_idx:
            X_raw_sumsq = np.delete(X_raw_sumsq, chol_idx)
        (X_demeaned, coefnames, var_vars, var_idx) = _drop_multicollinear_variables_var(
            X_demeaned, coefnames, X_raw_sumsq, collin_tol_var
        )
        collin_vars.extend(var_vars)
        if var_idx:
            remaining = np.delete(np.arange(N), chol_idx)
            collin_index.extend(remaining[var_idx].tolist())

    return X_demeaned, coefnames, collin_vars, collin_index
