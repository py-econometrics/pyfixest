"""Multicollinearity detection utilities."""

import warnings
from typing import Callable, Optional

import numpy as np


def _drop_multicollinear_variables_chol(
    X: np.ndarray,
    names: list[str],
    collin_tol: float,
    backend_func: Callable,
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """
    Check for multicollinearity via Cholesky decomposition of X'X.

    Parameters
    ----------
    X : numpy.ndarray
        The design matrix X.
    names : list[str]
        The names of the coefficients.
    collin_tol : float
        The tolerance level for the multicollinearity check.
    backend_func: Callable
        Which backend function to use for the multicollinearity check.

    Returns
    -------
    Xd : numpy.ndarray
        The design matrix X after checking for multicollinearity.
    names : list[str]
        The names of the coefficients, excluding those identified as collinear.
    collin_vars : list[str]
        The collinear variables identified during the check.
    collin_index : list[int]
        Indices of the collinear variables.
    """
    # TODO: avoid doing this computation twice, e.g. compute tXXinv here as fixest does

    tXX = X.T @ X
    id_excl, n_excl, all_removed = backend_func(tXX, collin_tol)

    collin_vars = []
    collin_index = []

    if all_removed:
        raise ValueError(
            """
            All variables are collinear. Maybe your model specification introduces multicollinearity? If not, please reach out to the package authors!.
            """
        )

    names_array = np.array(names)
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

        X = np.delete(X, id_excl, axis=1)
        if X.ndim == 2 and X.shape[1] == 0:
            raise ValueError(
                """
                All variables are collinear. Please check your model specification.
                """
            )

        names_array = np.delete(names_array, id_excl)
        collin_index = id_excl.tolist()

    return X, list(names_array), collin_vars, collin_index


def _drop_multicollinear_variables_var(
    X_demeaned: np.ndarray,
    coefnames: list[str],
    X_pre_norms: Optional[np.ndarray],
    collin_tol_var: float,
    has_fixef: bool,
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
    X_pre_norms : numpy.ndarray or None
        Squared column norms of X before demeaning.
    collin_tol_var : float
        Tolerance for the variance ratio check. Variables with
        ||x_demeaned||^2 / ||x||^2 below this are flagged as absorbed.
    has_fixef : bool
        Whether the model has fixed effects.

    Returns
    -------
    X_clean : numpy.ndarray
        The design matrix after removing absorbed variables.
    names_clean : list[str]
        Coefficient names after removing absorbed variables.
    absorbed_names : list[str]
        Names of absorbed variables.
    absorbed_indices : list[int]
        Indices of absorbed variables.
    """
    if (
        X_pre_norms is None
        or collin_tol_var <= 0
        or X_demeaned.shape[1] == 0
        or not has_fixef
    ):
        return X_demeaned, coefnames, [], []

    demeaned_norms = (X_demeaned**2).sum(axis=0)
    ratios = demeaned_norms / X_pre_norms
    absorbed_mask = ratios < collin_tol_var
    if not absorbed_mask.any():
        return X_demeaned, coefnames, [], []

    absorbed_indices = np.where(absorbed_mask)[0]
    absorbed_names = [coefnames[i] for i in absorbed_indices]
    absorbed_set = set(absorbed_indices)
    X_clean = np.delete(X_demeaned, absorbed_indices, axis=1)
    names_clean = [
        n for i, n in enumerate(coefnames) if i not in absorbed_set
    ]

    warnings.warn(
        f"""
        {len(absorbed_names)} variables dropped (absorbed by fixed effects).
        The following variables are dropped: {absorbed_names}.
        """
    )

    return X_clean, names_clean, absorbed_names, absorbed_indices.tolist()
