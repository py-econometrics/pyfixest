"""Detect and remove multicollinear regressors."""

from __future__ import annotations

import warnings

import numpy as np

from pyfixest.core.collinear import find_collinear_variables


def drop_multicollinear_variables(
    X: np.ndarray,
    names: list[str],
    collin_tol: float,
) -> tuple[np.ndarray, list[str], list[str], list[bool]]:
    """
    Check for multicollinearity in the design matrices X and Z.

    Parameters
    ----------
    X : numpy.ndarray
        The design matrix X.
    names : list[str]
        The names of the coefficients.
    collin_tol : float
        The tolerance level for the multicollinearity check.

    Returns
    -------
    Xd : numpy.ndarray
        The design matrix X after checking for multicollinearity.
    names : list[str]
        The names of the coefficients, excluding those identified as collinear.
    collin_vars : list[str]
        The collinear variables identified during the check.
    collin_index : list[bool]
        Boolean mask over X's input columns: True marks a dropped column.
        Empty list when no columns were dropped.
    """
    # TODO: avoid doing this computation twice, e.g. compute tXXinv here as fixest does

    tXX = np.ascontiguousarray(X.T @ X, dtype=np.float64)
    id_excl, n_excl, all_removed = find_collinear_variables(tXX, collin_tol)

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
