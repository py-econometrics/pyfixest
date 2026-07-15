from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from pyfixest.core.collinear import find_collinear_variables


@dataclass(frozen=True, slots=True)
class CollinearityResult:
    """Outcome of a collinearity drop operation.

    Attributes
    ----------
    X : np.ndarray
        The design matrix with collinear columns removed.
    names : list[str]
        Coefficient names corresponding to the columns of ``X``.
    collin_vars : list[str]
        Names of the variables that were dropped. Empty when nothing
        was collinear.
    collin_index : list[bool]
        Boolean mask over the *input* X's columns: ``True`` marks a
        dropped column.  Empty list when no columns were dropped.
    """

    X: np.ndarray
    names: list[str]
    collin_vars: list[str]
    collin_index: list[bool]


class CollinearityHandler:
    """Detect and drop multicollinear variables from a design matrix.

    Wraps :func:`drop_multicollinear_variables` and returns a
    :class:`CollinearityResult` so callers don't have to unpack a
    four-tuple and track the pieces individually.

    Parameters
    ----------
    collin_tol : float
        Tolerance for the multicollinearity check.
    """

    def __init__(self, collin_tol: float):
        self._collin_tol = collin_tol

    def drop(self, X: np.ndarray, names: list[str]) -> CollinearityResult:
        """Drop collinear columns from ``X`` and return the result."""
        Xd, names_d, collin_vars, collin_index = drop_multicollinear_variables(
            X, names, self._collin_tol
        )
        return CollinearityResult(
            X=Xd,
            names=names_d,
            collin_vars=collin_vars,
            collin_index=collin_index,
        )


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
