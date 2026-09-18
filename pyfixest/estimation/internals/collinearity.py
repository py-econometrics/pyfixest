from __future__ import annotations

import warnings

import numpy as np

from pyfixest.core.collinear import find_collinear_variables
from pyfixest.estimation.internals.model_state import CollinearityCheck


def drop_multicollinear_variables(
    X: np.ndarray,
    names: list[str],
    collin_tol: float,
) -> tuple[np.ndarray, CollinearityCheck]:
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
    check : CollinearityCheck
        The names of the dropped columns, the mask over X's input columns,
        and the names of the retained columns. The design matrix stays out of
        the value so that a fitted model can publish the check without keeping
        the design alive under `lean=True`.
    """
    # TODO: avoid doing this computation twice, e.g. compute tXXinv here as fixest does

    tXX = np.ascontiguousarray(X.T @ X, dtype=np.float64)
    id_excl, n_excl, all_removed = find_collinear_variables(tXX, collin_tol)

    collin_vars: list[str] = []
    collin_index = np.zeros(len(names), dtype=bool)

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
        collin_index = np.asarray(id_excl, dtype=bool)

    check = CollinearityCheck(
        dropped=tuple(collin_vars),
        mask=tuple(collin_index.tolist()),
        coefnames=tuple(names_array.tolist()),
    )

    return X, check
