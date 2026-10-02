from __future__ import annotations

import warnings

import numpy as np

from pyfixest.core.collinear import collinear_cholesky
from pyfixest.estimation.internals.model_state import CollinearityCheck
from pyfixest.estimation.internals.solvers import GramFactorization
from pyfixest.utils.dev_utils import _find_stack_level


def drop_multicollinear_variables(
    X: np.ndarray,
    names: list[str],
    collin_tol: float,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, CollinearityCheck, GramFactorization]:
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
    weights : np.ndarray or None
        Observation weights, or the current IRLS working weights. The rank
        check uses X' W X; X itself remains in its original units.

    Returns
    -------
    Xd : numpy.ndarray
        The design matrix X after checking for multicollinearity.
    check : CollinearityCheck
        The names of the dropped columns, the mask over X's input columns,
        and the names of the retained columns. The design matrix stays out of
        the value so that a fitted model can publish the check without keeping
        the design alive under `lean=True`.
    factorization : GramFactorization
        Fit-local Gram matrix and upper Cholesky factor for the retained
        columns. These arrays are separate from the published rank metadata.
    """
    design_solver = X if weights is None else X * np.sqrt(weights.reshape(-1, 1))
    tXX = np.ascontiguousarray(design_solver.T @ design_solver, dtype=np.float64)
    id_excl, n_excl, all_removed, upper = collinear_cholesky(tXX, collin_tol)

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
            """,
            UserWarning,
            stacklevel=_find_stack_level(),
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
        keep = ~collin_index
        tXX = np.ascontiguousarray(tXX[np.ix_(keep, keep)])

    check = CollinearityCheck(
        dropped_coef_names=tuple(collin_vars),
        mask=tuple(collin_index.tolist()),
        coefnames=tuple(names_array.tolist()),
    )

    return X, check, GramFactorization(gram=tXX, upper=upper)
