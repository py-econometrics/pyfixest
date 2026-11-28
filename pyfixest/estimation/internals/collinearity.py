from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from pyfixest.core.collinear import find_collinear_variables
from pyfixest.estimation.internals.model_state import CollinearityCheck
from pyfixest.utils.dev_utils import _find_stack_level


def drop_multicollinear_variables(
    X: np.ndarray,
    names: list[str],
    collin_tol: float,
    matrix: Literal["design", "instrument"] = "design",
) -> tuple[np.ndarray, CollinearityCheck]:
    """
    Check for collinear regressors and instruments in the design and instrument matrices.

    Parameters
    ----------
    X : numpy.ndarray
        The design matrix X.
    names : list[str]
        The names of the coefficients.
    collin_tol : float
        The tolerance level for the multicollinearity check.
    matrix : {"design", "instrument"}
        The matrix that `X` holds. It only selects the wording of the warning.

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
    tXX = np.ascontiguousarray(X.T @ X, dtype=np.float64)
    id_excl, n_excl, all_removed = find_collinear_variables(tXX, collin_tol)

    collin_vars: list[str] = []
    collin_index = np.zeros(len(names), dtype=bool)

    if all_removed:
        dropped = repr(names[:5]) + (" ..." if len(names) > 5 else "")
        raise ValueError(
            f"All regressors are collinear at collin_tol={collin_tol:g}: {dropped}. "
            "Check for redundant terms and regressors without variation after "
            "absorbing any fixed effects."
        )

    names_array = np.array(names)
    if n_excl > 0:
        collin_vars = names_array[id_excl].tolist()
        shown = repr(collin_vars[:5])
        if len(collin_vars) > 5:
            shown = shown[:-1] + ", ...]"
        n_collin = len(collin_vars)
        singular = n_collin == 1
        if matrix == "instrument":
            message = (
                f"{n_collin} {'column' if singular else 'columns'} of the "
                f"instrument matrix {'is' if singular else 'are'} collinear and "
                f"{'was' if singular else 'were'} dropped: {shown}."
            )
        else:
            message = (
                f"{n_collin} collinear {'regressor' if singular else 'regressors'} "
                f"{'was' if singular else 'were'} dropped from the design matrix "
                f"and {'gets' if singular else 'get'} no coefficient: {shown}."
            )

        warnings.warn(
            message,
            UserWarning,
            stacklevel=_find_stack_level(),
        )

        X = np.delete(X, id_excl, axis=1)
        names_array = np.delete(names_array, id_excl)
        collin_index = np.asarray(id_excl, dtype=bool)

    check = CollinearityCheck(
        dropped_coef_names=tuple(collin_vars),
        mask=tuple(collin_index.tolist()),
        coefnames=tuple(names_array.tolist()),
    )

    return X, check
