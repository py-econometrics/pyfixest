# ----------------------------------------------------------------------
#
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025  PyFixest Authors
#
# This function is a Python/Numba re-implementation of the algorithm
# published by Laurent Berge in the **fixest**
# project (see See the fixest repo
# [here](https:#github.com/lrberge/fixest/blob/a4d1a9bea20aa7ab7ab0e0f1d2047d8097971ad7/src/lm_related.cpp#L130)),
# originally licensed under GPL-3.0.  Laurent Berge granted the maintainers of this
# project an irrevocable, written permission to
# redistribute and re-license the relevant code under the MIT License.
# The full text of that permission is archived at:
#
# docs/THIRD_PARTY_PERMISSIONS.md
# ----------------------------------------------------------------------

import numba as nb
import numpy as np


@nb.njit(parallel=False)
def _find_collinear_variables_nb(
    X: np.ndarray, tol: float = 1e-10
) -> tuple[np.ndarray, int, bool]:
    """
    Detect multicollinear variables.

    Detect multicollinear variables, replicating Laurent Berge's C++ implementation
    from the fixest package. See the fixest repo [here](https://github.com/lrberge/fixest/blob/a4d1a9bea20aa7ab7ab0e0f1d2047d8097971ad7/src/lm_related.cpp#L130)

    Parameters
    ----------
    X : numpy.ndarray
        A symmetric matrix X used to check for multicollinearity.
    tol : float
        The tolerance level for the multicollinearity check.

    Returns
    -------
    - id_excl (numpy.ndarray): A boolean array, where True indicates a collinear
        variable.
    - n_excl (int): The number of collinear variables.
    - all_removed (bool): True if all variables are identified as collinear.
    """
    K = X.shape[1]
    R = np.zeros((K, K))
    id_excl = np.zeros(K, dtype=np.int32)
    n_excl = 0
    min_norm = X[0, 0]

    for j in range(K):
        R_jj = X[j, j]
        for k in range(j):
            if id_excl[k]:
                continue
            R_jj -= R[k, j] * R[k, j]

        if R_jj < tol:
            n_excl += 1
            id_excl[j] = 1

            if n_excl == K:
                all_removed = True
                return id_excl.astype(np.bool_), n_excl, all_removed

            continue

        if min_norm > R_jj:
            min_norm = R_jj

        R_jj = np.sqrt(R_jj)
        R[j, j] = R_jj

        for i in range(j + 1, K):
            value = X[i, j]
            for k in range(j):
                if id_excl[k]:
                    continue
                value -= R[k, i] * R[k, j]
            R[j, i] = value / R_jj

    return id_excl.astype(np.bool_), n_excl, False
