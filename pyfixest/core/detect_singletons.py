import numpy as np
from numpy.typing import NDArray

from pyfixest.core._core_impl import _detect_singletons_rs


def detect_singletons(ids: NDArray[np.integer]) -> NDArray[np.bool_]:
    """
    Detect singleton fixed effects in a dataset.

    This function iterates over the columns of a 2D numpy array representing
    fixed effects to identify singleton fixed effects.
    An observation is considered a singleton if it is the only one in its group
    (fixed effect identifier).

    Parameters
    ----------
    ids : np.ndarray
        A 2D numpy array representing fixed effects, with a shape of (n_samples,
        n_features).
        Elements should be non-negative integers representing fixed effect identifiers.

    Returns
    -------
    numpy.ndarray
        A boolean array of shape (n_samples,), indicating which observations have
        a singleton fixed effect.

    Notes
    -----
    The algorithm iterates over columns to identify fixed effects. After each
    column is processed, it updates the record of non-singleton rows. This approach
    accounts for the possibility that removing an observation in one column can
    lead to the emergence of new singletons in subsequent columns.

    For performance reasons, the input array should be in column-major order.
    Operating on a row-major array can lead to significant performance losses.
    """
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError("Fixed effects must be integers")

    # Convert to uint32 F-contiguous array for optimal performance
    # (matches numba implementation behavior)
    # Using empty((m,n)).T gives F-order (n,m) layout
    n, m = ids.shape
    out: NDArray[np.uint32] = np.empty((m, n), dtype=np.uint32).T
    out[:] = ids
    return _detect_singletons_rs(out)
