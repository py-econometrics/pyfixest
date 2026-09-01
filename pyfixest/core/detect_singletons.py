import numpy as np
from numpy.typing import NDArray

from pyfixest.core._core_impl import _detect_singletons_rs


def detect_singletons(
    ids: NDArray[np.integer],
    frequency_weights: NDArray[np.number] | None = None,
) -> NDArray[np.bool_]:
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
    frequency_weights : np.ndarray or None, optional
        Positive frequency counts for the physical rows. When provided, a fixed
        effect is a singleton only when its active frequency total is one.

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

    Examples
    --------
    Each row is an observation, each column a fixed effect. Only the last
    observation is alone in both of its groups.

    ```{python}
    import numpy as np
    from pyfixest.core.detect_singletons import detect_singletons

    ids = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [2, 2]])
    detect_singletons(ids)
    ```

    Dropping a singleton can create new singletons, so detection cascades across
    columns. Here all observations are singletons.

    ```{python}
    detect_singletons(np.array([[0, 0], [0, 1], [1, 2], [2, 2]]))
    ```
    """
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError("Fixed effects must be integers")

    if frequency_weights is not None:
        weights = np.asarray(frequency_weights, dtype=np.float64).reshape(-1)
        if len(weights) != ids.shape[0]:
            raise ValueError(
                "Frequency weights must contain one value per fixed-effect row."
            )
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Frequency weights must be finite and strictly positive.")
        return _detect_frequency_singletons(ids=ids, weights=weights)

    # Convert to uint32 F-contiguous array for optimal performance
    # (matches numba implementation behavior)
    # Using empty((m,n)).T gives F-order (n,m) layout
    n, m = ids.shape
    out: NDArray[np.uint32] = np.empty((m, n), dtype=np.uint32).T
    out[:] = ids
    return _detect_singletons_rs(out)


def _detect_frequency_singletons(
    *,
    ids: NDArray[np.integer],
    weights: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Detect cascading singleton groups using expanded-sample row counts."""
    dropped = np.zeros(ids.shape[0], dtype=bool)

    while True:
        changed = False
        for column in range(ids.shape[1]):
            active_rows = np.flatnonzero(~dropped)
            if active_rows.size == 0:
                return dropped

            _, inverse = np.unique(
                ids[active_rows, column],
                return_inverse=True,
            )
            group_frequency = np.bincount(
                inverse,
                weights=weights[active_rows],
            )
            newly_dropped = group_frequency[inverse] <= 1.0
            if np.any(newly_dropped):
                dropped[active_rows[newly_dropped]] = True
                changed = True

        if not changed:
            return dropped
