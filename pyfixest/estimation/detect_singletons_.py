import numba as nb
import numpy as np
from numba.extending import overload


def _prepare_fixed_effects(ary):
    pass


@overload(_prepare_fixed_effects)
def _ol_preproc_fixed_effects(ary):
    # If array is already an F-array we tolerate
    # any dtype because it saves us a copy
    if ary.layout == "F":
        return lambda ary: ary

    if not isinstance(ary.dtype, nb.types.Integer):
        raise nb.TypingError("Fixed effects must be integers")

    max_nbits = 32
    nbits = min(max_nbits, ary.dtype.bitwidth)
    dtype = nb.types.Integer.from_bitwidth(nbits, signed=False)

    def impl(ary):
        n, m = ary.shape
        out = np.empty((m, n), dtype=dtype).T
        out[:] = ary[:]
        return out

    return impl


@nb.njit
def detect_singletons(ids: np.ndarray) -> np.ndarray:
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
    ids = _prepare_fixed_effects(ids)
    n_samples, n_features = ids.shape

    max_fixef = np.max(ids)
    counts = np.empty(max_fixef + 1, dtype=np.uint32)

    n_non_singletons = n_samples
    non_singletons = np.arange(n_non_singletons, dtype=np.uint32)

    while True:
        n_non_singletons_curr = n_non_singletons

        for j in range(n_features):
            col = ids[:, j]

            counts[:] = 0
            n_singletons = 0
            for i in range(n_non_singletons):
                e = col[non_singletons[i]]
                c = counts[e]
                # Branchless version of:
                #
                # if counts[e] == 1:
                #     n_singletons -= 1
                # elif counts[e] == 0:
                #     n_singletons += 1
                #
                n_singletons += (c == 0) - (c == 1)
                counts[e] += 1

            if not n_singletons:
                continue

            cnt = 0
            for i in range(n_non_singletons):
                e = col[non_singletons[i]]
                if counts[e] != 1:
                    non_singletons[cnt] = non_singletons[i]
                    cnt += 1

            n_non_singletons = cnt

        if n_non_singletons_curr == n_non_singletons:
            break

    is_singleton = np.ones(n_samples, dtype=np.bool_)
    for i in range(n_non_singletons):
        is_singleton[non_singletons[i]] = False

    return is_singleton
