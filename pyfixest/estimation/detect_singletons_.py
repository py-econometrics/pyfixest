from functools import partial

import jax
import jax.numpy as jnp
import numba as nb
import numpy as np
from jax import lax
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


@partial(jax.jit, static_argnames=("n_samples", "n_features", "max_fixef"))
def _process_features_jax(
    ids, non_singletons, n_non_singletons, n_samples, n_features, max_fixef
):
    """JIT-compiled inner loop for processing features with static shapes."""

    def process_feature(carry, j):
        non_singletons, n_non_singletons = carry
        col = ids[:, j]

        # Initialize counts array
        counts = jnp.zeros(max_fixef + 1, dtype=jnp.int32)

        # Count occurrences and track singletons
        def count_loop(i, state):
            counts, n_singletons = state
            e = col[non_singletons[i]]
            c = counts[e]
            # Exactly match Numba: n_singletons += (c == 0) - (c == 1)
            n_singletons = n_singletons + (c == 0) - (c == 1)
            counts = counts.at[e].add(1)
            return (counts, n_singletons)

        counts, n_singletons = lax.fori_loop(
            0, n_non_singletons, count_loop, (counts, 0)
        )

        # Early return if no singletons found
        def no_singletons(_):
            return (non_singletons, n_non_singletons)

        # Update non_singletons if singletons found
        def update_singletons(_):
            def update_loop(i, state):
                new_non_singletons, cnt = state
                e = col[non_singletons[i]]
                keep = counts[e] != 1
                # Exactly match Numba's update logic
                new_non_singletons = lax.cond(
                    keep,
                    lambda x: x[0].at[x[1]].set(non_singletons[i]),
                    lambda x: x[0],
                    (new_non_singletons, cnt),
                )
                return (new_non_singletons, cnt + keep)

            new_non_singletons = jnp.zeros_like(non_singletons)
            new_non_singletons, new_cnt = lax.fori_loop(
                0, n_non_singletons, update_loop, (new_non_singletons, 0)
            )
            return (new_non_singletons, new_cnt)

        return lax.cond(n_singletons == 0, no_singletons, update_singletons, None), None

    return lax.scan(
        process_feature, (non_singletons, n_non_singletons), jnp.arange(n_features)
    )[0]


def detect_singletons_jax(ids: np.ndarray) -> np.ndarray:
    """
    JAX implementation of singleton detection in fixed effects.

    Parameters
    ----------
    ids : numpy.ndarray
        A 2D numpy array representing fixed effects, with shape (n_samples, n_features).
        Elements should be non-negative integers representing fixed effect identifiers.

    Returns
    -------
    numpy.ndarray
        A boolean array of shape (n_samples,), indicating which observations have
        a singleton fixed effect.
    """
    # Get dimensions and max_fixef before JIT
    n_samples, n_features = ids.shape
    max_fixef = int(np.max(ids))  # Use numpy.max instead of jax.numpy.max

    # Convert input to JAX array
    ids = jnp.array(ids, dtype=jnp.int32)

    # Initialize with all indices as non-singletons
    init_non_singletons = jnp.arange(n_samples)
    init_n_non_singletons = n_samples

    @partial(jax.jit, static_argnames=("n_samples", "n_features", "max_fixef"))
    def _singleton_detection_loop(
        ids, non_singletons, n_non_singletons, n_samples, n_features, max_fixef
    ):
        def cond_fun(state):
            prev_n, curr_carry = state
            return prev_n != curr_carry[1]

        def body_fun(state):
            prev_n, curr_carry = state
            new_carry = _process_features_jax(
                ids, curr_carry[0], curr_carry[1], n_samples, n_features, max_fixef
            )
            return (curr_carry[1], new_carry)

        init_state = (n_samples + 1, (non_singletons, n_non_singletons))
        final_state = lax.while_loop(cond_fun, body_fun, init_state)
        return final_state[1]

    # Run iterations until convergence
    final_non_singletons, final_n = _singleton_detection_loop(
        ids,
        init_non_singletons,
        init_n_non_singletons,
        n_samples,
        n_features,
        max_fixef,
    )

    # Create final boolean mask
    is_singleton = jnp.ones(n_samples, dtype=jnp.bool_)

    @jax.jit
    def _mark_non_singletons(is_singleton, final_non_singletons, final_n):
        def mark_non_singleton(i, acc):
            return acc.at[final_non_singletons[i]].set(False)

        return lax.fori_loop(0, final_n, mark_non_singleton, is_singleton)

    is_singleton = _mark_non_singletons(is_singleton, final_non_singletons, final_n)

    return np.array(is_singleton)
