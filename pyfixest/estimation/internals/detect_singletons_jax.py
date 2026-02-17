from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


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

        counts, n_singletons = jax.lax.fori_loop(
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
                new_non_singletons = jax.lax.cond(
                    keep,
                    lambda x: x[0].at[x[1]].set(non_singletons[i]),
                    lambda x: x[0],
                    (new_non_singletons, cnt),
                )
                return (new_non_singletons, cnt + keep)

            new_non_singletons = jnp.zeros_like(non_singletons)
            new_non_singletons, new_cnt = jax.lax.fori_loop(
                0, n_non_singletons, update_loop, (new_non_singletons, 0)
            )
            return (new_non_singletons, new_cnt)

        return jax.lax.cond(
            n_singletons == 0, no_singletons, update_singletons, None
        ), None

    return jax.lax.scan(
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
            _, curr_carry = state
            new_carry = _process_features_jax(
                ids, curr_carry[0], curr_carry[1], n_samples, n_features, max_fixef
            )
            return (curr_carry[1], new_carry)

        init_state = (n_samples + 1, (non_singletons, n_non_singletons))
        final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
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

        return jax.lax.fori_loop(0, final_n, mark_non_singleton, is_singleton)

    is_singleton = _mark_non_singletons(is_singleton, final_non_singletons, final_n)

    return np.array(is_singleton)
