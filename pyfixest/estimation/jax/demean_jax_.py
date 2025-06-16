from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import config


@partial(jax.jit, static_argnames=("n_groups", "tol", "maxiter"))
def _demean_jax_impl(
    x: jnp.ndarray,
    flist: jnp.ndarray,
    weights: jnp.ndarray,
    n_groups: int,
    tol: float,
    maxiter: int = 100_000,
) -> tuple[jnp.ndarray, bool]:
    """JIT-compiled implementation of demeaning."""
    n_factors = flist.shape[1]

    @jax.jit
    def _apply_factor(carry, j):
        """Process a single factor."""
        x = carry
        factor_ids = flist[:, j]
        wx = x * weights[:, None]

        # Compute group weights and weighted sums
        group_weights = jnp.bincount(factor_ids, weights=weights, length=n_groups)
        group_sums = jax.vmap(
            lambda col: jnp.bincount(factor_ids, weights=col, length=n_groups)
        )(wx.T).T

        # Compute and subtract means
        means = group_sums / group_weights[:, None]
        return x - means[factor_ids], None

    @jax.jit
    def _demean_step(x_curr):
        """Single demeaning step for all factors."""
        # Process all factors using scan
        result, _ = jax.lax.scan(_apply_factor, x_curr, jnp.arange(n_factors))
        return result

    @jax.jit
    def _body_fun(state):
        """Body function for while_loop."""
        i, x_curr, x_prev, converged = state
        x_new = _demean_step(x_curr)
        max_diff = jnp.max(jnp.abs(x_new - x_curr))
        has_converged = max_diff < tol
        return i + 1, x_new, x_curr, has_converged

    @jax.jit
    def _cond_fun(state):
        """Condition function for while_loop."""
        i, _, _, converged = state
        return jnp.logical_and(i < maxiter, jnp.logical_not(converged))

    # Run the iteration loop using while_loop
    init_state = (0, x, x - 1.0, False)
    final_i, final_x, _, converged = jax.lax.while_loop(
        _cond_fun, _body_fun, init_state
    )

    return final_x, converged


def demean_jax(
    x: np.ndarray,
    flist: np.ndarray,
    weights: np.ndarray,
    tol: float = 1e-08,
    maxiter: int = 100_000,
) -> tuple[np.ndarray, bool]:
    """Fast and reliable JAX implementation with static shapes."""
    # Enable float64 precision
    config.update("jax_enable_x64", True)

    # Compute n_groups before JIT
    n_groups = int(np.max(flist) + 1)

    # Convert inputs to JAX arrays
    x_jax = jnp.asarray(x, dtype=jnp.float64)
    flist_jax = jnp.asarray(flist, dtype=jnp.int32)
    weights_jax = jnp.asarray(weights, dtype=jnp.float64)

    # Call the JIT-compiled implementation
    result_jax, converged = _demean_jax_impl(
        x_jax, flist_jax, weights_jax, n_groups, tol, maxiter
    )
    return np.array(result_jax), converged
