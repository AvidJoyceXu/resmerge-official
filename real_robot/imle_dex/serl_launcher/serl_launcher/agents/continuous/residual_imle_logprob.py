"""
Log-density correction for residual r = alpha*tanh(u), where u is Gaussian. Subtract the sum of
log absolute Jacobian terms. For alpha=1 this matches the standard SAC tanh transform;
otherwise retain the additional log|alpha| scale term.
"""

from __future__ import annotations

import jax.numpy as jnp


def residual_tanh_squash_log_prob(
    log_p_u: jnp.ndarray, u: jnp.ndarray, alpha
) -> jnp.ndarray:
    """
    Compute log p_r(r) = log p_u(u) - sum(log|alpha*(1-tanh(u)^2)|). log_p_u is the batch
    scalar log probability of a diagonal multivariate Gaussian. With alpha=1 this matches the
    unscaled tanh bijector.
    """
    alpha = jnp.asarray(alpha, dtype=u.dtype)
    t = jnp.tanh(u)
    per_dim = jnp.maximum(
        alpha * (1.0 - t * t),
        jnp.asarray(1e-8, dtype=u.dtype),
    )
    log_det = jnp.sum(jnp.log(per_dim), axis=-1)
    return log_p_u - log_det
