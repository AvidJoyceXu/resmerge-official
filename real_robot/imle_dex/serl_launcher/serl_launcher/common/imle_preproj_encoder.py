"""Encode observations['imle_preproj'], supplied by the frozen PyTorch IMLE encoder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn


class ImlePreprojEncoder(nn.Module):
    """
    Map flattened IMLE features through an MLP. If concat_action_base_key is set, concatenate
    that observation after the MLP and apply another projection.
    """

    feat_key: str = "imle_preproj"
    mlp_hidden_dim: int = 512
    concat_action_base_key: Optional[str] = None

    @nn.compact
    def __call__(
        self,
        observations,
        train: bool = False,
        stop_gradient: bool = False,
        is_encoded: bool = False,
    ):
        del train, is_encoded
        
        
        
        if isinstance(observations, Mapping):
            x = observations[self.feat_key]
        else:
            x = observations
        x = jnp.asarray(x, dtype=jnp.float32)
        if x.ndim > 2:
            x = jnp.reshape(x, (x.shape[0], -1))
        elif x.ndim == 1:
            x = x[jnp.newaxis, :]
        if stop_gradient:
            x = jax.lax.stop_gradient(x)
        x = nn.Dense(self.mlp_hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.LayerNorm()(x)
        x = jnp.tanh(x)

        key = self.concat_action_base_key
        if key is not None and isinstance(observations, Mapping) and key in observations:
            ab = jnp.asarray(observations[key], dtype=jnp.float32)
            if ab.ndim == 1:
                ab = ab[jnp.newaxis, :]
            x = jnp.concatenate([x, ab], axis=-1)
            x = nn.Dense(self.mlp_hidden_dim, kernel_init=nn.initializers.xavier_uniform())(
                x
            )
            x = nn.LayerNorm()(x)
            x = jnp.tanh(x)
        return x
