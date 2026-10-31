# Modified by the ResMerge contributors, 2026-10-31; see real_robot/THIRD_PARTY.md.
"""
Residual SAC with frozen IMLE: critics use full executed actions; target/actor actions add
alpha*tanh(u) to base actions. The Gaussian policy emits unsquashed u. Entropy uses the
Jacobian-corrected residual density. Replay stores action_base and next_action_base separately
from observations.
"""

from __future__ import annotations

import chex
import flax
import jax
import jax.numpy as jnp

from serl_launcher.agents.continuous.residual_imle_logprob import (
    residual_tanh_squash_log_prob,
)
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.typing import Batch, Params, PRNGKey

class SACResidualIMLEAgent(SACAgent):
    """Requires full executed actions and action_base/next_action_base in each batch."""

    def _maybe_normalize_logp(self, log_p: jax.Array, action_dim: int) -> jax.Array:
        """Normalize joint log-prob by action dimension to reduce high-dim scale effects."""
        if bool(self.config.get("entropy_normalize_by_action_dim", True)):
            return log_p / jnp.asarray(max(int(action_dim), 1), dtype=log_p.dtype)
        return log_p

    def _scaled_target_entropy(self, action_dim: int, dtype) -> jax.Array:
        target_entropy = jnp.asarray(self.config["target_entropy"], dtype=dtype)
        if bool(self.config.get("entropy_normalize_by_action_dim", True)):
            target_entropy = target_entropy / jnp.asarray(
                max(int(action_dim), 1), dtype=dtype
            )
        return target_entropy

    def _next_residual_sample(
        self, observations: Batch, rng: PRNGKey, *, grad_params: Params
    ) -> tuple[jax.Array, jax.Array]:
        """
        Sample Gaussian u and its log probability from the target policy at next_observations.
        """
        k_fwd, k_smp = jax.random.split(rng)
        next_action_distributions = self.forward_policy(
            observations,
            rng=k_fwd,
            grad_params=grad_params,
            train=False,
        )
        u_next, log_p = next_action_distributions.sample_and_log_prob(seed=k_smp)
        alpha = jnp.asarray(self.config["residual_alpha"], dtype=u_next.dtype)
        log_p = residual_tanh_squash_log_prob(log_p, u_next, alpha)
        chex.assert_shape(log_p, (u_next.shape[0],))
        return u_next, log_p

    def critic_loss_fn(self, batch: Batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
        alpha = jnp.asarray(self.config["residual_alpha"], dtype=jnp.float32)

        u_next, log_p_next = self._next_residual_sample(
            batch["next_observations"], k1, grad_params=self.state.target_params
        )
        a_res_next = alpha * jnp.tanh(u_next)
        a_base_next = batch["next_action_base"]
        if self.config["fix_gripper"]:
            a_base_next = a_base_next[..., :-1]
            a_res_next = a_res_next[..., :-1]
        a_combined_next = a_base_next + a_res_next

        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            a_combined_next,
            rng=k2,
        )

        if self.config["critic_subsample_size"] is not None:
            subsample_idcs = jax.random.randint(
                k3,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_shape(target_next_min_q, (batch_size,))

        target_next_value = target_next_min_q
        if self.config["backup_entropy"]:
            action_dim = u_next.shape[-1]
            log_p_next = self._maybe_normalize_logp(log_p_next, action_dim)
            temperature = self.forward_temperature()
            target_next_value = target_next_value - temperature * log_p_next

        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_value
        )
        chex.assert_shape(target_q, (batch_size,))

        a_taken = batch["actions"][..., :-1] if self.config["fix_gripper"] else batch["actions"]
        predicted_qs = self.forward_critic(
            batch["observations"], a_taken, rng=k4, grad_params=params
        )

        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
            "rewards": batch["rewards"].mean(),
        }
        return critic_loss, info

    def policy_loss_fn(self, batch: Batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()
        alpha = jnp.asarray(self.config["residual_alpha"], dtype=jnp.float32)

        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        action_distributions = self.forward_policy(
            batch["observations"], rng=policy_rng, grad_params=params
        )
        u, log_p = action_distributions.sample_and_log_prob(seed=sample_rng)
        log_p = residual_tanh_squash_log_prob(log_p, u, alpha)
        log_p = self._maybe_normalize_logp(log_p, u.shape[-1])

        a_res = alpha * jnp.tanh(u)
        a_base = batch["action_base"]
        if self.config["fix_gripper"]:
            a_base = a_base[..., :-1]
            a_res = a_res[..., :-1]
        a_combined = a_base + a_res

        predicted_qs = self.forward_critic(
            batch["observations"],
            a_combined,
            rng=critic_rng,
            grad_params=params,
        )
        predicted_q = predicted_qs.min(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_p, (batch_size,))

        actor_objective = predicted_q - temperature * log_p
        actor_loss = -jnp.mean(actor_objective)

        info = {
            "actor_loss": actor_loss,
            "temperature": temperature,
            "entropy": -log_p.mean(),
            
            
            "actor_residual_l1": jnp.mean(jnp.abs(a_res)),
            "actor_residual_l2": jnp.mean(a_res**2),
            "actor_residual_max": jnp.max(jnp.abs(a_res)),
        }
        return actor_loss, info

    def temperature_loss_fn(self, batch: Batch, params: Params, rng: PRNGKey):
        rng, k1 = jax.random.split(rng)
        u_next, log_p_next = self._next_residual_sample(
            batch["next_observations"], k1, grad_params=self.state.target_params
        )
        action_dim = u_next.shape[-1]
        log_p_next = self._maybe_normalize_logp(log_p_next, action_dim)
        entropy = -log_p_next.mean()
        target_entropy = self._scaled_target_entropy(action_dim, dtype=entropy.dtype)
        temperature_loss = self.state.apply_fn(
            {"params": params},
            lhs=entropy,
            rhs=target_entropy,
            name="temperature",
        )
        return temperature_loss, {"temperature_loss": temperature_loss}
