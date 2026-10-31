"""
Separate resfit-aligned TD3 residual implementation; see the corresponding launcher factory.
"""

from __future__ import annotations

from functools import partial
from typing import Any, FrozenSet, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.typing import Batch, Params, PRNGKey
from serl_launcher.utils.train_utils import _unpack

def _action_scale(config: dict) -> jnp.ndarray:
    return jnp.asarray(
        config.get("action_scale", config.get("residual_alpha", 0.1)),
        dtype=jnp.float32,
    )

def _scaled_residual(mu: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    return scale * mu

def _clamp_combined(
    base: jnp.ndarray,
    residual: jnp.ndarray,
    low: jnp.ndarray,
    high: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.clip(base + residual, low, high)

def _obs_for_policy(observations: Batch, action_base: Optional[jnp.ndarray]) -> Batch:
    if action_base is None:
        return observations
    if isinstance(observations, dict):
        return {**observations, "action_base": action_base}
    return observations

class TD3ResidualIMLEAgentResfit(SACAgent):
    """Residual TD3 with frozen IMLE base (resfit-aligned parameterization)."""

    def _action_bounds(self, ref: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        low = jnp.asarray(self.config["action_low"], dtype=ref.dtype)
        high = jnp.asarray(self.config["action_high"], dtype=ref.dtype)
        if self.config["fix_gripper"]:
            low = low[..., :-1]
            high = high[..., :-1]
        return low, high

    def _policy_observations(
        self, observations: Batch, action_base: Optional[jnp.ndarray] = None
    ) -> Batch:
        if not self.config.get("residual_actor_concat_base", False):
            return observations
        if action_base is not None:
            return _obs_for_policy(observations, action_base)
        if isinstance(observations, dict) and "action_base" in observations:
            return observations
        return observations

    def _mu(
        self,
        observations: Batch,
        rng: Optional[PRNGKey],
        *,
        grad_params: Params,
        train: bool,
        action_base: Optional[jnp.ndarray] = None,
    ) -> jax.Array:
        obs_p = self._policy_observations(observations, action_base)
        dist = self.forward_policy(
            obs_p, rng=rng, grad_params=grad_params, train=train
        )
        
        return dist.mode()

    def _target_scaled_residual(
        self, observations: Batch, rng: PRNGKey, *, action_base: jnp.ndarray
    ) -> jax.Array:
        mu_next = self._mu(
            observations,
            rng=None,
            grad_params=self.state.target_params,
            train=False,
            action_base=action_base,
        )
        scale = _action_scale(self.config)
        scaled_mu = _scaled_residual(mu_next, scale)
        sigma = jnp.asarray(self.config["target_policy_noise"], dtype=scaled_mu.dtype)
        clip = jnp.asarray(
            self.config.get("stddev_clip", self.config["target_noise_clip"]),
            dtype=scaled_mu.dtype,
        )
        noise = jax.random.normal(rng, scaled_mu.shape, dtype=scaled_mu.dtype) * sigma
        noise = jnp.clip(noise, -clip, clip)
        return scaled_mu + noise

    def critic_loss_fn(self, batch: Batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        rng, k_noise, k_qt, k_q = jax.random.split(rng, 4)

        a_base_next = batch["next_action_base"]
        if self.config["fix_gripper"]:
            a_base_next = a_base_next[..., :-1]

        res_next = self._target_scaled_residual(
            batch["next_observations"],
            k_noise,
            action_base=a_base_next,
        )
        low, high = self._action_bounds(res_next)
        a_combined_next = _clamp_combined(a_base_next, res_next, low, high)

        target_next_qs = self.forward_target_critic(
            batch["next_observations"], a_combined_next, rng=k_qt
        )

        if self.config["critic_subsample_size"] is not None:
            rng, k_sub = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                k_sub,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]
        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_shape(target_next_min_q, (batch_size,))

        if "discounts" in batch:
            discounts = batch["discounts"].astype(target_next_min_q.dtype)
        else:
            discounts = (
                jnp.asarray(self.config["discount"], dtype=target_next_min_q.dtype)
                * batch["masks"].astype(target_next_min_q.dtype)
            )
        target_q = batch["rewards"] + discounts * target_next_min_q
        chex.assert_shape(target_q, (batch_size,))

        a_taken = batch["actions"][..., :-1] if self.config["fix_gripper"] else batch["actions"]
        predicted_qs = self.forward_critic(
            batch["observations"], a_taken, rng=k_q, grad_params=params
        )
        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])

        td_per_sample_per_head = jnp.abs(predicted_qs - target_qs)
        td_errors = td_per_sample_per_head.mean(axis=0)

        squared = (predicted_qs - target_qs) ** 2
        if "is_weights" in batch:
            w = batch["is_weights"].astype(squared.dtype)
            critic_loss = (squared * w[None, :]).mean()
        else:
            critic_loss = squared.mean()

        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
            "rewards": batch["rewards"].mean(),
            "td_errors_mean": td_errors.mean(),
            "td_errors_max": td_errors.max(),
            "_td_errors": td_errors,
        }
        if "is_weights" in batch:
            info["is_weights_mean"] = batch["is_weights"].mean()
            info["is_weights_max"] = batch["is_weights"].max()
        return critic_loss, info

    def policy_loss_fn(self, batch: Batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        rng, k_actor, k_q = jax.random.split(rng, 3)
        scale = _action_scale(self.config)

        mu = self._mu(
            batch["observations"],
            rng=k_actor,
            grad_params=params,
            train=True,
            action_base=batch["action_base"],
        )
        a_res = _scaled_residual(mu, scale)
        a_base = batch["action_base"]
        if self.config["fix_gripper"]:
            a_base = a_base[..., :-1]
            a_res = a_res[..., :-1]
        low, high = self._action_bounds(a_res)
        a_combined = _clamp_combined(a_base, a_res, low, high)

        predicted_qs = self.forward_critic(
            batch["observations"], a_combined, rng=k_q, grad_params=params
        )
        predicted_q = predicted_qs.mean(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))

        actor_loss_base = -jnp.mean(predicted_q)
        l2_w = float(self.config.get("actor_action_l2_reg_weight", 0.0))
        if l2_w > 0.0:
            l2_penalty = l2_w * jnp.mean(jnp.sum(a_res**2, axis=-1))
            actor_loss = actor_loss_base + l2_penalty
        else:
            l2_penalty = jnp.float32(0.0)
            actor_loss = actor_loss_base

        info = {
            "actor_loss": actor_loss,
            "actor_loss_base": actor_loss_base,
            "actor_q": predicted_q.mean(),
            "actor_residual_l1": jnp.mean(jnp.abs(a_res)),
            "actor_residual_l2": jnp.mean(a_res**2),
            "actor_mu_l1": jnp.mean(jnp.abs(mu)),
        }
        if l2_w > 0.0:
            info["actor_l2_penalty"] = l2_penalty
        return actor_loss, info

    def temperature_loss_fn(self, batch: Batch, params: Params, rng: PRNGKey):
        del batch, params, rng
        return jnp.zeros((), dtype=jnp.float32), {"temperature_loss": jnp.float32(0.0)}

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(
        self,
        observations: Batch,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs: Any,
    ) -> jnp.ndarray:
        del kwargs
        obs_p = self._policy_observations(observations)
        dist = self.forward_policy(obs_p, rng=seed, train=False)
        if argmax:
            mu = dist.mode()
        else:
            assert seed is not None, "Stochastic sampling requires a seed"
            mu = dist.sample(seed=seed)
        scale = _action_scale(self.config)
        residual = _scaled_residual(mu, scale)
        if self.config["fix_gripper"]:
            residual = jnp.concatenate(
                [residual, jnp.zeros(residual.shape[:-1] + (1,), dtype=residual.dtype)],
                axis=-1,
            )
        return residual

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset({"actor", "critic"}),
        **kwargs: Any,
    ) -> Tuple["TD3ResidualIMLEAgentResfit", dict]:
        _drop = ("_online_indices", "_offline_indices", "_indices", "n_step_used")
        if any(k in batch for k in _drop):
            from flax.core import frozen_dict as _fd

            if isinstance(batch, _fd.FrozenDict):
                stripped = {k: v for k, v in batch.unfreeze().items() if k not in _drop}
                batch = _fd.freeze(stripped)
            else:
                batch = {k: v for k, v in batch.items() if k not in _drop}
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        ik = self.config.get("image_keys") or ()
        if ik and ik[0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if (
            "augmentation_function" in self.config
            and self.config["augmentation_function"] is not None
        ):
            batch = self.config["augmentation_function"](batch, aug_rng)

        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]}
        )

        loss_fns = self.loss_fns(batch, **kwargs)

        assert networks_to_update.issubset(loss_fns.keys()), (
            f"Invalid gradient steps: {networks_to_update}"
        )
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        new_state = new_state.replace(rng=rng)

        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info
