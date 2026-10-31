"""
TD3 residual implementation: critics consume full executed actions; targets use
next_action_base + alpha*tanh(target_u), and actors use action_base + alpha*tanh(mu). The
resfit variant is a separate module.
"""

from __future__ import annotations

from functools import partial
from typing import FrozenSet, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.typing import Batch, Params, PRNGKey
from serl_launcher.utils.train_utils import _unpack


class TD3ResidualIMLEAgent(SACAgent):
    """Residual TD3 with frozen IMLE base (original imle_dex parameterization)."""

    def _mu(
        self,
        observations: Batch,
        rng: Optional[PRNGKey],
        *,
        grad_params: Params,
        train: bool,
    ) -> jax.Array:
        dist = self.forward_policy(
            observations,
            rng=rng,
            grad_params=grad_params,
            train=train,
        )
        return dist.mean()

    def _target_residual(
        self, observations: Batch, rng: PRNGKey
    ) -> jax.Array:
        mu_next = self._mu(
            observations, rng=None, grad_params=self.state.target_params, train=False
        )
        sigma = jnp.asarray(self.config["target_policy_noise"], dtype=mu_next.dtype)
        clip = jnp.asarray(self.config["target_noise_clip"], dtype=mu_next.dtype)
        noise = jax.random.normal(rng, mu_next.shape, dtype=mu_next.dtype) * sigma
        noise = jnp.clip(noise, -clip, clip)
        return mu_next + noise

    def critic_loss_fn(self, batch: Batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        rng, k_noise, k_qt, k_q = jax.random.split(rng, 4)
        alpha = jnp.asarray(self.config["residual_alpha"], dtype=jnp.float32)

        u_next = self._target_residual(batch["next_observations"], k_noise)
        a_res_next = alpha * jnp.tanh(u_next)
        a_base_next = batch["next_action_base"]
        if self.config["fix_gripper"]:
            a_base_next = a_base_next[..., :-1]
            a_res_next = a_res_next[..., :-1]
        a_combined_next = a_base_next + a_res_next

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
        alpha = jnp.asarray(self.config["residual_alpha"], dtype=jnp.float32)

        mu = self._mu(batch["observations"], rng=k_actor, grad_params=params, train=True)
        a_res = alpha * jnp.tanh(mu)
        a_base = batch["action_base"]
        if self.config["fix_gripper"]:
            a_base = a_base[..., :-1]
            a_res = a_res[..., :-1]
        a_combined = a_base + a_res

        predicted_qs = self.forward_critic(
            batch["observations"], a_combined, rng=k_q, grad_params=params
        )
        predicted_q = predicted_qs.mean(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))

        actor_loss = -jnp.mean(predicted_q)

        info = {
            "actor_loss": actor_loss,
            "actor_q": predicted_q.mean(),
            "actor_residual_l1": jnp.mean(jnp.abs(a_res)),
            "actor_residual_l2": jnp.mean(a_res**2),
            "actor_mu_l1": jnp.mean(jnp.abs(mu)),
        }
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
        **kwargs,
    ) -> jnp.ndarray:
        del kwargs
        dist = self.forward_policy(observations, rng=seed, train=False)
        if argmax:
            actions = dist.mean()
        else:
            assert seed is not None, "Stochastic sampling requires a seed"
            actions = dist.sample(seed=seed)
        if self.config["fix_gripper"]:
            actions = jnp.concatenate([actions, jnp.array([0])], axis=-1)
        return actions

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset({"actor", "critic"}),
        **kwargs,
    ) -> Tuple["TD3ResidualIMLEAgent", dict]:
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
