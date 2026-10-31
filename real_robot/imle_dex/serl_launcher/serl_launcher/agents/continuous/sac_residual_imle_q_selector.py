"""
Residual IMLE SAC with candidate Double-DQN selector. selector_q shares the parameter
container, not the actor network. Online next-action argmax is evaluated by the target network
without heuristic bootstrapping. Feature/candidate inputs produce (B, K) Q values.
"""

from __future__ import annotations

from functools import partial
from typing import FrozenSet, Optional, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.agents.continuous.sac_residual_imle import SACResidualIMLEAgent
from serl_launcher.common.common import JaxRLTrainState, ModuleDict
from serl_launcher.common.imle_preproj_encoder import ImlePreprojEncoder
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.networks.actor_critic_nets import CandidateSelectorCritic
from serl_launcher.networks.mlp import MLP


class SACResidualIMLEQSelectorAgent(SACResidualIMLEAgent):
    """Add selector_q and candidate DDQN loss to SACResidualIMLEAgent."""

    def forward_selector_q(
        self,
        obs_feat: jnp.ndarray,
        cands: jnp.ndarray,
        *,
        train: bool = False,
        params: Optional[dict] = None,
    ) -> jnp.ndarray:
        """Accept obs_feat as (D,) or (B,D) and candidates as (K,A) or (B,K,A)."""
        p = params if params is not None else self.state.params
        x = obs_feat
        if x.ndim == 1:
            x = x[None, ...]
        a = cands
        if a.ndim == 2:
            a = a[None, ...]
        obs_input = {self.config["q_selector_feat_key"]: x}
        return self.state.apply_fn(
            {"params": p},
            obs_input,
            a,
            train=train,
            name="selector_q",
        )

    def selector_q_loss_fn(self, batch: Batch, params, rng: PRNGKey):
        del rng
        feat_key = self.config["q_selector_feat_key"]
        gamma = jnp.asarray(self.config["q_selector_discount"], dtype=jnp.float32)
        obs = batch["observations"][feat_key]
        next_obs = batch["next_observations"][feat_key]
        chosen_cand = batch["selector_chosen_cand"]
        next_cands = batch["selector_next_cands"]
        r = batch["rewards"]
        if r.ndim > 1:
            r = r.reshape(r.shape[0])
        m = batch["masks"]
        if m.ndim > 1:
            m = m.reshape(m.shape[0])

        b = obs.shape[0]
        ar = jnp.arange(b)
        apply_fn = self.state.apply_fn
        target_params = self.state.target_params

        variables = {"params": params}
        obs_in = {feat_key: obs}
        next_obs_in = {feat_key: next_obs}
        q_chosen = apply_fn(
            variables, obs_in, chosen_cand[:, None, :], train=True, name="selector_q"
        )
        q_sa = q_chosen[:, 0]

        q_on_next = apply_fn(variables, next_obs_in, next_cands, train=False, name="selector_q")
        variables_t = {"params": target_params}
        q_tgt_next = apply_fn(variables_t, next_obs_in, next_cands, train=False, name="selector_q")

        a_star = jnp.argmax(q_on_next, axis=-1)
        q_tp1 = q_tgt_next[ar, a_star]
        y = r + gamma * m * jax.lax.stop_gradient(q_tp1)
        loss = jnp.mean((q_sa - y) ** 2)

        info = {
            "q_selector_loss": loss,
            "q_selector_q_sa_mean": jnp.mean(q_sa),
            "q_selector_target_mean": jnp.mean(y),
        }
        return loss, info

    def loss_fns(self, batch, **kwargs):
        base = super().loss_fns(batch, **kwargs)
        base["selector_q"] = partial(self.selector_q_loss_fn, batch)
        return base

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset(
            {"actor", "critic", "temperature", "selector_q"}
        ),
        **kwargs,
    ) -> Tuple["SACResidualIMLEQSelectorAgent", dict]:
        from serl_launcher.utils.train_utils import _unpack

        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        ik = self.config["image_keys"]
        if ik and ik[0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if (
            "augmentation_function" in self.config.keys()
            and self.config["augmentation_function"] is not None
        ):
            batch = self.config["augmentation_function"](batch, aug_rng)

        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]}
        )

        loss_fns = self.loss_fns(batch, **kwargs)

        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid gradient steps: {networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        if "critic" in networks_to_update or "selector_q" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        new_state = new_state.replace(rng=rng)

        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    @classmethod
    def create_preproj(
        cls,
        rng: PRNGKey,
        observations: Batch,
        actions: jnp.ndarray,
        *,
        residual_alpha: float,
        q_num_candidates: int,
        q_selector_action_dim: int = -1,
        q_selector_feat_key: str = "imle_preproj",
        q_selector_discount: float = 0.97,
        mlp_hidden_dim: int = 512,
        image_keys: tuple = (),
        reward_bias: float = 0.0,
        target_entropy=None,
        alpha_max: float = 1.0,
        entropy_normalize_by_action_dim: bool = True,
        discount: float = 0.97,
        soft_target_update_rate: float = 0.005,
        critic_ensemble_size: int = 5,
        selector_q_optimizer_kwargs: Optional[dict] = None,
        selector_q_mlp_hidden_dims: Tuple[int, ...] = (256, 256),
        selector_q_preproj_encoder_dim: int = 512,
        actor_learning_rate: float = 3e-5,
        critic_learning_rate: float = 3e-4,
        temperature_learning_rate: float = 3e-4,
    ):
        """
        Construct the IMLE pre-projection SAC backbone with an additional selector_q module.
        """
        from functools import partial

        from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize

        if selector_q_optimizer_kwargs is None:
            selector_q_optimizer_kwargs = {"learning_rate": 3e-4}

        enc_shared = ImlePreprojEncoder(
            feat_key=q_selector_feat_key, mlp_hidden_dim=mlp_hidden_dim
        )

        policy_network_kwargs = dict(
            activations=nn.tanh,
            use_layer_norm=True,
            hidden_dims=[256, 256],
            activate_final=True,
        )
        critic_network_kwargs = dict(
            activations=nn.tanh,
            use_layer_norm=True,
            hidden_dims=[256, 256],
            activate_final=True,
        )

        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(Critic, encoder=enc_shared, network=critic_backbone)(
            name="critic"
        )

        policy_def = Policy(
            encoder=enc_shared,
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            tanh_squash_distribution=False,
            std_parameterization="exp",
            std_min=1e-5,
            std_max=5,
            stop_gradient_on_encoder_output=True,
            zero_init_mean_head=True,
            name="actor",
        )

        from serl_launcher.networks.lagrange import GeqLagrangeMultiplier

        temperature_def = GeqLagrangeMultiplier(
            init_value=1e-2,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        selector_state_encoder = ImlePreprojEncoder(
            feat_key=q_selector_feat_key,
            mlp_hidden_dim=int(selector_q_preproj_encoder_dim),
        )
        selector_q_def = CandidateSelectorCritic(
            encoder=selector_state_encoder,
            network=MLP(
                hidden_dims=selector_q_mlp_hidden_dims,
                activations=nn.swish,
                use_layer_norm=True,
            ),
            init_final=0.01,
            name="selector_q",
        )

        networks = {
            "actor": policy_def,
            "critic": critic_def,
            "temperature": temperature_def,
            "selector_q": selector_q_def,
        }
        model_def = ModuleDict(networks)

        txs = {
            "actor": make_optimizer(learning_rate=float(actor_learning_rate)),
            "critic": make_optimizer(learning_rate=float(critic_learning_rate)),
            "temperature": make_optimizer(learning_rate=float(temperature_learning_rate)),
            "selector_q": make_optimizer(**selector_q_optimizer_kwargs),
        }

        sample_obs_jax = jax.tree_util.tree_map(jnp.asarray, observations)
        sample_act_jax = jnp.asarray(actions)
        if sample_act_jax.ndim == 1:
            sample_act_jax = sample_act_jax[jnp.newaxis, ...]

        feat = sample_obs_jax[q_selector_feat_key]
        if feat.ndim == 1:
            feat_b = feat[jnp.newaxis, ...]
        else:
            feat_b = feat[:1]
        k = max(1, int(q_num_candidates))
        adim = (
            int(q_selector_action_dim)
            if int(q_selector_action_dim) > 0
            else int(sample_act_jax.shape[-1])
        )
        c0 = sample_act_jax[:1, :]
        if c0.shape[-1] != adim:
            c0 = jnp.zeros((1, adim), dtype=sample_act_jax.dtype)
        cands_init = jnp.broadcast_to(c0[:, jnp.newaxis, :], (1, k, adim))

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[sample_obs_jax],
            critic=[sample_obs_jax, sample_act_jax],
            temperature=[],
            selector_q=[feat_b, cands_init],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        action_dim = actions.shape[-1]
        if target_entropy is None:
            target_entropy = -action_dim / 2

        cfg = dict(
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=None,
            discount=discount,
            soft_target_update_rate=float(soft_target_update_rate),
            target_entropy=target_entropy,
            alpha_max=float(alpha_max) if alpha_max is not None else None,
            backup_entropy=True,
            entropy_normalize_by_action_dim=bool(entropy_normalize_by_action_dim),
            image_keys=image_keys,
            reward_bias=reward_bias,
            augmentation_function=None,
            fix_gripper=False,
            residual_alpha=float(residual_alpha),
            q_selector_feat_key=q_selector_feat_key,
            q_selector_discount=float(q_selector_discount),
        )

        return cls(state=state, config=cfg)
