"""
DrQ residual SAC with candidate selector_q. Pixel crops affect the main networks; selector_q
uses uncropped IMLE features/candidates. All parameters are synchronized together.
"""

from __future__ import annotations

from functools import partial
from typing import FrozenSet, Iterable, Optional, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.agents.continuous.drq_residual_imle import DrQResidualIMLEAgent
from serl_launcher.common.common import JaxRLTrainState, ModuleDict
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.imle_preproj_encoder import ImlePreprojEncoder
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, PRNGKey
from serl_launcher.networks.actor_critic_nets import CandidateSelectorCritic, Critic, Policy, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack


class DrQResidualIMLEQSelectorAgent(DrQResidualIMLEAgent):
    """Add selector_q and candidate DDQN loss to DrQResidualIMLEAgent."""

    def forward_selector_q(
        self,
        obs_feat: jnp.ndarray,
        cands: jnp.ndarray,
        *,
        train: bool = False,
        params: Optional[dict] = None,
    ) -> jnp.ndarray:
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
    ) -> Tuple["DrQResidualIMLEQSelectorAgent", dict]:
        batch_size = batch["rewards"].shape[0]
        ik = self.config["image_keys"]
        if ik and ik[0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng = self.state.rng
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = self.data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = self.data_augmentation_fn(next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )
        new_state = self.state.replace(rng=rng)
        new_agent = self.replace(state=new_state)

        batch = batch.copy(
            add_or_replace={
                "rewards": batch["rewards"] + new_agent.config["reward_bias"]
            }
        )
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        loss_fns = new_agent.loss_fns(batch, **kwargs)

        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid gradient steps: {networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = new_agent.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        if "critic" in networks_to_update or "selector_q" in networks_to_update:
            new_state = new_state.target_update(new_agent.config["soft_target_update_rate"])

        new_state = new_state.replace(rng=rng)

        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return new_agent.replace(state=new_state), info

    @classmethod
    def create_drq_pixel(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        *,
        residual_alpha: float,
        q_num_candidates: int,
        q_selector_action_dim: int = -1,
        q_selector_feat_key: str = "imle_preproj",
        q_selector_discount: float = 0.97,
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = True,
        critic_network_kwargs: Optional[dict] = None,
        policy_network_kwargs: Optional[dict] = None,
        policy_kwargs: Optional[dict] = None,
        critic_ensemble_size: int = 5,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1e-2,
        image_keys: Iterable[str] = ("side_policy_256",),
        q_selector_optimizer_kwargs: Optional[dict] = None,
        selector_q_mlp_hidden_dims: Tuple[int, ...] = (256, 256),
        selector_q_preproj_encoder_dim: int = 512,
        discount: float = 0.97,
        soft_target_update_rate: float = 0.005,
        target_entropy=None,
        alpha_max: float = 1.0,
        entropy_normalize_by_action_dim: bool = True,
        reward_bias: float = 0.0,
        backup_entropy: bool = True,
        fix_gripper: bool = False,
        **kwargs,
    ):
        """
        Construct the DrQ pixel backbone and selector_q. observations must contain
        q_selector_feat_key for candidate-Q initialization and training.
        """
        if critic_network_kwargs is None:
            critic_network_kwargs = {
                "activations": nn.tanh,
                "use_layer_norm": True,
                "hidden_dims": [256, 256],
            }
        if policy_network_kwargs is None:
            policy_network_kwargs = {
                "activations": nn.tanh,
                "use_layer_norm": True,
                "hidden_dims": [256, 256],
            }
        if policy_kwargs is None:
            policy_kwargs = {
                "tanh_squash_distribution": False,
                "std_parameterization": "exp",
                "std_min": 1e-5,
                "std_max": 5,
                "zero_init_mean_head": True,
            }
        if q_selector_optimizer_kwargs is None:
            q_selector_optimizer_kwargs = {"learning_rate": 3e-4}

        critic_network_kwargs = {**critic_network_kwargs, "activate_final": True}
        policy_network_kwargs = {**policy_network_kwargs, "activate_final": True}

        if encoder_type == "small":
            from serl_launcher.vision.small_encoders import SmallEncoder

            encoders = {
                image_key: SmallEncoder(
                    features=(32, 64, 128, 256),
                    kernel_sizes=(3, 3, 3, 3),
                    strides=(2, 2, 2, 2),
                    padding="VALID",
                    pool_method="avg",
                    bottleneck_dim=256,
                    spatial_block_size=8,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        encoders_dc = {
            "critic": encoder_def,
            "actor": encoder_def,
        }

        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders_dc["critic"], network=critic_backbone
        )(name="critic")

        policy_def = Policy(
            encoder=encoders_dc["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1] - (1 if fix_gripper else 0),
            **policy_kwargs,
            name="actor",
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        selector_q_def = CandidateSelectorCritic(
            encoder=ImlePreprojEncoder(
                feat_key=q_selector_feat_key,
                mlp_hidden_dim=int(selector_q_preproj_encoder_dim),
            ),
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

        _aok = kwargs.get("actor_optimizer_kwargs") or {}
        _cok = kwargs.get("critic_optimizer_kwargs") or {}
        _tok = kwargs.get("temperature_optimizer_kwargs") or {}
        actor_lr = _aok.get("learning_rate", 3e-4)
        critic_lr = _cok.get("learning_rate", 3e-4)
        temp_lr = _tok.get("learning_rate", 3e-4)

        txs = {
            "actor": make_optimizer(learning_rate=actor_lr),
            "critic": make_optimizer(learning_rate=critic_lr),
            "temperature": make_optimizer(learning_rate=temp_lr),
            "selector_q": make_optimizer(**q_selector_optimizer_kwargs),
        }

        sample_obs_jax = jax.tree_util.tree_map(jnp.asarray, observations)
        sample_act_jax = jnp.asarray(actions)
        if sample_act_jax.ndim == 1:
            sample_act_jax = sample_act_jax[jnp.newaxis, ...]

        if q_selector_feat_key not in sample_obs_jax:
            raise KeyError(
                f"create_drq_pixel requires observation key {q_selector_feat_key!r} for candidate Q state."
            )
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

        action_dim = actions.shape[-1] - (1 if fix_gripper else 0)
        if target_entropy is None:
            target_entropy = -action_dim / 2

        cfg = dict(
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            discount=discount,
            soft_target_update_rate=soft_target_update_rate,
            target_entropy=target_entropy,
            alpha_max=float(alpha_max) if alpha_max is not None else None,
            backup_entropy=backup_entropy,
            entropy_normalize_by_action_dim=bool(entropy_normalize_by_action_dim),
            image_keys=tuple(image_keys),
            reward_bias=reward_bias,
            augmentation_function=None,
            fix_gripper=fix_gripper,
            residual_alpha=float(residual_alpha),
            q_selector_feat_key=q_selector_feat_key,
            q_selector_discount=float(q_selector_discount),
        )

        agent = cls(state=state, config=cfg)

        if encoder_type == "resnet-pretrained":
            from serl_launcher.utils.train_utils import load_resnet10_params

            agent = load_resnet10_params(agent, image_keys)

        return agent
