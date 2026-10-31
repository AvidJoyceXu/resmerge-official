# !/usr/bin/env python3

import copy
from typing import Iterable, Optional, Tuple

import gymnasium as gym
import jax
from jax import nn
import jax.numpy as jnp

from agentlace.trainer import TrainerConfig

from serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_single import SACAgentSingleArm
from serl_launcher.vision.data_augmentations import batched_random_crop

##############################################################################


def make_bc_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained"
):
    return BCAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [512, 512, 512],
            "dropout_rate": 0.25,
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        use_proprio=True,
        encoder_type=encoder_type,
        image_keys=image_keys,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )


def make_sac_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    fix_gripper: bool = False,
):
    agent = SACAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        fix_gripper=fix_gripper,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def make_drq_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    fix_gripper: bool = False,
    actor_learning_rate: float = 3e-4,
):
    """
    Construct pixel DrQ with cropping inside DrQAgent.update, without a separate
    augmentation_function.
    """
    return DrQAgent.create_drq(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        temperature_init=1e-2,
        discount=discount,
        fix_gripper=fix_gripper,
        backup_entropy=False,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        critic_ensemble_size=10,
        critic_subsample_size=2,
        actor_optimizer_kwargs={"learning_rate": actor_learning_rate},
    )


def make_sac_pixel_agent_single_arm(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
):
    agent = SACAgentSingleArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def make_conrft_octo_cp_pixel_agent_single_arm(
    seed,
    sample_obs,
    sample_action,
    sample_tasks,
    octo_model,
    encoder_type="resnet-pretrained",
    image_keys=("image",),
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    num_scales=40,
    sigma_data: float = 0.5,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    fix_gripper: bool = False,
    q_weight: float = 0.1,
    bc_weight: float = 1.0,
    optimizer_warmup_steps: int = 0,
    optimizer_cosine_decay_steps: int = None,
    optimizer_peak_learning_rate: float = 5e-5,
):
    agent = ConrftCPOctoAgentSingleArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        sample_tasks,
        encoder_type=encoder_type,
        use_proprio=True,
        octo_model=octo_model,
        image_keys=image_keys,
        fix_gripper=fix_gripper,
        policy_kwargs={
            "sigma_data": sigma_data,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "rho": rho,
            "steps": num_scales,
            "clip_denoised": True,
        },
        critic_network_kwargs={"activate_final": True, "hidden_dims": [256, 256]},
        policy_network_kwargs={"activate_final": True, "hidden_dims": [256, 256]},
        policy_t_network_kwargs={"t_dim": 16},
        actor_optimizer_kwargs={
            "learning_rate": optimizer_peak_learning_rate,
            "warmup_steps": optimizer_warmup_steps,
            "cosine_decay_steps": optimizer_cosine_decay_steps,
        },
        critic_optimizer_kwargs={
            "learning_rate": optimizer_peak_learning_rate,
            "warmup_steps": optimizer_warmup_steps,
            "cosine_decay_steps": optimizer_cosine_decay_steps,
        },
        num_scales=num_scales,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
        rho=rho,
        discount=discount,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        augmentation_function=make_batch_augmentation_func(image_keys),
        q_weight=q_weight,
        bc_weight=bc_weight,
    )
    return agent


def linear_schedule(step):
    init_value = 10.0
    end_value = 50.0
    decay_steps = 15_000

    linear_step = jnp.minimum(step, decay_steps)
    decayed_value = init_value + \
        (end_value - init_value) * (linear_step / decay_steps)
    return decayed_value


def make_batch_augmentation_func(image_keys) -> callable:

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    def augment_batch(batch: Batch, rng: PRNGKey) -> Batch:
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = data_augmentation_fn(
            next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )
        return batch

    return augment_batch


def dict_observation_space_drop_keys(
    obs_space: gym.spaces.Dict, drop_keys: Iterable[str]
) -> gym.spaces.Dict:
    """Create a Dict observation space excluding drop_keys, e.g. for image-free replay."""
    drop = frozenset(drop_keys)
    spaces = {
        k: copy.deepcopy(v) for k, v in obs_space.spaces.items() if k not in drop
    }
    return gym.spaces.Dict(spaces)


def strip_replay_batch_pixel_keys(batch: Batch, pixel_keys: Iterable[str]) -> Batch:
    """Remove pixel keys from current/next observations before learner transfer."""
    from flax.core import frozen_dict

    keys = tuple(pixel_keys)
    if not keys:
        return batch
    d = frozen_dict.unfreeze(batch)
    obs = dict(d["observations"])
    nxt = dict(d["next_observations"])
    for k in keys:
        obs.pop(k, None)
        nxt.pop(k, None)
    d["observations"] = obs
    d["next_observations"] = nxt
    return frozen_dict.freeze(d)


def make_sac_imle_preproj_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("side_policy_256",),
    feat_key="imle_preproj",
    mlp_hidden_dim=256,
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
):
    """
    SAC using only IMLE pre-projection features and a small MLP. No image augmentation is
    performed; image_keys is accepted for call compatibility only. Replay may retain images,
    which can be stripped before device transfer.
    """
    _ = tuple(image_keys)
    import flax.linen as nn
    from functools import partial

    from serl_launcher.common.imle_preproj_encoder import ImlePreprojEncoder
    from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
    from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
    from serl_launcher.networks.mlp import MLP

    
    enc_shared = ImlePreprojEncoder(feat_key=feat_key, mlp_hidden_dim=mlp_hidden_dim)

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
    critic_backbone = ensemblize(critic_backbone, 2)(name="critic_ensemble")
    critic_def = partial(Critic, encoder=enc_shared, network=critic_backbone)(name="critic")

    policy_def = Policy(
        encoder=enc_shared,
        network=MLP(**policy_network_kwargs),
        action_dim=sample_action.shape[-1],
        tanh_squash_distribution=True,
        std_parameterization="exp",
        std_min=1e-5,
        std_max=5,
        stop_gradient_on_encoder_output=True,
        name="actor",
    )

    temperature_def = GeqLagrangeMultiplier(
        init_value=1e-2,
        constraint_shape=(),
        constraint_type="geq",
        name="temperature",
    )

    sample_obs_jax = jax.tree_util.tree_map(jnp.asarray, sample_obs)
    sample_act_jax = jnp.asarray(sample_action)

    return SACAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs_jax,
        sample_act_jax,
        actor_def=policy_def,
        critic_def=critic_def,
        temperature_def=temperature_def,
        discount=discount,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        soft_target_update_rate=0.005,
        target_entropy=target_entropy,
        backup_entropy=False,
        reward_bias=reward_bias,
        image_keys=(),
        augmentation_function=None,
        fix_gripper=False,
    )


def make_sac_residual_imle_preproj_agent(
    seed,
    sample_obs,
    sample_action,
    *,
    residual_alpha: float,
    image_keys=("side_policy_256",),
    feat_key="imle_preproj",
    mlp_hidden_dim=256,
    reward_bias=0.0,
    target_entropy=None,
    alpha_max: float = 1.0,
    optimizer_lr: float = 3e-4,
    actor_learning_rate: float = 1e-5,
    discount=0.97,
    critic_ensemble_size: int = 5,
):
    """
    Construct residual SAC with unsquashed Gaussian u and a_full = a_base +
    residual_alpha*tanh(u). The shared encoder is updated by the critic; actor encoder output
    is stop-gradient and the mean head is zero-initialized.
    """
    _ = tuple(image_keys)
    import flax.linen as nn
    from functools import partial

    from serl_launcher.agents.continuous.sac_residual_imle import SACResidualIMLEAgent
    from serl_launcher.common.imle_preproj_encoder import ImlePreprojEncoder
    from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
    from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
    from serl_launcher.networks.mlp import MLP

    enc_shared = ImlePreprojEncoder(feat_key=feat_key, mlp_hidden_dim=mlp_hidden_dim)

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
    critic_def = partial(Critic, encoder=enc_shared, network=critic_backbone)(name="critic")

    policy_def = Policy(
        encoder=enc_shared,
        network=MLP(**policy_network_kwargs),
        action_dim=sample_action.shape[-1],
        tanh_squash_distribution=False,
        std_parameterization="exp",
        std_min=1e-5,
        std_max=5,
        stop_gradient_on_encoder_output=True,
        zero_init_mean_head=True,
        name="actor",
    )

    temperature_def = GeqLagrangeMultiplier(
        init_value=1e-2,
        constraint_shape=(),
        constraint_type="geq",
        name="temperature",
    )

    sample_obs_jax = jax.tree_util.tree_map(jnp.asarray, sample_obs)
    sample_act_jax = jnp.asarray(sample_action)
    if sample_act_jax.ndim == 1:
        sample_act_jax = sample_act_jax[jnp.newaxis, ...]

    agent = SACAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs_jax,
        sample_act_jax,
        actor_def=policy_def,
        critic_def=critic_def,
        temperature_def=temperature_def,
        actor_optimizer_kwargs={"learning_rate": float(actor_learning_rate)},
        critic_optimizer_kwargs={"learning_rate": float(optimizer_lr)},
        temperature_optimizer_kwargs={"learning_rate": float(optimizer_lr)},
        discount=discount,
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=None,
        soft_target_update_rate=0.005,
        target_entropy=target_entropy,
        backup_entropy=True,
        reward_bias=reward_bias,
        alpha_max=alpha_max,
        image_keys=(),
        augmentation_function=None,
        fix_gripper=False,
    )
    cfg = dict(agent.config)


    import numpy as _np
    cfg["residual_alpha"] = (
        _np.asarray(residual_alpha, dtype=_np.float32)
        if _np.ndim(residual_alpha) > 0
        else float(residual_alpha)
    )
    cfg["backup_entropy"] = True
    return SACResidualIMLEAgent(state=agent.state, config=cfg)


def make_sac_residual_imle_preproj_q_selector_agent(
    seed,
    sample_obs,
    sample_action,
    *,
    residual_alpha: float,
    q_num_candidates: int,
    q_selector_discount: float,
    q_selector_action_dim: int = -1,
    q_selector_lr: float = 3e-4,
    q_target_tau: float = 0.005,
    image_keys=("side_policy_256",),
    feat_key="imle_preproj",
    mlp_hidden_dim=512,
    reward_bias=0.0,
    target_entropy=None,
    alpha_max: float = 1.0,
    entropy_normalize_by_action_dim: bool = True,
    discount=0.97,
    critic_ensemble_size: int = 5,
    selector_q_mlp_hidden_dims: Tuple[int, int] = (256, 256),
    selector_q_preproj_encoder_dim: int = 512,
    actor_learning_rate: float = 1e-5,
    critic_learning_rate: float = 3e-4,
    temperature_learning_rate: float = 3e-4,
):
    """Construct residual IMLE SAC with pixel observations."""
    from serl_launcher.agents.continuous.sac_residual_imle_q_selector import (
        SACResidualIMLEQSelectorAgent,
    )

    _ = tuple(image_keys)
    return SACResidualIMLEQSelectorAgent.create_preproj(
        jax.random.PRNGKey(int(seed)),
        sample_obs,
        jnp.asarray(sample_action),
        residual_alpha=float(residual_alpha),
        q_num_candidates=int(q_num_candidates),
        q_selector_action_dim=int(q_selector_action_dim),
        q_selector_feat_key=feat_key,
        q_selector_discount=float(q_selector_discount),
        mlp_hidden_dim=int(mlp_hidden_dim),
        image_keys=tuple(image_keys),
        reward_bias=float(reward_bias),
        target_entropy=target_entropy,
        alpha_max=alpha_max,
        entropy_normalize_by_action_dim=entropy_normalize_by_action_dim,
        discount=float(discount),
        soft_target_update_rate=float(q_target_tau),
        critic_ensemble_size=int(critic_ensemble_size),
        selector_q_optimizer_kwargs={"learning_rate": float(q_selector_lr)},
        selector_q_mlp_hidden_dims=selector_q_mlp_hidden_dims,
        selector_q_preproj_encoder_dim=int(selector_q_preproj_encoder_dim),
        actor_learning_rate=float(actor_learning_rate),
        critic_learning_rate=float(critic_learning_rate),
        temperature_learning_rate=float(temperature_learning_rate),
    )


def make_sac_residual_imle_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    *,
    residual_alpha: float,
    image_keys=("side_policy_256",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    alpha_max: float = 1.0,
    optimizer_lr: float = 3e-4,
    actor_learning_rate: float = 1e-5,
    discount=0.97,
    fix_gripper: bool = False,
    critic_ensemble_size: int = 5,
):
    """Construct residual IMLE DrQ with pixel observations."""
    from serl_launcher.agents.continuous.sac_residual_imle import SACResidualIMLEAgent

    agent = SACAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
            "zero_init_mean_head": True,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        actor_optimizer_kwargs={"learning_rate": float(actor_learning_rate)},
        critic_optimizer_kwargs={"learning_rate": float(optimizer_lr)},
        temperature_optimizer_kwargs={"learning_rate": float(optimizer_lr)},
        temperature_init=1e-2,
        discount=discount,
        fix_gripper=fix_gripper,
        backup_entropy=True,
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        alpha_max=alpha_max,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    cfg = dict(agent.config)


    import numpy as _np
    cfg["residual_alpha"] = (
        _np.asarray(residual_alpha, dtype=_np.float32)
        if _np.ndim(residual_alpha) > 0
        else float(residual_alpha)
    )
    cfg["backup_entropy"] = True
    return SACResidualIMLEAgent(state=agent.state, config=cfg)


def make_drq_residual_imle_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    *,
    residual_alpha: float,
    image_keys=("side_policy_256",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    alpha_max: float = 1.0,
    optimizer_lr: float = 3e-4,
    actor_learning_rate: float = 1e-5,
    discount=0.97,
    fix_gripper: bool = False,
    critic_ensemble_size: int = 5,
):
    """Construct residual IMLE SAC with a candidate Q selector."""
    from serl_launcher.agents.continuous.drq_residual_imle import DrQResidualIMLEAgent

    agent = DrQAgent.create_drq(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
            "zero_init_mean_head": True,
        },
        actor_optimizer_kwargs={"learning_rate": float(actor_learning_rate)},
        critic_optimizer_kwargs={"learning_rate": float(optimizer_lr)},
        temperature_optimizer_kwargs={"learning_rate": float(optimizer_lr)},
        temperature_init=1e-2,
        discount=discount,
        fix_gripper=fix_gripper,
        backup_entropy=True,
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        alpha_max=alpha_max,
    )
    cfg = dict(agent.config)
    cfg["residual_alpha"] = float(residual_alpha)
    cfg["backup_entropy"] = True
    return DrQResidualIMLEAgent(state=agent.state, config=cfg)


def make_drq_residual_imle_pixel_q_selector_agent(
    seed,
    sample_obs,
    sample_action,
    *,
    residual_alpha: float,
    q_num_candidates: int,
    q_selector_discount: float,
    q_selector_action_dim: int = -1,
    q_selector_lr: float = 3e-4,
    q_target_tau: float = 0.005,
    image_keys=("side_policy_256",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    alpha_max: float = 1.0,
    entropy_normalize_by_action_dim: bool = True,
    discount=0.97,
    fix_gripper: bool = False,
    critic_ensemble_size: int = 5,
    selector_q_mlp_hidden_dims: Tuple[int, int] = (256, 256),
    selector_q_preproj_encoder_dim: int = 512,
    feat_key: str = "imle_preproj",
    optimizer_lr: float = 3e-4,
    actor_learning_rate: float = 1e-5,
):
    """Construct residual IMLE pixel DrQ with selector_q; sample_obs must include feat_key."""
    from serl_launcher.agents.continuous.drq_residual_imle_q_selector import (
        DrQResidualIMLEQSelectorAgent,
    )

    sa = jnp.asarray(sample_action)
    if sa.ndim == 1:
        sa = sa[jnp.newaxis, ...]

    return DrQResidualIMLEQSelectorAgent.create_drq_pixel(
        jax.random.PRNGKey(int(seed)),
        sample_obs,
        sa,
        residual_alpha=float(residual_alpha),
        q_num_candidates=int(q_num_candidates),
        q_selector_action_dim=int(q_selector_action_dim),
        q_selector_feat_key=str(feat_key),
        q_selector_discount=float(q_selector_discount),
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        critic_ensemble_size=int(critic_ensemble_size),
        critic_subsample_size=None,
        temperature_init=1e-2,
        discount=float(discount),
        soft_target_update_rate=float(q_target_tau),
        target_entropy=target_entropy,
        alpha_max=alpha_max,
        entropy_normalize_by_action_dim=entropy_normalize_by_action_dim,
        reward_bias=float(reward_bias),
        backup_entropy=True,
        fix_gripper=bool(fix_gripper),
        q_selector_optimizer_kwargs={"learning_rate": float(q_selector_lr)},
        selector_q_mlp_hidden_dims=selector_q_mlp_hidden_dims,
        selector_q_preproj_encoder_dim=int(selector_q_preproj_encoder_dim),
        actor_optimizer_kwargs={"learning_rate": float(actor_learning_rate)},
        critic_optimizer_kwargs={"learning_rate": float(optimizer_lr)},
        temperature_optimizer_kwargs={"learning_rate": float(optimizer_lr)},
    )


def make_trainer_config(port_number: int = 3333, broadcast_port: int = 3334):
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )


def make_wandb_logger(
    project: str = "conrft",
    description: str = "serl_launcher",
    debug: bool = False,
    entity: Optional[str] = None,
    group: Optional[str] = None,
    wandb_mode: Optional[str] = None,
):
    wandb_config = WandBLogger.get_default_config()
    upd = {
        "project": project,
        "exp_descriptor": description,
        "tag": description,
    }
    if entity is not None and str(entity).strip():
        upd["entity"] = str(entity).strip()
    if group is not None and str(group).strip():
        upd["group"] = str(group).strip()
    if wandb_mode in ("online", "offline"):
        upd["wandb_mode"] = wandb_mode
    wandb_config.update(upd)
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger


def _td3_preproj_internal(
    seed,
    sample_obs,
    sample_action,
    *,
    residual_alpha: float,
    image_keys,
    feat_key: str,
    mlp_hidden_dim: int,
    reward_bias: float,
    discount: float,
    critic_ensemble_size: int,
    critic_subsample_size: int,
    explore_noise: float,
    target_policy_noise: float,
    target_noise_clip: float,
    soft_target_update_rate: float,
    actor_lr: float,
    critic_lr: float,
):
    """Original TD3 variant: alpha*tanh(u), shared encoder and no action-space clamp."""
    import flax.linen as flax_nn
    from functools import partial as _partial

    from serl_launcher.agents.continuous.td3_residual_imle import TD3ResidualIMLEAgent
    from serl_launcher.common.imle_preproj_encoder import ImlePreprojEncoder
    from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
    from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
    from serl_launcher.networks.mlp import MLP

    _ = tuple(image_keys)
    enc_shared = ImlePreprojEncoder(feat_key=feat_key, mlp_hidden_dim=mlp_hidden_dim)

    policy_network_kwargs = dict(
        activations=flax_nn.tanh,
        use_layer_norm=True,
        hidden_dims=[256, 256],
        activate_final=True,
    )
    critic_network_kwargs = dict(
        activations=flax_nn.tanh,
        use_layer_norm=True,
        hidden_dims=[256, 256],
        activate_final=True,
    )

    critic_backbone = _partial(MLP, **critic_network_kwargs)
    critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
        name="critic_ensemble"
    )
    critic_def = _partial(Critic, encoder=enc_shared, network=critic_backbone)(name="critic")

    policy_def = Policy(
        encoder=enc_shared,
        network=MLP(**policy_network_kwargs),
        action_dim=sample_action.shape[-1],
        tanh_squash_distribution=False,
        std_parameterization="fixed",
        fixed_std=jnp.full((sample_action.shape[-1],), float(explore_noise), dtype=jnp.float32),
        std_min=1e-8,
        std_max=10.0,
        stop_gradient_on_encoder_output=True,
        zero_init_mean_head=True,
        name="actor",
    )
    temperature_def = GeqLagrangeMultiplier(
        init_value=1e-2, constraint_shape=(), constraint_type="geq", name="temperature"
    )

    sample_obs_jax = jax.tree_util.tree_map(jnp.asarray, sample_obs)
    sample_act_jax = jnp.asarray(sample_action)
    if sample_act_jax.ndim == 1:
        sample_act_jax = sample_act_jax[jnp.newaxis, ...]

    agent = SACAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs_jax,
        sample_act_jax,
        actor_def=policy_def,
        critic_def=critic_def,
        temperature_def=temperature_def,
        actor_optimizer_kwargs={"learning_rate": float(actor_lr)},
        critic_optimizer_kwargs={"learning_rate": float(critic_lr)},
        temperature_optimizer_kwargs={"learning_rate": 0.0},
        discount=discount,
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=critic_subsample_size,
        soft_target_update_rate=soft_target_update_rate,
        target_entropy=None,
        backup_entropy=False,
        reward_bias=reward_bias,
        image_keys=(),
        augmentation_function=None,
        fix_gripper=False,
    )
    cfg = dict(agent.config)
    cfg.update(
        residual_alpha=float(residual_alpha),
        target_policy_noise=float(target_policy_noise),
        target_noise_clip=float(target_noise_clip),
        explore_noise=float(explore_noise),
        backup_entropy=False,
    )
    return TD3ResidualIMLEAgent(state=agent.state, config=cfg)


def _td3_preproj_resfit_internal(
    seed,
    sample_obs,
    sample_action,
    *,
    residual_alpha: float,
    image_keys,
    feat_key: str,
    mlp_hidden_dim: int,
    reward_bias: float,
    discount: float,
    critic_ensemble_size: int,
    critic_subsample_size: int,
    explore_noise: float,
    target_policy_noise: float,
    target_noise_clip: float,
    stddev_clip: float,
    soft_target_update_rate: float,
    actor_lr: float,
    critic_lr: float,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    actor_action_l2_reg_weight: float = 0.0,
    residual_actor_concat_base: bool = False,
):
    """Resfit TD3 variant: scaled mu, box clamp and optional base-action concatenation."""
    import flax.linen as flax_nn
    from functools import partial as _partial

    from serl_launcher.agents.continuous.td3_residual_imle_resfit import (
        TD3ResidualIMLEAgentResfit,
    )
    from serl_launcher.common.imle_preproj_encoder import ImlePreprojEncoder
    from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
    from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
    from serl_launcher.networks.mlp import MLP

    _ = tuple(image_keys)
    enc_critic = ImlePreprojEncoder(feat_key=feat_key, mlp_hidden_dim=mlp_hidden_dim)
    if residual_actor_concat_base:
        enc_actor = ImlePreprojEncoder(
            feat_key=feat_key,
            mlp_hidden_dim=mlp_hidden_dim,
            concat_action_base_key="action_base",
        )
    else:
        enc_actor = enc_critic

    policy_network_kwargs = dict(
        activations=flax_nn.tanh,
        use_layer_norm=True,
        hidden_dims=[256, 256],
        activate_final=True,
    )
    critic_network_kwargs = dict(
        activations=flax_nn.tanh,
        use_layer_norm=True,
        hidden_dims=[256, 256],
        activate_final=True,
    )

    critic_backbone = _partial(MLP, **critic_network_kwargs)
    critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
        name="critic_ensemble"
    )
    critic_def = _partial(Critic, encoder=enc_critic, network=critic_backbone)(name="critic")

    policy_def = Policy(
        encoder=enc_actor,
        network=MLP(**policy_network_kwargs),
        action_dim=sample_action.shape[-1],
        tanh_squash_distribution=True,
        std_parameterization="fixed",
        fixed_std=jnp.full((sample_action.shape[-1],), float(explore_noise), dtype=jnp.float32),
        std_min=1e-8,
        std_max=10.0,
        stop_gradient_on_encoder_output=True,
        zero_init_mean_head=True,
        name="actor",
    )
    temperature_def = GeqLagrangeMultiplier(
        init_value=1e-2, constraint_shape=(), constraint_type="geq", name="temperature"
    )

    sample_obs_jax = jax.tree_util.tree_map(jnp.asarray, sample_obs)
    sample_act_jax = jnp.asarray(sample_action)
    if sample_act_jax.ndim == 1:
        sample_act_jax = sample_act_jax[jnp.newaxis, ...]

    agent = SACAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs_jax,
        sample_act_jax,
        actor_def=policy_def,
        critic_def=critic_def,
        temperature_def=temperature_def,
        actor_optimizer_kwargs={"learning_rate": float(actor_lr)},
        critic_optimizer_kwargs={"learning_rate": float(critic_lr)},
        temperature_optimizer_kwargs={"learning_rate": 0.0},
        discount=discount,
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=critic_subsample_size,
        soft_target_update_rate=soft_target_update_rate,
        target_entropy=None,
        backup_entropy=False,
        reward_bias=reward_bias,
        image_keys=(),
        augmentation_function=None,
        fix_gripper=False,
    )
    cfg = dict(agent.config)
    cfg.update(
        action_scale=float(residual_alpha),
        residual_alpha=float(residual_alpha),
        action_low=jnp.asarray(action_low, dtype=jnp.float32),
        action_high=jnp.asarray(action_high, dtype=jnp.float32),
        target_policy_noise=float(target_policy_noise),
        target_noise_clip=float(target_noise_clip),
        stddev_clip=float(stddev_clip),
        explore_noise=float(explore_noise),
        actor_action_l2_reg_weight=float(actor_action_l2_reg_weight),
        residual_actor_concat_base=bool(residual_actor_concat_base),
        backup_entropy=False,
    )
    return TD3ResidualIMLEAgentResfit(state=agent.state, config=cfg)


def make_td3_residual_imle_preproj_agent(
    seed,
    sample_obs,
    sample_action,
    *,
    residual_alpha: float,
    image_keys=("side_policy_256",),
    feat_key: str = "imle_preproj",
    mlp_hidden_dim: int = 256,
    reward_bias: float = 0.0,
    discount: float = 0.97,
    critic_ensemble_size: int = 10,
    critic_subsample_size: int = 2,
    explore_noise: float = 0.1,
    target_policy_noise: float = 0.2,
    target_noise_clip: float = 0.5,
    soft_target_update_rate: float = 0.005,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
):
    """
    Construct the original TD3 IMLE feature residual agent independently of the resfit variant.
    """
    return _td3_preproj_internal(
        seed=seed,
        sample_obs=sample_obs,
        sample_action=sample_action,
        residual_alpha=residual_alpha,
        image_keys=image_keys,
        feat_key=feat_key,
        mlp_hidden_dim=mlp_hidden_dim,
        reward_bias=reward_bias,
        discount=discount,
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=critic_subsample_size,
        explore_noise=explore_noise,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        soft_target_update_rate=soft_target_update_rate,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
    )


def make_td3_residual_imle_preproj_agent_resfit(
    seed,
    sample_obs,
    sample_action,
    *,
    residual_alpha: float,
    image_keys=("side_policy_256",),
    feat_key: str = "imle_preproj",
    mlp_hidden_dim: int = 256,
    reward_bias: float = 0.0,
    discount: float = 0.99,
    critic_ensemble_size: int = 10,
    critic_subsample_size: int = 2,
    explore_noise: float = 0.05,
    target_policy_noise: float = 0.05,
    target_noise_clip: float = 0.3,
    stddev_clip: float = 0.3,
    soft_target_update_rate: float = 0.005,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    action_low=None,
    action_high=None,
    actor_action_l2_reg_weight: float = 0.0,
    residual_actor_concat_base: bool = True,
):
    """Construct resfit TD3 using IMLE features; action_low and action_high are required."""
    if action_low is None or action_high is None:
        raise ValueError(
            "make_td3_residual_imle_preproj_agent_resfit requires action_low / action_high"
        )
    return _td3_preproj_resfit_internal(
        seed=seed,
        sample_obs=sample_obs,
        sample_action=sample_action,
        residual_alpha=residual_alpha,
        image_keys=image_keys,
        feat_key=feat_key,
        mlp_hidden_dim=mlp_hidden_dim,
        reward_bias=reward_bias,
        discount=discount,
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=critic_subsample_size,
        explore_noise=explore_noise,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        stddev_clip=stddev_clip,
        soft_target_update_rate=soft_target_update_rate,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        action_low=action_low,
        action_high=action_high,
        actor_action_l2_reg_weight=actor_action_l2_reg_weight,
        residual_actor_concat_base=residual_actor_concat_base,
    )


def make_td3_residual_imle_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    *,
    residual_alpha: float,
    image_keys=("side_policy_256",),
    encoder_type: str = "resnet-pretrained",
    reward_bias: float = 0.0,
    discount: float = 0.97,
    fix_gripper: bool = False,
    critic_ensemble_size: int = 10,
    critic_subsample_size: int = 2,
    explore_noise: float = 0.1,
    target_policy_noise: float = 0.2,
    target_noise_clip: float = 0.5,
    soft_target_update_rate: float = 0.005,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
):
    """TD3 pixel residual agent with ResNet/proprio and augmentation in SACAgent.update."""
    from serl_launcher.agents.continuous.td3_residual_imle import TD3ResidualIMLEAgent

    agent = SACAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "fixed",
            "fixed_std": jnp.full(
                (sample_action.shape[-1],), float(explore_noise), dtype=jnp.float32
            ),
            "std_min": 1e-8,
            "std_max": 10.0,
            "zero_init_mean_head": True,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        actor_optimizer_kwargs={"learning_rate": float(actor_lr)},
        critic_optimizer_kwargs={"learning_rate": float(critic_lr)},
        temperature_optimizer_kwargs={"learning_rate": 0.0},
        temperature_init=1e-2,
        discount=discount,
        fix_gripper=fix_gripper,
        backup_entropy=False,
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=critic_subsample_size,
        reward_bias=reward_bias,
        target_entropy=None,
        soft_target_update_rate=soft_target_update_rate,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    cfg = dict(agent.config)
    cfg.update(
        residual_alpha=float(residual_alpha),
        target_policy_noise=float(target_policy_noise),
        target_noise_clip=float(target_noise_clip),
        explore_noise=float(explore_noise),
        backup_entropy=False,
    )
    return TD3ResidualIMLEAgent(state=agent.state, config=cfg)


def make_drq_td3_residual_imle_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    *,
    residual_alpha: float,
    image_keys=("side_policy_256",),
    encoder_type: str = "resnet-pretrained",
    reward_bias: float = 0.0,
    discount: float = 0.97,
    fix_gripper: bool = False,
    critic_ensemble_size: int = 10,
    critic_subsample_size: int = 2,
    explore_noise: float = 0.1,
    target_policy_noise: float = 0.2,
    target_noise_clip: float = 0.5,
    soft_target_update_rate: float = 0.005,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
):
    """TD3 pixel residual agent with internal DrQ cropping and no augmentation_function."""
    from serl_launcher.agents.continuous.td3_residual_imle import TD3ResidualIMLEAgent

    agent = DrQAgent.create_drq(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "fixed",
            "fixed_std": jnp.full(
                (sample_action.shape[-1],), float(explore_noise), dtype=jnp.float32
            ),
            "std_min": 1e-8,
            "std_max": 10.0,
            "zero_init_mean_head": True,
        },
        actor_optimizer_kwargs={"learning_rate": float(actor_lr)},
        critic_optimizer_kwargs={"learning_rate": float(critic_lr)},
        temperature_optimizer_kwargs={"learning_rate": 0.0},
        temperature_init=1e-2,
        discount=discount,
        fix_gripper=fix_gripper,
        backup_entropy=False,
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=critic_subsample_size,
        reward_bias=reward_bias,
        target_entropy=None,
        soft_target_update_rate=soft_target_update_rate,
    )
    cfg = dict(agent.config)
    cfg.update(
        residual_alpha=float(residual_alpha),
        target_policy_noise=float(target_policy_noise),
        target_noise_clip=float(target_noise_clip),
        explore_noise=float(explore_noise),
        backup_entropy=False,
    )
    
    
    from serl_launcher.agents.continuous.drq import DrQAgent as _DrQAgent

    class DrQTD3ResidualIMLEAgent(TD3ResidualIMLEAgent, _DrQAgent):
        """Combine DrQ update-time cropping with TD3 losses through method resolution order."""

    return DrQTD3ResidualIMLEAgent(state=agent.state, config=cfg)

