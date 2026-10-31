from functools import partial
from typing import FrozenSet, Iterable, Optional, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.typing import Batch, Data, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack
from serl_launcher.vision.data_augmentations import batched_random_crop


class DrQAgent(SACAgent):
    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        temperature_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[callable] = None,
        reward_bias: float = 0.0,
        fix_gripper: bool = False,
        **kwargs,
    ):
        """
        Match SACAgent.create configuration, including reward_bias and fix_gripper used by
        update.
        """
        return super().create(
            rng,
            observations,
            actions,
            actor_def=actor_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            actor_optimizer_kwargs=actor_optimizer_kwargs,
            critic_optimizer_kwargs=critic_optimizer_kwargs,
            temperature_optimizer_kwargs=temperature_optimizer_kwargs,
            discount=discount,
            soft_target_update_rate=soft_target_update_rate,
            target_entropy=target_entropy,
            entropy_per_dim=entropy_per_dim,
            backup_entropy=backup_entropy,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            reward_bias=reward_bias,
            fix_gripper=fix_gripper,
            **kwargs,
        )

    @classmethod
    def create_drq(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "small",
        shared_encoder: bool = True,
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        **kwargs,
    ):
        """
        Create a new pixel-based agent, with no encoders.
        """

        fix_gripper = bool(kwargs.get("fix_gripper", False))

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

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

        encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
        }

        # Define networks
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")

        policy_def = Policy(
            encoder=encoders["actor"],
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

        drq_kwargs = dict(kwargs)
        drq_kwargs["augmentation_function"] = None

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            **drq_kwargs,
        )

        if encoder_type == "resnet-pretrained":  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params

            agent = load_resnet10_params(agent, image_keys)

        return agent

    def data_augmentation_fn(self, rng, observations):
        for pixel_key in self.config["image_keys"]:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset(
            {"actor", "critic", "temperature"}
        ),
        **kwargs,
    ) -> Tuple["DrQAgent", dict]:
        """
        SAC update with independent random crops of current/next pixel observations before loss
        computation; mutually exclusive with augmentation_function.
        """
        new_agent = self
        ik = self.config["image_keys"]
        if ik and ik[0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng = new_agent.state.rng
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
        return SACAgent.update(
            new_agent,
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=networks_to_update,
            **kwargs,
        )

    @partial(jax.jit, static_argnames=("utd_ratio", "pmap_axis"))
    def update_high_utd(
        self,
        batch: Batch,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        """
        Fast JITted high-UTD version of `.update`.

        Splits the batch into minibatches, performs `utd_ratio` critic
        (and target) updates, and then one actor/temperature update.

        Batch dimension must be divisible by `utd_ratio`.

        It also performs data augmentation on the observations and next_observations
        before updating the network.
        """
        ik = self.config["image_keys"]
        if ik and ik[0] not in batch["next_observations"]:
            batch = _unpack(batch)
        batch_size = batch["rewards"].shape[0]
        chex.assert_equal(batch_size % utd_ratio, 0)
        m = batch_size // utd_ratio

        def critic_step(i, agent: "DrQAgent") -> "DrQAgent":
            sub = jax.tree_util.tree_map(
                lambda x: jax.lax.dynamic_slice_in_dim(x, i * m, m, axis=0),
                batch,
            )
            new_a, _ = DrQAgent.update(
                agent,
                sub,
                pmap_axis=pmap_axis,
                networks_to_update=frozenset({"critic"}),
            )
            return new_a

        agent = jax.lax.fori_loop(0, utd_ratio, critic_step, self)
        return DrQAgent.update(
            agent,
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=frozenset({"actor", "temperature"}),
        )

    @partial(jax.jit, static_argnames=("pmap_axis",))
    def update_critics(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        new_agent, critic_infos = self.update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=frozenset({"critic"}),
        )
        del critic_infos["actor"]
        del critic_infos["temperature"]

        return new_agent, critic_infos