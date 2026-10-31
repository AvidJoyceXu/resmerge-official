# Modified by the ResMerge contributors, 2026-10-31; see real_robot/THIRD_PARTY.md.
"""
Low-rank actor for IMLE-feature residual SAC. The encoder and two trunk layers use A(in,rank)
and B(rank,out), LayerNorm and tanh; full-rank mean/logstd heads follow. Mean is
zero-initialized. The critic has a separate full-rank encoder. Actor loss updates the entire
actor path. Exports transpose JAX kernels into PyTorch layout and preserve LayerNorm under
_jax_extras. Use analyze_correction_field.py and quick_merge.py for the multi-task merge workflow.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import partial
from typing import Callable, List, Optional, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_residual_imle import SACResidualIMLEAgent
from serl_launcher.common.common import default_init
from serl_launcher.common.imle_preproj_encoder import ImlePreprojEncoder
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP


# ============================================================================

# ============================================================================

def _lora_a_init(key, shape, dtype=jnp.float32):
    """Initialize A with Kaiming uniform, matching the default PyTorch Linear convention."""
    fan_in = shape[0]
    bound = jnp.sqrt(1.0 / fan_in)
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def _lora_b_init(key, shape, dtype=jnp.float32):
    """Initialize B with normal standard deviation 0.01 for small initial residuals."""
    return jax.random.normal(key, shape, dtype) * 0.01


# ============================================================================

# ============================================================================

class LoRAImlePreprojEncoder(nn.Module):
    """
    Low-rank IMLE encoder: factor the dense projection into A and B, followed by
    LayerNorm/tanh. Actor gradients flow through this encoder; it is not shared with the
    critic.
    """

    feat_key: str = "imle_preproj"
    hidden_dim: int = 256
    rank: int = 16

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

        # LoRA: Dense(in→R, no bias) ∘ Dense(R→hidden_dim, bias)
        x = nn.Dense(
            self.rank,
            use_bias=False,
            kernel_init=_lora_a_init,
            name="enc_lora_A",
        )(x)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=_lora_b_init,
            bias_init=nn.initializers.zeros,
            name="enc_lora_B",
        )(x)
        x = nn.LayerNorm()(x)
        x = jnp.tanh(x)
        return x


# ============================================================================

# ============================================================================

class LoRAMLP(nn.Module):
    """
    MLP with low-rank hidden projections; other behavior follows the shared MLP implementation.
    """

    hidden_dims: Sequence[int]
    rank: int = 16
    activations: Callable[[jnp.ndarray], jnp.ndarray] | str = nn.swish
    activate_final: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        activations = self.activations
        if isinstance(activations, str):
            activations = getattr(nn, activations)

        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(
                self.rank,
                use_bias=False,
                kernel_init=_lora_a_init,
                name=f"lora_{i}_A",
            )(x)
            x = nn.Dense(
                size,
                kernel_init=_lora_b_init,
                bias_init=nn.initializers.zeros,
                name=f"lora_{i}_B",
            )(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not train
                    )
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = activations(x)
        return x


# ============================================================================

# ============================================================================

def make_sac_residual_imle_preproj_lora_agent(
    seed,
    sample_obs,
    sample_action,
    *,
    residual_alpha: float,
    image_keys=("side_policy_256",),
    feat_key: str = "imle_preproj",
    mlp_hidden_dim: int = 256,
    reward_bias: float = 0.0,
    target_entropy=None,
    alpha_max: float = 1.0,
    optimizer_lr: float = 3e-4,
    actor_learning_rate: float = 1e-5,
    discount: float = 0.97,
    critic_ensemble_size: int = 5,
    lora_rank: int = 16,
):
    """
    Construct a low-rank encoder and two-layer actor trunk, full-rank output heads, and an
    independent full-rank critic. Actor loss updates the actor encoder. lora_rank applies to
    encoder and trunk projections.
    """
    _ = tuple(image_keys)

    
    actor_encoder = LoRAImlePreprojEncoder(
        feat_key=feat_key,
        hidden_dim=mlp_hidden_dim,
        rank=int(lora_rank),
    )
    critic_encoder = ImlePreprojEncoder(
        feat_key=feat_key,
        mlp_hidden_dim=mlp_hidden_dim,
    )

    actor_trunk_kwargs = dict(
        rank=int(lora_rank),
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
    critic_def = partial(Critic, encoder=critic_encoder, network=critic_backbone)(
        name="critic"
    )

    policy_def = Policy(
        encoder=actor_encoder,
        network=LoRAMLP(**actor_trunk_kwargs),
        action_dim=sample_action.shape[-1],
        tanh_squash_distribution=False,
        std_parameterization="exp",
        std_min=1e-5,
        std_max=5,
        
        stop_gradient_on_encoder_output=False,
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
    cfg["residual_alpha"] = float(residual_alpha)
    cfg["backup_entropy"] = True
    cfg["lora_rank"] = int(lora_rank)
    return SACResidualIMLEAgent(state=agent.state, config=cfg)


# ============================================================================

# ============================================================================

def _as_np(x) -> np.ndarray:
    """JAX device array → host numpy (float32)."""
    return np.asarray(jax.device_get(x), dtype=np.float32)


def _to_torch(arr: np.ndarray, transpose: bool = False):
    """
    Convert NumPy to torch; transpose JAX (in,out) kernels to PyTorch (out,in) weights when
    requested.
    """
    import torch
    if transpose:
        arr = arr.T
    return torch.from_numpy(np.ascontiguousarray(arr))


def _params_subtree(actor_params, key):
    """Read a parameter subtree from a dict or FrozenDict."""
    if hasattr(actor_params, "unfreeze"):
        actor_params = actor_params.unfreeze()
    return actor_params[key]


def _actor_policy_params(params) -> dict:
    """
    Support the modules_actor and actor checkpoint parameter layouts.
    """
    if hasattr(params, "unfreeze"):
        params = params.unfreeze()
    if "modules_actor" in params:
        return _params_subtree(params, "modules_actor")
    if "actor" in params:
        return _params_subtree(params, "actor")
    raise KeyError(
        "Neither modules_actor nor actor was found in params; "
        f"top-level keys={list(params.keys())[:12]}"
    )


def save_for_merge(
    agent,
    save_path: str,
    task_id,
    additional_info: Optional[dict] = None,
):
    """
    Export actor weights as task_id, params, _jax_extras and additional_info. params contains
    fc1/fc2/fc3 low-rank A/B/bias tensors, mean/logstd output heads and rank/shape metadata.
    Transpose JAX kernels into PyTorch layout. _jax_extras preserves encoder/trunk LayerNorm
    scale and bias for reloading.
    """
    import torch

    actor_params = _actor_policy_params(agent.state.params)

    # --- encoder (fc1) ---
    enc_subtree = _params_subtree(actor_params, "encoder")
    enc_A_jax = _as_np(enc_subtree["enc_lora_A"]["kernel"])          # (in, R)
    enc_B_jax = _as_np(enc_subtree["enc_lora_B"]["kernel"])          # (R, hidden)
    enc_B_bias = _as_np(enc_subtree["enc_lora_B"]["bias"])           # (hidden,)
    enc_LN_scale = _as_np(enc_subtree["LayerNorm_0"]["scale"])
    enc_LN_bias = _as_np(enc_subtree["LayerNorm_0"]["bias"])

    # --- trunk (fc2 = lora_0, fc3 = lora_1) ---
    trunk_subtree = _params_subtree(actor_params, "network")
    fc2_A_jax = _as_np(trunk_subtree["lora_0_A"]["kernel"])
    fc2_B_jax = _as_np(trunk_subtree["lora_0_B"]["kernel"])
    fc2_B_bias = _as_np(trunk_subtree["lora_0_B"]["bias"])
    fc3_A_jax = _as_np(trunk_subtree["lora_1_A"]["kernel"])
    fc3_B_jax = _as_np(trunk_subtree["lora_1_B"]["kernel"])
    fc3_B_bias = _as_np(trunk_subtree["lora_1_B"]["bias"])
    trunk_LN0_scale = _as_np(trunk_subtree["LayerNorm_0"]["scale"])
    trunk_LN0_bias = _as_np(trunk_subtree["LayerNorm_0"]["bias"])
    trunk_LN1_scale = _as_np(trunk_subtree["LayerNorm_1"]["scale"])
    trunk_LN1_bias = _as_np(trunk_subtree["LayerNorm_1"]["bias"])

    
    mean_w_jax = _as_np(actor_params["Dense_0"]["kernel"])           # (256, action_dim)
    mean_b = _as_np(actor_params["Dense_0"]["bias"])
    logstd_w_jax = _as_np(actor_params["Dense_1"]["kernel"])
    logstd_b = _as_np(actor_params["Dense_1"]["bias"])

    
    R = int(agent.config.get("lora_rank", enc_A_jax.shape[1]))
    obs_dim = int(enc_A_jax.shape[0])
    action_dim = int(mean_w_jax.shape[1])

    params = {
        "lora_layers": {
            "fc1": {
                "A": _to_torch(enc_A_jax, transpose=True),          # (R, in)
                "B": _to_torch(enc_B_jax, transpose=True),          # (out, R)
                "B_bias": _to_torch(enc_B_bias),
            },
            "fc2": {
                "A": _to_torch(fc2_A_jax, transpose=True),
                "B": _to_torch(fc2_B_jax, transpose=True),
                "B_bias": _to_torch(fc2_B_bias),
            },
            "fc3": {
                "A": _to_torch(fc3_A_jax, transpose=True),
                "B": _to_torch(fc3_B_jax, transpose=True),
                "B_bias": _to_torch(fc3_B_bias),
            },
        },
        "output_layers": {
            "fc_mean": {
                "weight": _to_torch(mean_w_jax, transpose=True),    # (action_dim, 256)
                "bias": _to_torch(mean_b),
            },
            "fc_logstd": {
                "weight": _to_torch(logstd_w_jax, transpose=True),
                "bias": _to_torch(logstd_b),
            },
        },
        "meta": {
            "rank": R,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
        },
    }

    
    jax_extras = {
        "enc_LN": {
            "scale": _to_torch(enc_LN_scale),
            "bias": _to_torch(enc_LN_bias),
        },
        "trunk_LN_0": {
            "scale": _to_torch(trunk_LN0_scale),
            "bias": _to_torch(trunk_LN0_bias),
        },
        "trunk_LN_1": {
            "scale": _to_torch(trunk_LN1_scale),
            "bias": _to_torch(trunk_LN1_bias),
        },
    }

    save_dict = {
        "task_id": task_id,
        "params": params,
        "_jax_extras": jax_extras,
        "additional_info": additional_info or {},
    }

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(save_dict, save_path)
    print(
        f"[imle_lora] save_for_merge: task_id={task_id} rank={R} obs_dim={obs_dim} "
        f"action_dim={action_dim} → {save_path}"
    )


def load_merged_for_actor(
    agent,
    merged_pt_path: str,
    source_pt_paths_for_ln: Optional[List[str]] = None,
    weights_for_ln: Optional[List[float]] = None,
):
    """
    Load exported/merged actor tensors into a compatible agent. Embedded _jax_extras are
    authoritative unless explicit weights_for_ln request a source-based override. Legacy
    exports can recover LayerNorm from source exports, with verified source order for saved
    nonuniform weights. Validate all tensors before replacing actor parameters.
    Return an agent with replaced parameters. Use quick_merge.py for multi-task merging.
    """
    import torch

    merged = torch.load(merged_pt_path, map_location="cpu", weights_only=False)
    saved_alpha = merged.get("additional_info", {}).get("residual_alpha")
    if saved_alpha is not None:
        expected_alpha = float(agent.config["residual_alpha"])
        if (not np.isfinite(saved_alpha) or saved_alpha <= 0
                or not np.isclose(float(saved_alpha), expected_alpha)):
            raise ValueError(
                f"Actor residual_alpha={saved_alpha} does not match configured scale={expected_alpha}"
            )
    
    if "merged_params" in merged:
        m_params = merged["merged_params"]
    elif "params" in merged:
        m_params = merged["params"]
    else:
        raise KeyError(
            f"merged_pt contains neither 'merged_params' nor 'params': {list(merged.keys())}"
        )

    
    if "_jax_extras" in merged and weights_for_ln is None:
        ln_extras = merged["_jax_extras"]
    else:
        if not source_pt_paths_for_ln:
            raise ValueError(
                "Source-based LayerNorm recovery requires source_pt_paths_for_ln. "
                "Prefer a self-contained export made with quick_merge.py."
            )
        weights = _legacy_ln_weights(merged, source_pt_paths_for_ln, weights_for_ln)
        ln_extras = _average_ln_from_sources(source_pt_paths_for_ln, weights)

    
    def t2j(t, transpose=False):
        if not torch.is_tensor(t) or not torch.is_floating_point(t):
            raise ValueError(f"{merged_pt_path}: actor and LayerNorm parameters must be floating tensors")
        arr = t.detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(arr).all():
            raise ValueError(f"{merged_pt_path}: actor or LayerNorm parameters contain nonfinite values")
        if transpose:
            arr = arr.T
        return jnp.asarray(np.ascontiguousarray(arr))

    enc_A = t2j(m_params["lora_layers"]["fc1"]["A"], transpose=True)
    enc_B = t2j(m_params["lora_layers"]["fc1"]["B"], transpose=True)
    enc_B_bias = t2j(m_params["lora_layers"]["fc1"]["B_bias"])
    fc2_A = t2j(m_params["lora_layers"]["fc2"]["A"], transpose=True)
    fc2_B = t2j(m_params["lora_layers"]["fc2"]["B"], transpose=True)
    fc2_B_bias = t2j(m_params["lora_layers"]["fc2"]["B_bias"])
    fc3_A = t2j(m_params["lora_layers"]["fc3"]["A"], transpose=True)
    fc3_B = t2j(m_params["lora_layers"]["fc3"]["B"], transpose=True)
    fc3_B_bias = t2j(m_params["lora_layers"]["fc3"]["B_bias"])

    mean_w = t2j(m_params["output_layers"]["fc_mean"]["weight"], transpose=True)
    mean_b = t2j(m_params["output_layers"]["fc_mean"]["bias"])
    logstd_w = t2j(m_params["output_layers"]["fc_logstd"]["weight"], transpose=True)
    logstd_b = t2j(m_params["output_layers"]["fc_logstd"]["bias"])

    enc_LN_scale = t2j(ln_extras["enc_LN"]["scale"])
    enc_LN_bias = t2j(ln_extras["enc_LN"]["bias"])
    trunk_LN0_scale = t2j(ln_extras["trunk_LN_0"]["scale"])
    trunk_LN0_bias = t2j(ln_extras["trunk_LN_0"]["bias"])
    trunk_LN1_scale = t2j(ln_extras["trunk_LN_1"]["scale"])
    trunk_LN1_bias = t2j(ln_extras["trunk_LN_1"]["bias"])

    
    is_frozen = hasattr(agent.state.params, "unfreeze")
    # unfreeze also recursively copies plain dictionaries; a failed load must not mutate the
    # existing agent or leave it with a partially replaced actor.
    params = flax.core.unfreeze(agent.state.params)

    actor_key = "modules_actor" if "modules_actor" in params else "actor"
    actor = params[actor_key]
    expected_meta = {
        "rank": actor["encoder"]["enc_lora_A"]["kernel"].shape[1],
        "obs_dim": actor["encoder"]["enc_lora_A"]["kernel"].shape[0],
        "action_dim": actor["Dense_0"]["bias"].shape[0],
    }
    for key, expected in expected_meta.items():
        if int(m_params["meta"][key]) != expected:
            raise ValueError(
                f"{merged_pt_path}: actor {key}={m_params['meta'][key]} does not match "
                f"the configured agent ({expected}); use matching rank/observation/action settings"
            )
    updates = [
        (("encoder", "enc_lora_A", "kernel"), enc_A),
        (("encoder", "enc_lora_B", "kernel"), enc_B),
        (("encoder", "enc_lora_B", "bias"), enc_B_bias),
        (("encoder", "LayerNorm_0", "scale"), enc_LN_scale),
        (("encoder", "LayerNorm_0", "bias"), enc_LN_bias),
        (("network", "lora_0_A", "kernel"), fc2_A),
        (("network", "lora_0_B", "kernel"), fc2_B),
        (("network", "lora_0_B", "bias"), fc2_B_bias),
        (("network", "lora_1_A", "kernel"), fc3_A),
        (("network", "lora_1_B", "kernel"), fc3_B),
        (("network", "lora_1_B", "bias"), fc3_B_bias),
        (("network", "LayerNorm_0", "scale"), trunk_LN0_scale),
        (("network", "LayerNorm_0", "bias"), trunk_LN0_bias),
        (("network", "LayerNorm_1", "scale"), trunk_LN1_scale),
        (("network", "LayerNorm_1", "bias"), trunk_LN1_bias),
        (("Dense_0", "kernel"), mean_w),
        (("Dense_0", "bias"), mean_b),
        (("Dense_1", "kernel"), logstd_w),
        (("Dense_1", "bias"), logstd_b),
    ]
    for path, value in updates:
        current = actor
        for key in path:
            current = current[key]
        if value.shape != current.shape:
            raise ValueError(
                f"{merged_pt_path}: {'.'.join(path)} has shape {value.shape}; "
                f"the configured actor requires {current.shape}"
            )
    for path, value in updates:
        target = actor
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    params[actor_key] = actor
    if is_frozen:
        params = flax.core.freeze(params)

    new_state = agent.state.replace(params=params)
    return agent.replace(state=new_state)


def _normalized_ln_weights(source_pt_paths, weights):
    if not source_pt_paths:
        raise ValueError("Provide at least one LayerNorm source export")
    values = np.asarray(weights if weights is not None else [1.0] * len(source_pt_paths),
                        dtype=np.float64)
    if (values.shape != (len(source_pt_paths),) or not np.isfinite(values).all()
            or np.any(values < 0) or not np.any(values > 0)):
        raise ValueError("Provide one finite nonnegative LayerNorm weight per source, with a positive sum")
    values = values / values.max()
    return (values / values.sum()).tolist()


def _legacy_ln_weights(merged, source_pt_paths, explicit_weights):
    if explicit_weights is not None:
        return _normalized_ln_weights(source_pt_paths, explicit_weights)
    saved_weights = merged.get("task_weights")
    if saved_weights is None:
        return None
    weights = _normalized_ln_weights(source_pt_paths, saved_weights)
    if all(weight == weights[0] for weight in weights):
        return weights  # Uniform averaging is independent of source order.
    saved_sources = merged.get("source_checkpoints")
    canonical = lambda paths: [os.path.realpath(os.path.expanduser(str(path))) for path in paths]
    if saved_sources is None or canonical(source_pt_paths) != canonical(saved_sources):
        raise ValueError(
            "Cannot verify source order for this legacy merge's nonuniform task_weights. "
            "Supply sources in the recorded source_checkpoints order and at those paths, or "
            "recreate a self-contained export with quick_merge.py --checkpoint_paths "
            "ORIGINAL_TASK_EXPORTS --task_weights ORIGINAL_WEIGHTS using the original method. "
            "Python callers may instead provide weights_for_ln explicitly in source order."
        )
    return weights


def _average_ln_from_sources(
    source_pt_paths: List[str],
    weights: Optional[List[float]] = None,
):
    """Compute a weighted average of LayerNorm parameters from save_for_merge exports."""
    import torch

    weights = _normalized_ln_weights(source_pt_paths, weights)

    acc = None
    for path, w in zip(source_pt_paths, weights):
        if w == 0:
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        if "_jax_extras" not in d:
            raise KeyError(f"{path} has no _jax_extras; expected a save_for_merge export.")
        ex = d["_jax_extras"]
        layers = ("enc_LN", "trunk_LN_0", "trunk_LN_1")
        for layer in layers:
            if layer not in ex or any(key not in ex[layer] for key in ("scale", "bias")):
                raise ValueError(f"{path}: missing LayerNorm parameters for {layer}")
            for key in ("scale", "bias"):
                value = ex[layer][key]
                if (not torch.is_tensor(value) or not torch.is_floating_point(value)
                        or not torch.isfinite(value).all()):
                    raise ValueError(f"{path}: invalid LayerNorm tensor {layer}.{key}")
        if acc is None:
            acc = {
                layer: {key: torch.zeros_like(ex[layer][key]) for key in ("scale", "bias")}
                for layer in layers
            }
        for layer in layers:
            for key in ("scale", "bias"):
                value = ex[layer][key]
                if value.shape != acc[layer][key].shape:
                    raise ValueError(f"{path}: incompatible LayerNorm tensor {layer}.{key}")
                acc[layer][key] = acc[layer][key] + w * value
    return acc
