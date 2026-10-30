# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.residual_policy.residual_policy import ResidualPolicy
from rlinf.models.embodiment.residual_policy.lora_residual_policy import LoRAResidualPolicy
from rlinf.models.embodiment.residual_policy.merge_lora_actors import (
    RobustMergeLoRA,
    RobustMergeLoRAOptimized
)


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    # Get model_type to determine which policy class to use
    model_type = cfg.get("model_type", "residual_policy")
    
    # Get actor_input from network config if available, otherwise from model config
    actor_input = cfg.get("actor_input", None)
    if actor_input is None:
        # Try to get from parent config (network.actor_input)
        actor_input = "obs"  # default
    
    # Determine which policy class to instantiate
    if model_type == "lora_residual_policy":
        rank = cfg.get("lora_rank", 16)
        model = LoRAResidualPolicy(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            add_value_head=cfg.get("add_value_head", False),
            add_q_head=cfg.get("add_q_head", True),
            q_head_type=cfg.get("q_head_type", "default"),
            actor_input=actor_input,
            rank=rank,
        )
    else:
        model = ResidualPolicy(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            add_value_head=cfg.get("add_value_head", False),
            add_q_head=cfg.get("add_q_head", True),
            q_head_type=cfg.get("q_head_type", "default"),
            actor_input=actor_input,
        )

    # Optional warm start. When model_path points at an RLinf residual checkpoint
    # directory, its weights are loaded into the freshly built module; an empty or
    # absent model_path keeps the historical behaviour (random init), so configs
    # that do not set it are unaffected.
    #
    # Without this, a caller that sets model_path and expects "load this
    # checkpoint" silently gets a randomly initialised policy instead.
    model_path = cfg.get("model_path", None)
    if model_path:
        from pathlib import Path

        from safetensors.torch import load_file

        path = Path(model_path)
        shards = sorted(path.glob("*.safetensors")) if path.is_dir() else []
        if not shards:
            raise FileNotFoundError(
                f"residual model_path={model_path} contains no .safetensors to load"
            )

        state_dict = {}
        for shard in shards:
            state_dict.update(load_file(str(shard)))

        target_dtype = next(model.parameters()).dtype
        state_dict = {
            k: (v.to(target_dtype) if v.is_floating_point() else v)
            for k, v in state_dict.items()
        }

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # quick_merge outputs carry no q_head: it is a training-time critic and is
        # fine to start fresh. Anything else missing means a genuine mismatch.
        missing = [k for k in missing if not k.startswith("q_head.")]
        if missing or unexpected:
            raise RuntimeError(
                f"residual checkpoint {model_path} does not match the configured model.\n"
                f"  missing: {sorted(missing)}\n  unexpected: {sorted(unexpected)}"
            )

    return model


__all__ = [
    "ResidualPolicy",
    "LoRAResidualPolicy",
    "get_model",
    "RobustMergeLoRA",
    "RobustMergeLoRAOptimized"
]
