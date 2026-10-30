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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.modules.q_head import MultiCrossQHead, MultiQHead
from rlinf.models.embodiment.modules.utils import get_act_func, layer_init
from rlinf.models.embodiment.modules.value_head import ValueHead


class ResidualPolicy(BasePolicy):
    """
    Residual Policy for residual SAC training.
    
    Supports two input modes:
    - "obs": only observation
    - "obs_base_action": observation + base action
    
    Outputs residual action that will be added to base action:
    final_action = base_action + res_scale * residual_action
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        num_action_chunks,
        add_value_head=False,
        add_q_head=True,
        q_head_type="default",
        actor_input="obs",  # "obs" or "obs_base_action"
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.actor_input = actor_input

        # For SAC, we need Q head
        self.independent_std = False
        self.final_tanh = True
        self.logstd_range = (-5, 2)
        action_scale = (-1, 1)
        
        assert add_value_head + add_q_head <= 1
        if add_value_head:
            self.value_head = ValueHead(
                obs_dim, hidden_sizes=(256, 256, 256), activation="tanh"
            )
        if add_q_head:
            # Q-head needs to handle chunk actions: [B, num_chunks * action_dim]
            chunk_action_dim = num_action_chunks * action_dim
            if q_head_type == "default":
                self.q_head = MultiQHead(
                    hidden_size=obs_dim,
                    hidden_dims=[256, 256, 256],
                    num_q_heads=2,
                    action_feature_dim=chunk_action_dim,
                )
            elif q_head_type == "crossq":
                self.q_head = MultiCrossQHead(
                    hidden_size=obs_dim,
                    hidden_dims=[256, 256, 256],
                    num_q_heads=2,
                    action_feature_dim=chunk_action_dim,
                )
            else:
                raise ValueError(f"Invalid q_head_type: {q_head_type}")

        activation = "tanh"
        act = get_act_func(activation)

        # Determine input dimension based on actor_input mode
        # Note: base_action is chunk action: [B, num_chunks * action_dim]
        if actor_input == "obs":
            input_dim = obs_dim
        elif actor_input == "obs_base_action":
            input_dim = obs_dim + num_action_chunks * action_dim
        else:
            raise ValueError(f"Invalid actor_input: {actor_input}")

        # Backbone network
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(input_dim, 512)),
            act(),
            layer_init(nn.Linear(512, 512)),
            act(),
            layer_init(nn.Linear(512, 256)),
            act(),
        )
        
        # Actor head: outputs residual action
        # Output shape: [B, num_action_chunks * action_dim]
        self.actor_mean = layer_init(
            nn.Linear(256, num_action_chunks * action_dim), std=0.01 * np.sqrt(2)
        )
        self.actor_logstd = nn.Linear(256, num_action_chunks * action_dim)

        # Action scaling (for tanh transformation)
        l, h = action_scale
        self.register_buffer(
            "action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32)
        )

    def preprocess_env_obs(self, env_obs):
        """Preprocess environment observation for residual policy.
        
        Args:
            env_obs: Raw environment observation dict
            
        Returns:
            Processed observation dict with states on the correct device
        """
        device = next(self.parameters()).device
        processed_obs = {}
        if isinstance(env_obs["rl_flatten_obs"], torch.Tensor):
            processed_obs["rl_flatten_obs"] = env_obs["rl_flatten_obs"].to(device)
        else:
            processed_obs["rl_flatten_obs"] = torch.from_numpy(env_obs["rl_flatten_obs"]).to(device)
        return processed_obs

    def forward(self, forward_type="default_forward", **kwargs):
        if forward_type == "sac_forward":
            return self.sac_forward(**kwargs)
        elif forward_type == "sac_q_forward":
            return self.sac_q_forward(**kwargs)
        elif forward_type == "crossq_forward":
            return self.crossq_forward(**kwargs)
        elif forward_type == "crossq_q_forward":
            return self.crossq_q_forward(**kwargs)
        elif forward_type == "default_forward":
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def _compute_action_distribution(self, input_feat):
        """
        Compute action distribution from input features.
        
        Args:
            input_feat: Input features tensor [B, input_dim]
        
        Returns:
            action_mean: [B, num_action_chunks, action_dim]
            action_logstd: [B, num_action_chunks, action_dim]
            probs: Normal distribution
            B: Batch size
        """
        # Forward through backbone
        feat = self.backbone(input_feat)
        
        # Get action mean and logstd
        action_mean = self.actor_mean(feat)  # [B, num_action_chunks * action_dim]
        action_logstd = self.actor_logstd(feat)  # [B, num_action_chunks * action_dim]
        
        # Apply tanh and scale to logstd_range
        action_logstd = torch.tanh(action_logstd)
        action_logstd = self.logstd_range[0] + 0.5 * (
            self.logstd_range[1] - self.logstd_range[0]
        ) * (action_logstd + 1)
        
        # Reshape for chunk processing
        B = action_mean.shape[0]
        action_mean = action_mean.reshape(B, self.num_action_chunks, self.action_dim)
        action_logstd = action_logstd.reshape(B, self.num_action_chunks, self.action_dim)
        
        # Compute distribution
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        return action_mean, action_logstd, probs, B

    def sac_forward(self, obs, base_action=None, **kwargs):
        """
        Forward pass for SAC training.
        
        Args:
            obs: Observation dict with "rl_flatten_obs" key, shape [B, obs_dim]
            base_action: Base action (only needed if actor_input == "obs_base_action")
                        shape [B, num_action_chunks * action_dim]
        
        Returns:
            residual_action: [B, num_action_chunks * action_dim]
            chunk_logprobs: [B, num_action_chunks, action_dim]
            shared_feature: None (for compatibility)
        """
        # Build input based on actor_input mode
        rl_flatten_obs = obs["rl_flatten_obs"]
        if self.actor_input == "obs":
            if base_action is not None:
                raise ValueError("base_action should not be provided when actor_input=='obs'")
            input_feat = rl_flatten_obs
        elif self.actor_input == "obs_base_action":
            if base_action is None:
                raise ValueError("base_action must be provided when actor_input=='obs_base_action'")
            input_feat = torch.cat([rl_flatten_obs, base_action], dim=-1)
        else:
            raise ValueError(f"Invalid actor_input: {self.actor_input}")

        # Compute action distribution
        _, _, probs, B = self._compute_action_distribution(input_feat)
        raw_action = probs.rsample()  # [B, num_action_chunks, action_dim]

        # Apply tanh transformation
        action_normalized = torch.tanh(raw_action)
        residual_action = action_normalized * self.action_scale + self.action_bias
        # Flatten back: [B, num_action_chunks * action_dim]
        residual_action = residual_action.reshape(B, -1)

        # Compute log probabilities (with tanh correction)
        chunk_logprobs = probs.log_prob(raw_action)  # [B, num_action_chunks, action_dim]
        chunk_logprobs = chunk_logprobs - torch.log(
            self.action_scale * (1 - action_normalized.pow(2)) + 1e-6
        )

        return residual_action, chunk_logprobs, None

    def default_forward(
        self,
        data,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
        **kwargs,
    ):
        obs = data["obs"]
        action = data["action"]
        base_action = data.get("base_action", None)

        # Build input
        if self.actor_input == "obs":
            input_feat = obs
        elif self.actor_input == "obs_base_action":
            if base_action is None:
                raise ValueError("base_action must be provided in data when actor_input=='obs_base_action'")
            input_feat = torch.cat([obs, base_action], dim=-1)
        else:
            raise ValueError(f"Invalid actor_input: {self.actor_input}")

        # Compute action distribution
        _, _, probs, B = self._compute_action_distribution(input_feat)
        action = action.reshape(B, self.num_action_chunks, self.action_dim)

        output_dict = {}
        if compute_logprobs:
            logprobs = probs.log_prob(action)
            output_dict.update(logprobs=logprobs)
        if compute_entropy:
            entropy = probs.entropy()
            output_dict.update(entropy=entropy)
        if compute_values:
            if getattr(self, "value_head", None):
                values = self.value_head(obs)
                output_dict.update(values=values)
            else:
                raise NotImplementedError
        return output_dict

    def predict_action_batch(
        self,
        env_obs,
        base_action=None,
        calulate_logprobs=True,
        calulate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        """
        Predict residual action batch.
        
        Args:
            env_obs: Environment observation dict
            base_action: Base action (optional, shape [B, num_action_chunks * action_dim])
            mode: "train" or "eval"
        
        Returns:
            chunk_actions: [B, num_action_chunks, action_dim] (numpy)
            result: dict with prev_logprobs, prev_values, forward_inputs
        """
        # Build input
        rl_flatten_obs = env_obs["rl_flatten_obs"]
        if self.actor_input == "obs":
            input_feat = rl_flatten_obs
        elif self.actor_input == "obs_base_action":
            if base_action is None:
                raise ValueError("base_action must be provided when actor_input=='obs_base_action'")
            input_feat = torch.cat([rl_flatten_obs, base_action], dim=-1)
        else:
            raise ValueError(f"Invalid actor_input: {self.actor_input}")

        # Compute action distribution
        action_mean, _, probs, B = self._compute_action_distribution(input_feat)

        if mode == "train":
            raw_action = probs.sample()
        elif mode == "eval":
            raw_action = action_mean.clone()
        else:
            raise NotImplementedError(f"{mode=}")

        chunk_logprobs = probs.log_prob(raw_action)

        # Apply tanh transformation
        action_normalized = torch.tanh(raw_action)
        residual_action = action_normalized * self.action_scale + self.action_bias

        chunk_logprobs = chunk_logprobs - torch.log(
            self.action_scale * (1 - action_normalized.pow(2)) + 1e-6
        )

        chunk_actions = residual_action.cpu().numpy()

        rl_flatten_obs = env_obs["rl_flatten_obs"]
        if hasattr(self, "value_head") and calulate_values:
            chunk_values = self.value_head(rl_flatten_obs)
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        forward_inputs = {"action": residual_action.reshape(B, -1)}
        if return_obs:
            forward_inputs["obs"] = rl_flatten_obs

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result

    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        """Q network forward for residual actions."""
        rl_flatten_obs = obs["rl_flatten_obs"]
        return self.q_head(rl_flatten_obs, actions)

    def crossq_q_forward(
        self,
        obs,
        actions,
        next_obs=None,
        next_actions=None,
        shared_feature=None,
        detach_encoder=False,
    ):
        """CrossQ forward for residual actions."""
        return self.q_head(
            obs["rl_flatten_obs"],
            actions,
            next_state_features=next_obs["rl_flatten_obs"] if next_obs is not None else None,
            next_action_features=next_actions,
        )

    def crossq_forward(self, obs, base_action=None, **kwargs):
        """CrossQ forward, same as sac_forward."""
        return self.sac_forward(obs, base_action=base_action, **kwargs)

