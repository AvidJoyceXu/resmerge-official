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

import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from rlinf.data.replay_buffer import SACReplayBuffer
from rlinf.hybrid_engines.fsdp import FSDP, FSDPModule
from rlinf.scheduler import Channel
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict, compute_rollout_metrics
from rlinf.utils.nested_dict_process import (
    concat_batch,
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class ResidualSACFSDPPolicy(EmbodiedSACFSDPPolicy):
    """
    SAC Policy Worker with residual policy support.
    
    Supports different action construction modes:
    - critic_input: "res", "sum", "concat"
    - actor_input: "obs", "obs_base_action"
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        
        # Residual policy configuration
        self.residual_cfg = cfg.get("residual_policy", {})
        self.use_residual = self.residual_cfg.get("enabled", False)
        
        if self.use_residual:
            self.res_scale = self.residual_cfg.get("res_scale", 0.1)
            self.actor_input = cfg.get("network", {}).get("actor_input", "obs")
            self.critic_input = cfg.get("network", {}).get("critic_input", "res")
            
            print(f"[ResidualSACFSDPPolicy] Residual policy enabled")
            print(f"[ResidualSACFSDPPolicy] actor_input={self.actor_input}, critic_input={self.critic_input}")
            print(f"[ResidualSACFSDPPolicy] res_scale={self.res_scale}")

    def _build_action_for_critic(self, residual_action, base_action=None):
        """
        Build action for critic based on critic_input mode.
        
        Args:
            residual_action: Residual action [B, num_chunks * action_dim]
            base_action: Base action [B, num_chunks * action_dim] (optional)
        
        Returns:
            action: Action for critic [B, num_chunks * action_dim] or [B, 2 * num_chunks * action_dim]
        """
        if self.critic_input == "res":
            return residual_action
        elif self.critic_input == "sum":
            if base_action is None:
                raise ValueError("base_action must be provided when critic_input=='sum'")
            scaled_residual = self.res_scale * residual_action
            return base_action + scaled_residual
        elif self.critic_input == "concat":
            if base_action is None:
                raise ValueError("base_action must be provided when critic_input=='concat'")
            return torch.cat([residual_action, base_action], dim=-1)
        else:
            raise ValueError(f"Invalid critic_input: {self.critic_input}")

    def _extract_actions_from_batch(self, batch):
        """
        Extract residual_action, base_action, base_next_action from batch.
        
        Returns:
            residual_action: [B, num_chunks * action_dim]
            base_action: [B, num_chunks * action_dim] or None
            base_next_action: [B, num_chunks * action_dim] or None
        """
        if self.critic_input == "res" and self.actor_input == "obs":
            # Only residual action is stored
            residual_action = batch["action"] if "action" in batch else batch["action_tokens"]
            return residual_action, None, None
        else:
            # [residual_action, base_action, base_next_action] is stored
            if "residual_action" in batch:
                residual_action = batch["residual_action"]
                base_action = batch.get("base_action", None)
                base_next_action = batch.get("base_next_action", None)
            elif "action" in batch:
                # Fallback: assume action contains all components
                action = batch["action"]
                action_dim = self.cfg.actor.model.action_dim
                num_chunks = self.cfg.actor.model.num_action_chunks
                chunk_action_dim = num_chunks * action_dim
                
                residual_action = action[:, :chunk_action_dim]
                if action.shape[1] >= 2 * chunk_action_dim:
                    base_action = action[:, chunk_action_dim:2*chunk_action_dim]
                    if action.shape[1] >= 3 * chunk_action_dim:
                        base_next_action = action[:, 2*chunk_action_dim:3*chunk_action_dim]
                    else:
                        base_next_action = None
                else:
                    base_action = None
                    base_next_action = None
            else:
                raise ValueError("Cannot find action in batch")
            
            return residual_action, base_action, base_next_action

    def forward_critic(self, batch):
        """Forward pass for critic with residual policy support."""
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        rewards = batch["rewards"].to(self.torch_dtype)
        terminations = batch["terminations"].to(self.torch_dtype)

        curr_obs = batch["transitions"]["obs"]
        next_obs = batch["transitions"]["next_obs"]
        
        # Extract actions from batch
        curr_residual_action, curr_base_action, _ = self._extract_actions_from_batch(batch)
        
        with torch.no_grad():
            kwargs = {}
            if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
                kwargs["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
            
            # Get next state residual action
            if self.actor_input == "obs":
                next_residual_action, next_state_log_pi, shared_feature = self.model(
                    "sac_forward", obs=next_obs, **kwargs
                )
            elif self.actor_input == "obs_base_action":
                # Need base_next_action for actor input
                _, _, next_base_action = self._extract_actions_from_batch(batch)
                if next_base_action is None:
                    raise ValueError("base_next_action must be provided when actor_input=='obs_base_action'")
                next_residual_action, next_state_log_pi, shared_feature = self.model(
                    "sac_forward", obs=next_obs, base_action=next_base_action, **kwargs
                )
            else:
                raise ValueError(f"Invalid actor_input: {self.actor_input}")
            
            # Sum over both action_dim and num_action_chunks dimensions
            # chunk_logprobs shape: [B, num_action_chunks, action_dim]
            # Reshape to [B, num_action_chunks * action_dim] then sum to [B, 1]
            num_action_chunks = self.cfg.actor.model.num_action_chunks
            action_dim = self.cfg.actor.model.action_dim
            next_state_log_pi = next_state_log_pi.reshape(-1, num_action_chunks * action_dim).sum(dim=-1, keepdim=True)
            
            # Build next state action for Q evaluation
            _, _, next_base_action = self._extract_actions_from_batch(batch)
            next_state_action = self._build_action_for_critic(
                next_residual_action, next_base_action
            )
            
            if not use_crossq:
                all_qf_next_target = self.target_model(
                    "sac_q_forward",
                    obs=next_obs,
                    actions=next_state_action,
                    shared_feature=shared_feature,
                )
                if self.critic_subsample_size > 0:
                    sample_idx = torch.randint(
                        0,
                        all_qf_next_target.shape[-1],
                        (self.critic_subsample_size,),
                        generator=self.critic_sample_generator,
                        device=self.device,
                    )
                    all_qf_next_target = all_qf_next_target.index_select(
                        dim=-1, index=sample_idx
                    )

                if agg_q == "min":
                    qf_next_target, _ = torch.min(
                        all_qf_next_target, dim=1, keepdim=True
                    )
                elif agg_q == "mean":
                    qf_next_target = torch.mean(all_qf_next_target, dim=1, keepdim=True)

                if self.cfg.algorithm.get("backup_entropy", True):
                    qf_next_target = qf_next_target - self.alpha * next_state_log_pi
                    qf_next_target = qf_next_target.to(dtype=self.torch_dtype)
                if bootstrap_type == "always":
                    target_q_values = (
                        rewards.sum(dim=-1, keepdim=True)
                        + self.cfg.algorithm.gamma * qf_next_target
                    )  # [bsz, 1]
                elif bootstrap_type == "standard":
                    target_q_values = (
                        rewards.sum(dim=-1, keepdim=True)
                        + (~(terminations.any(dim=-1, keepdim=True)))
                        * self.cfg.algorithm.gamma
                        * qf_next_target
                    )  # [bsz, 1]
                else:
                    raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        # Build current state action for Q evaluation
        current_action = self._build_action_for_critic(
            curr_residual_action, curr_base_action
        )

        if not use_crossq:
            all_data_q_values = self.model(
                "sac_q_forward",
                obs=curr_obs,
                actions=current_action,
            )
        else:
            all_data_q_values, all_qf_next = self.model(
                "crossq_q_forward",
                obs=curr_obs,
                actions=current_action,
                next_obs=next_obs,
                next_actions=next_state_action,
            )

            all_qf_next = all_qf_next.detach()
            if agg_q == "min":
                qf_next, _ = torch.min(all_qf_next, dim=1, keepdim=True)
            elif agg_q == "mean":
                qf_next = torch.mean(all_qf_next, dim=1, keepdim=True)
            if self.cfg.algorithm.get("backup_entropy", True):
                qf_next = qf_next - self.alpha * next_state_log_pi
                qf_next = qf_next.to(dtype=self.torch_dtype)

            if bootstrap_type == "always":
                target_q_values = (
                    rewards.sum(dim=-1, keepdim=True)
                    + self.cfg.algorithm.gamma * qf_next
                )  # [bsz, 1]
            elif bootstrap_type == "standard":
                target_q_values = (
                    rewards.sum(dim=-1, keepdim=True)
                    + (~(terminations.any(dim=-1, keepdim=True)))
                    * self.cfg.algorithm.gamma
                    * qf_next
                )  # [bsz, 1]
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        return critic_loss

    def forward_actor(self, batch):
        """Forward pass for actor with residual policy support."""
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        curr_obs = batch["transitions"]["obs"]
        
        # Extract base_action if needed
        _, base_action, _ = self._extract_actions_from_batch(batch)
        
        kwargs = {}
        if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
            kwargs["temperature"] = self.cfg.algorithm.sampling_params.temperature_train
        
        # Get residual action from policy
        if self.actor_input == "obs":
            pi_residual, log_pi, shared_feature = self.model(
                "sac_forward", obs=curr_obs, **kwargs
            )
        elif self.actor_input == "obs_base_action":
            if base_action is None:
                raise ValueError("base_action must be provided when actor_input=='obs_base_action'")
            pi_residual, log_pi, shared_feature = self.model(
                "sac_forward", obs=curr_obs, base_action=base_action, **kwargs
            )
        else:
            raise ValueError(f"Invalid actor_input: {self.actor_input}")
        
        # Sum over both action_dim and num_action_chunks dimensions
        # chunk_logprobs shape: [B, num_action_chunks, action_dim]
        # Reshape to [B, num_action_chunks * action_dim] then sum to [B, 1]
        num_action_chunks = self.cfg.actor.model.num_action_chunks
        action_dim = self.cfg.actor.model.action_dim
        log_pi = log_pi.reshape(-1, num_action_chunks * action_dim).sum(dim=-1, keepdim=True)
        
        # Build action for Q evaluation
        pi_action = self._build_action_for_critic(pi_residual, base_action)
        
        if not use_crossq:
            all_qf_pi = self.model(
                "sac_q_forward",
                obs=curr_obs,
                actions=pi_action,
                shared_feature=shared_feature,
                detach_encoder=True,
            )
        else:
            all_qf_pi, _ = self.model(
                "crossq_q_forward",
                obs=curr_obs,
                actions=pi_action,
                next_obs=None,
                next_actions=None,
                shared_feature=shared_feature,
                detach_encoder=True,
            )

        if agg_q == "min":
            qf_pi, _ = torch.min(all_qf_pi, dim=1, keepdim=True)
        elif agg_q == "mean":
            qf_pi = torch.mean(all_qf_pi, dim=1, keepdim=True)
        actor_loss = ((self.alpha * log_pi) - qf_pi).mean()

        entropy = -log_pi.mean()
        return actor_loss, entropy

    def forward_alpha(self, batch):
        """Forward pass for alpha with residual policy support."""
        curr_obs = batch["transitions"]["obs"]
        
        # Extract base_action if needed
        _, base_action, _ = self._extract_actions_from_batch(batch)
        
        with torch.no_grad():
            kwargs = {}
            if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
                kwargs["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
            
            if self.actor_input == "obs":
                _, log_pi, _ = self.model("sac_forward", obs=curr_obs, **kwargs)
            elif self.actor_input == "obs_base_action":
                if base_action is None:
                    raise ValueError("base_action must be provided when actor_input=='obs_base_action'")
                _, log_pi, _ = self.model(
                    "sac_forward", obs=curr_obs, base_action=base_action, **kwargs
                )
            else:
                raise ValueError(f"Invalid actor_input: {self.actor_input}")
            
            # Sum over both action_dim and num_action_chunks dimensions
            # chunk_logprobs shape: [B, num_action_chunks, action_dim]
            # Reshape to [B, num_action_chunks * action_dim] then sum to [B, 1]
            num_action_chunks = self.cfg.actor.model.num_action_chunks
            action_dim = self.cfg.actor.model.action_dim
            log_pi = log_pi.reshape(-1, num_action_chunks * action_dim).sum(dim=-1, keepdim=True)

        alpha = self.compute_alpha()
        alpha_loss = -alpha * (log_pi.mean() + self.target_entropy)
        return alpha_loss

