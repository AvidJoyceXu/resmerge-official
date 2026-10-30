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

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from rlinf.models.embodiment.residual_policy.residual_policy import ResidualPolicy
from rlinf.models.embodiment.modules.utils import get_act_func, layer_init


class LoRAResidualPolicy(ResidualPolicy):
    """
    LoRA Residual Policy for residual SAC training.
    
    Uses LoRA (Low-Rank Adaptation) layers instead of full linear layers
    in the backbone network. Inherits from ResidualPolicy and only overrides
    `_compute_action_distribution` method.
    
    Supports all LoRA-related APIs for parameter management and merging.
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
        rank=16,  # LoRA rank
    ):
        # Initialize parent class but we'll override the backbone
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_action_chunks=num_action_chunks,
            add_value_head=add_value_head,
            add_q_head=add_q_head,
            q_head_type=q_head_type,
            actor_input=actor_input,
        )
        
        self.rank = rank
        
        # Remove the original backbone Sequential
        del self.backbone
        
        # Determine input dimension based on actor_input mode
        if actor_input == "obs":
            input_dim = obs_dim
        elif actor_input == "obs_base_action":
            input_dim = obs_dim + num_action_chunks * action_dim
        else:
            raise ValueError(f"Invalid actor_input: {actor_input}")
        
        # Get activation function
        activation = "tanh" # NOTE: or ReLU
        self.act = get_act_func(activation)()
        
        # LoRA layers: fc1: input_dim → 512
        self.fc1_A = nn.Linear(input_dim, rank, bias=False)
        self.fc1_B = nn.Linear(rank, 512)
        
        # LoRA layers: fc2: 512 → 512
        self.fc2_A = nn.Linear(512, rank, bias=False)
        self.fc2_B = nn.Linear(rank, 512)
        
        # LoRA layers: fc3: 512 → 256
        self.fc3_A = nn.Linear(512, rank, bias=False)
        self.fc3_B = nn.Linear(rank, 256)
        
        # Initialize LoRA weights
        self._init_lora_weights()
    
    def _init_lora_weights(self):
        """Initialize LoRA weights (A matrices with Kaiming, B matrices near zero)."""
        # A matrices use Kaiming initialization
        nn.init.kaiming_uniform_(self.fc1_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc3_A.weight, a=math.sqrt(5))
        
        # B matrices initialized near zero (so residual starts small)
        nn.init.normal_(self.fc1_B.weight, std=0.01)
        nn.init.normal_(self.fc2_B.weight, std=0.01)
        nn.init.normal_(self.fc3_B.weight, std=0.01)
        
        # Bias initialization
        if self.fc1_B.bias is not None:
            nn.init.zeros_(self.fc1_B.bias)
        if self.fc2_B.bias is not None:
            nn.init.zeros_(self.fc2_B.bias)
        if self.fc3_B.bias is not None:
            nn.init.zeros_(self.fc3_B.bias)
    
    def _compute_action_distribution(self, input_feat):
        """
        Compute action distribution from input features using LoRA layers.
        
        Args:
            input_feat: Input features tensor [B, input_dim]
        
        Returns:
            action_mean: [B, num_action_chunks, action_dim]
            action_logstd: [B, num_action_chunks, action_dim]
            probs: Normal distribution
            B: Batch size
        """
        # Forward through LoRA backbone
        x = self.act(self.fc1_B(self.fc1_A(input_feat)))
        x = self.act(self.fc2_B(self.fc2_A(x)))
        feat = self.act(self.fc3_B(self.fc3_A(x)))
        
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
    
    # ===== LoRA-related APIs =====
    
    def get_lora_parameters(self):
        """
        Get all LoRA parameters (for saving and subsequent merging).
        
        Returns:
            dict: Contains all layers' A and B matrices, and output layer parameters
        """
        params = {
            'lora_layers': {
                'fc1': {
                    'A': self.fc1_A.weight.data.clone().cpu(),
                    'B': self.fc1_B.weight.data.clone().cpu(),
                    'B_bias': self.fc1_B.bias.data.clone().cpu() if self.fc1_B.bias is not None else None
                },
                'fc2': {
                    'A': self.fc2_A.weight.data.clone().cpu(),
                    'B': self.fc2_B.weight.data.clone().cpu(),
                    'B_bias': self.fc2_B.bias.data.clone().cpu() if self.fc2_B.bias is not None else None
                },
                'fc3': {
                    'A': self.fc3_A.weight.data.clone().cpu(),
                    'B': self.fc3_B.weight.data.clone().cpu(),
                    'B_bias': self.fc3_B.bias.data.clone().cpu() if self.fc3_B.bias is not None else None
                }
            },
            'output_layers': {
                'fc_mean': {
                    'weight': self.actor_mean.weight.data.clone().cpu(),
                    'bias': self.actor_mean.bias.data.clone().cpu()
                },
                'fc_logstd': {
                    'weight': self.actor_logstd.weight.data.clone().cpu(),
                    'bias': self.actor_logstd.bias.data.clone().cpu()
                }
            },
            'meta': {
                'rank': self.rank,
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'num_action_chunks': self.num_action_chunks,
                'actor_input': self.actor_input
            }
        }
        return params
    
    def set_lora_parameters(self, params, device='cpu'):
        """
        Set LoRA parameters (for loading merged weights).
        
        Args:
            params: dict, format same as get_lora_parameters() return dict
            device: str, target device
        """
        device = torch.device(device)
        
        # Load LoRA layers
        for layer_name in ['fc1', 'fc2', 'fc3']:
            layer_params = params['lora_layers'][layer_name]
            
            # Load A matrix
            getattr(self, f'{layer_name}_A').weight.data = layer_params['A'].to(device)
            
            # Load B matrix
            getattr(self, f'{layer_name}_B').weight.data = layer_params['B'].to(device)
            
            # Load B's bias (if exists)
            if layer_params['B_bias'] is not None:
                getattr(self, f'{layer_name}_B').bias.data = layer_params['B_bias'].to(device)
        
        # Load output layers
        self.actor_mean.weight.data = params['output_layers']['fc_mean']['weight'].to(device)
        self.actor_mean.bias.data = params['output_layers']['fc_mean']['bias'].to(device)
        self.actor_logstd.weight.data = params['output_layers']['fc_logstd']['weight'].to(device)
        self.actor_logstd.bias.data = params['output_layers']['fc_logstd']['bias'].to(device)
    
    def get_parameter_count(self):
        """Count parameters (for verification)."""
        lora_params = sum([
            p.numel() for name, p in self.named_parameters() 
            if 'fc1_' in name or 'fc2_' in name or 'fc3_' in name
        ])
        output_params = sum([
            p.numel() for name, p in self.named_parameters() 
            if 'fc_mean' in name or 'fc_logstd' in name or 'actor_mean' in name or 'actor_logstd' in name
        ])
        total = sum(p.numel() for p in self.parameters())
        
        return {
            'total': total,
            'lora_layers': lora_params,
            'output_layers': output_params,
            'rank': self.rank
        }
    
    def save_for_merge(self, save_path, task_id, additional_info=None):
        """
        Save model for subsequent merging.
        
        Args:
            save_path: str, save path
            task_id: int, task ID
            additional_info: dict, additional info (e.g., training steps, performance)
        """
        save_dict = {
            'task_id': task_id,
            'params': self.get_lora_parameters(),
            'state_dict': self.state_dict(),  # Full state_dict as backup
            'additional_info': additional_info or {}
        }
        torch.save(save_dict, save_path)
        print(f"Saved LoRA parameters for task {task_id} to {save_path}")
    
    @classmethod
    def load_for_merge(cls, load_path):
        """
        Load model for merging.
        
        Args:
            load_path: str, model path
        
        Returns:
            dict: Dictionary containing parameters and metadata
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        return checkpoint

