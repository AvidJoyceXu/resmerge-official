import pickle
import inspect
import numpy as np
from typing import Dict, Tuple
from einops import rearrange, reduce
from scipy.optimize import linear_sum_assignment
import time
import torch
import torch.nn.functional as F

from psi_policy.common.pytorch_util import dict_apply
from psi_policy.policy.base_image_policy import BaseImagePolicy
from psi_policy.model.common.normalizer import LinearNormalizer
from psi_policy.model.diffusion.transformer_for_action import TransformerForAction
from psi_policy.model.vision.timm_obs_encoder_rgb_only import TimmObsEncoder
from psi_policy.model.common.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters


def noise_assignment(data, noise):
    device = data.device
    data, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (data, noise))
    dist = torch.cdist(data, noise)
    _, assign = linear_sum_assignment(dist.cpu())
    return torch.from_numpy(assign).to(device)

class PsiPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            obs_encoder: TimmObsEncoder,
            num_inference_steps=None,
            input_pertub=0.1,
            # arch
            n_layer=7,
            n_head=8,
            p_drop_attn=0.1,
            use_attn_mask=True,
            use_rotary_emb=True,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        
        obs_shape, obs_part_length = obs_encoder.output_shape()
        n_emb = obs_shape[-1]
        obs_tokens = obs_shape[-2]

        model = TransformerForAction(
            input_dim=action_dim,
            output_dim=action_dim,
            action_horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            max_cond_tokens=obs_tokens, # obs tokens + 1 token for time
            p_drop_attn=p_drop_attn,
            obs_part_length=obs_part_length,
            use_attn_mask=use_attn_mask,
            use_rotary_emb=use_rotary_emb,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.input_pertub = input_pertub
        self.kwargs = kwargs
    
    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], output_path: str = None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        
        # process input
        obs_tokens = self.obs_encoder(nobs, training=False)
        # (B, N, n_emb)
        
        noise = torch.randn(B, self.action_horizon, self.action_dim, device=self.device)

        pred_actions, _ = self.model(
            noise,
            cond=obs_tokens,
            training=False,
            gen_attn_map=False
        )

        # unnormalize prediction
        assert pred_actions.shape == (B, self.action_horizon, self.action_dim)
        pred_actions = self.normalizer['action'].unnormalize(pred_actions)

        result = {
            'action': pred_actions,
            'action_pred': pred_actions
        }
        return result

    def predict_action_all(
        self,
        obs_dict: Dict[str, torch.Tensor],
        output_path: str = None,
        prev_traj: torch.Tensor = None,
        latency_step: int = None,
        num_candidates: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Return normalized and unnormalized action candidates and heuristic_scores (negative
        distance; larger is better). action/action_pred contain the selected unnormalized
        candidate.
        """
        assert 'past_action' not in obs_dict  # not implemented yet
        nobs = self.normalizer.normalize(obs_dict)
        obs_tokens = self.obs_encoder(nobs, training=False).repeat_interleave(32, dim=0)
        noise = torch.randn(32, self.action_horizon, self.action_dim, device=self.device)

        pred_actions_normalized, _ = self.model(
            noise,
            cond=obs_tokens,
            training=False,
            gen_attn_map=False
        )

        prev_traj_end = prev_traj[:, 8 + latency_step:].reshape(1, -1)
        gen_traj_start = pred_actions_normalized[:, :self.action_horizon - (8 + latency_step)].reshape(32, -1)

        distances = torch.cdist(gen_traj_start, prev_traj_end).reshape(-1)
        heuristic_scores = -distances

        
        if num_candidates is not None and int(num_candidates) > 0:
            k = min(int(num_candidates), int(pred_actions_normalized.shape[0]))
            topk_idx = torch.topk(heuristic_scores, k=k, largest=True).indices
            pred_actions_normalized = pred_actions_normalized[topk_idx]
            heuristic_scores = heuristic_scores[topk_idx]
            distances = distances[topk_idx]

        min_idx = torch.argmin(distances)

        
        selected_action_normalized = pred_actions_normalized[min_idx:min_idx + 1]
        all_actions_unnormalized = self.normalizer['action'].unnormalize(pred_actions_normalized)
        selected_action = self.normalizer['action'].unnormalize(selected_action_normalized)

        result = {
            'action': selected_action,
            'action_pred': selected_action,
            'all_actions_normalized': pred_actions_normalized,
            'all_actions_unnormalized': all_actions_unnormalized,
            'heuristic_scores': heuristic_scores,
            'chosen_idx': min_idx
        }
        return result

    def bidirection_predict_action(self, obs_dict: Dict[str, torch.Tensor], output_path: str = None, prev_traj: torch.Tensor = None, latency_step: int = None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        
        # process input
        # obs_tokens = self.obs_encoder(nobs, training=False)
        obs_tokens = self.obs_encoder(nobs, training=False).repeat_interleave(32, dim=0)
        # (B, N, n_emb)
        
        noise = torch.randn(32, self.action_horizon, self.action_dim, device=self.device)

        pred_actions, _ = self.model(
            noise,
            cond=obs_tokens,
            training=False,
            gen_attn_map=False
        )

        prev_traj_end = prev_traj[:,8+latency_step:].reshape(1,-1)
        gen_traj_start = pred_actions[:,:self.action_horizon-(8+latency_step)].reshape(32,-1)

        distances = torch.cdist(gen_traj_start,prev_traj_end)
        min_dist, min_idx = distances.min(dim=0)
        pred_actions = pred_actions[min_idx]

        # unnormalize prediction
        pred_actions = self.normalizer['action'].unnormalize(pred_actions)

        result = {
            'action': pred_actions,
            'action_pred': pred_actions
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            is_ckpt_dit_lora: bool,
            is_ckpt_obs_encoder_lora: bool,
            start_ckpt_path: str,
            start_normalizer_path: str,
            lr: float,
            weight_decay: float,
            betas: Tuple[float, float],
            obs_encoder_mode: str,
            obs_encoder_lora_cfg: dict,
            dit_mode: str,
            dit_lora_cfg: dict
        ) -> torch.optim.Optimizer:

        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        print(f"Fused AdamW available: {fused_available}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas, fused=fused_available
        )
        return optimizer

    def rs_imle_loss(self, real_samples, fake_samples, epsilon=0.03):
        B, T, D = real_samples.shape
        n_samples = fake_samples.shape[1]

        real_flat = real_samples.reshape(B, 1, -1)
        fake_flat = fake_samples.reshape(B, n_samples, -1)

        distances = torch.cdist(real_flat, fake_flat).squeeze(1)

        valid_samples = (distances > epsilon).float()
        # wandb.log({"max_distance": distances.max().item(), "min_distance": distances.min().item(), "mean_distance": distances.mean().item(), "epsilon": epsilon})
        min_distances, _ = (distances + (1 - valid_samples) * distances.max()).min(dim=1)
        valid_real_samples = (min_distances < distances.max()).float()
        if valid_real_samples.sum() > 0:
            loss = (min_distances * valid_real_samples).sum() / valid_real_samples.sum()
        else:
            loss = torch.tensor(0.0, device=real_samples.device)

        return loss

    def compute_loss(self, batch, training=True):
        # normalize input
        assert 'valid_mask' not in batch

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        B = nactions.shape[0]

        n_samples_per_condition = 20

        # process input
        obs_tokens = self.obs_encoder(nobs, training)
        # (B, N, n_emb)

        noise = torch.randn(B * n_samples_per_condition, *nactions.shape[1:], device=self.device)
        repeated_obs_cond = obs_tokens.repeat_interleave(n_samples_per_condition, dim=0)

        pred_actions, _ = self.model(
            noise,
            cond=repeated_obs_cond,
            training=training,
            gen_attn_map=False
        )

        pred_actions = pred_actions.reshape(B, n_samples_per_condition, *nactions.shape[1:])

        loss = self.rs_imle_loss(nactions, pred_actions, epsilon=0.03)

        return loss

    def forward(self, batch, training=True):
        return self.compute_loss(batch, training)
