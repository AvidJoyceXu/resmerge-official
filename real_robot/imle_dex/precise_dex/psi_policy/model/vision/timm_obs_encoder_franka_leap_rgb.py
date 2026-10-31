import logging
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from timm.models.vision_transformer import Mlp, PatchEmbed
from einops.layers.torch import Rearrange

from psi_policy.model.common.module_attr_mixin import ModuleAttrMixin
from psi_policy.common.nested_dict_util import obs_keys_by_type

import os

logger = logging.getLogger(__name__)


feature_dim_dict = {
    
    'vit_small_r26_s32_224': 384, 
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
    
    
    'vit_tiny_patch16_224': 192,  
    'deit_tiny_patch16_224': 192,  
    'vit_tiny_patch16_384': 192,   
    'test_vit.r160_in1k': 192,     
    'mobilevit_xxs': 320,          
    'mobilevit_xs': 384,           
}

"""
The visual encoder extracts image features and concatenates them with proprioception before
projection.
"""
class TimmObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str = "vit_small_r26_s32_224",  
            use_augmentation: bool = True,  
            crop_ratio: float = 0.95,  
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        self.model_name = model_name
        self.use_augmentation = use_augmentation
        derived = obs_keys_by_type(shape_meta, "rgb")
        self.rgb_camera_keys = derived if derived else (
            "rgb_head", "rgb_left_hand", "rgb_right_hand"
        )

        
        self.vit_head = timm.create_model(
            model_name,
            pretrained=os.environ.get("RESMERGE_PRETRAINED", "1") == "1",
            global_pool="",
            num_classes=0 # No classification head is used, output feature vectors
        )

        
        image_size = 224  
        crop_size = int(image_size * crop_ratio)
        
        # Transforms expect image tensors in [0, 1]. We convert to/from [-1, 1]
        # in forward_head to be robust to whether inputs are already normalized.
        self.train_transforms = T.Compose([
            T.RandomCrop(size=crop_size),
            T.Resize(size=image_size, antialias=True),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.4,
                saturation=0.2,
                hue=0.05
            ),
        ])
        
        
        self.eval_transforms = T.Compose([
            T.Resize(size=image_size, antialias=True),
        ])

        
        if model_name in feature_dim_dict:
            feature_dim = feature_dim_dict[model_name]
        else:
            
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_output = self.vit_head(dummy_input)
                feature_dim = dummy_output.shape[-1]
                print(f"Auto-detected feature_dim for {model_name}: {feature_dim}")

        
        self.shape_meta = shape_meta
        self.feature_dim = feature_dim


        obs_shape_meta = shape_meta['obs']
        self.state_keys = [
            k for k, v in obs_shape_meta.items()
            if v.get('type') == 'low_dim'
        ]

        
        state_dim = 0
        for key in self.state_keys:
            state_dim += obs_shape_meta[key]['shape'][0]

        
        raw_output_dim = feature_dim * len(self.rgb_camera_keys) + state_dim  
        self.raw_output_dim = int(raw_output_dim)

        self.output_dim = 400  
        self.output_proj = nn.Linear(raw_output_dim, self.output_dim)

        print(f"Feature dim: {feature_dim}, State dim: {state_dim}, Raw output: {raw_output_dim}, Projected output: {self.output_dim}")


        logger.info(
            "Number of parameters in obs encoder: %e", sum(p.numel() for p in self.parameters())
        )

        print(f"==================================== Using {model_name} with feature_dim={feature_dim} ====================================")

    def forward_head(self, rgb_data, training=True):
        B, T = rgb_data.shape[:2]
        rgb_data = rgb_data[:, :, :3]

        # B*T,3,H,W
        rgb_data = rgb_data.reshape(B*T, *rgb_data.shape[2:])

        # Convert to [0, 1] for torchvision transforms, then convert back to [-1, 1].
        # Heuristic: if min < 0 it's assumed to be already in [-1, 1].
        x_min = float(rgb_data.min())
        if x_min < 0.0:
            rgb_01 = (rgb_data + 1.0) / 2.0
        else:
            rgb_01 = rgb_data
        rgb_01 = rgb_01.clamp(0.0, 1.0)

        # Apply image transforms
        if self.use_augmentation and training:
            
            rgb_01 = self.train_transforms(rgb_01)
        else:
            
            rgb_01 = self.eval_transforms(rgb_01)

        rgb_data = (rgb_01 * 2.0 - 1.0).clamp(-1.0, 1.0)

        # B*T,num_patches,feature_dim
        rgb_feature = self.vit_head(rgb_data)[:, :1]


        # (B, T, feature_dim)
        rgb_feature = rgb_feature.reshape(B, T*rgb_feature.shape[1], rgb_feature.shape[-1])  
        return rgb_feature

    def _embeddings_concat_pre_proj(self, obs_dict, training: bool = True) -> torch.Tensor:
        """
        Return the pre-projection concatenated embeddings with shape (B, T, raw_output_dim).
        """
        embeddings = []
        for k in self.rgb_camera_keys:
            embeddings.append(self.forward_head(obs_dict[k], training))  # (B,T,feature_dim)
        state = torch.cat([obs_dict[k] for k in self.state_keys], dim=2)  # (B, T, D_state)
        embeddings.append(state)  # (B,T,feature_dim)

        
        features = torch.cat(embeddings, dim=2)  # (B,T,n_cams*feature_dim + D_state) == (B,T,raw_output_dim)
        return features

    def forward(self, obs_dict, training=True):
        features = self._embeddings_concat_pre_proj(obs_dict, training)  # (B, T, raw_output_dim)
        features = self.output_proj(features)  # (B, T, 400)
        return features

    def forward_pre_projection(
        self,
        obs_dict,
        training: bool = True,
        action_base: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Expose pre-projection features for residual RL. Optionally concatenate action_base on
        the last axis; accepts (B, D) or (B, T, D).
        """
        out = self._embeddings_concat_pre_proj(obs_dict, training)  # (B, T, raw_output_dim)
        if action_base is None:
            return out
        ab = action_base
        if ab.ndim == 2:
            ab = ab[:, None, :].expand(-1, out.shape[1], -1)  # (B, T, D_ab)
        ab = ab.to(device=out.device, dtype=out.dtype)
        return torch.cat([out, ab], dim=-1)  # (B, T, raw_output_dim + D_ab)

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        T = None
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            T = attr['horizon']
            this_obs = torch.zeros(
                (1, T) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
            
        rgb_feats = [self.forward_head(example_obs_dict[k], training=False) for k in self.rgb_camera_keys]

        state = torch.cat([example_obs_dict[k] for k in self.state_keys], dim=2)

        obs_part_length = [rgb_feats[0].shape[1]]
        # obs_part_length = [head_feature.shape[1], state_feature.shape[1]]
        # example_embeddings = torch.cat([head_feature, left_hand_feature, right_hand_feature], dim=2)
        example_embeddings = torch.cat([*rgb_feats, state], dim=2)  # (B, T, raw_output_dim)
        example_embeddings = self.output_proj(example_embeddings)  
        assert len(example_embeddings.shape) == 3
        assert example_embeddings.shape[0] == 1
        # assert example_embeddings.shape[1] == sum(obs_part_length)
        
        return example_embeddings.shape, obs_part_length