import logging
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from timm.models.vision_transformer import Mlp, PatchEmbed
from einops.layers.torch import Rearrange

from psi_policy.model.common.module_attr_mixin import ModuleAttrMixin

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
        
        
        self.vit_head = timm.create_model(
            model_name,
            pretrained=True,
            global_pool="",
            num_classes=0 # No classification head is used, output feature vectors
        )

        
        image_size = 224  
        crop_size = int(image_size * crop_ratio)
        
        self.train_transforms = T.Compose([
            # from [-1, 1] to [0, 1]
            T.Lambda(lambda x: (x + 1) / 2),
            T.RandomCrop(size=crop_size),
            T.Resize(size=image_size, antialias=True),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.4,
                saturation=0.2,
                hue=0.05
            ),
            # from [0, 1] to [-1, 1]
            T.Lambda(lambda x: x * 2 - 1),
        ])
        
        
        self.eval_transforms = T.Compose([
            # from [-1, 1] to [0, 1]
            T.Lambda(lambda x: (x + 1) / 2),
            T.Resize(size=image_size, antialias=True),
            # from [0, 1] to [-1, 1]
            T.Lambda(lambda x: x * 2 - 1),
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

        logger.info(
            "Number of parameters in obs encoder: %e", sum(p.numel() for p in self.parameters())
        )

        print(f"==================================== Using {model_name} with feature_dim={feature_dim} ====================================")

    def forward_head(self, rgb_data, training=True):
        B, T = rgb_data.shape[:2]
        rgb_data = rgb_data[:, :, :3]

        # B*T,3,H,W
        rgb_data = rgb_data.reshape(B*T, *rgb_data.shape[2:])

        
        if self.use_augmentation and training:
            
            rgb_data = self.train_transforms(rgb_data)
        else:
            
            rgb_data = self.eval_transforms(rgb_data)

        # B*T,num_patches,feature_dim
        rgb_feature = self.vit_head(rgb_data)[:, :1] # Remove CLS token

        # (B, T*num_patches, feature_dim)
        rgb_feature = rgb_feature.reshape(B, T*rgb_feature.shape[1], rgb_feature.shape[-1])  
        return rgb_feature

    def forward(self, obs_dict, training=True):
        embeddings = list()
        embeddings.append(self.forward_head(obs_dict['rgb_head'], training))  # (B,T*num_patches,feature_dim)
        embeddings.append(self.forward_head(obs_dict['rgb_left_hand'], training))  # (B,T*num_patches,feature_dim)
        embeddings.append(self.forward_head(obs_dict['rgb_right_hand'], training))  # (B,T*num_patches,feature_dim)
        state = torch.cat([obs_dict['qrcode_detected'], obs_dict['qrcode_detected'], obs_dict['qrcode_detected'], obs_dict['qrcode_detected']], dim=2)
        embeddings.append(state)  # (B,T,feature_dim)

        
        return torch.cat(embeddings, dim=2)  # (B,T*num_patches,feature_dim)

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
            
        example_obs_dict['rgb_head'] = example_obs_dict['rgb_head']
        example_obs_dict['rgb_left_hand'] = example_obs_dict['rgb_left_hand']
        example_obs_dict['rgb_right_hand'] = example_obs_dict['rgb_right_hand']

        head_feature = self.forward_head(example_obs_dict['rgb_head'], training=False)
        left_hand_feature = self.forward_head(example_obs_dict['rgb_left_hand'], training=False)
        right_hand_feature = self.forward_head(example_obs_dict['rgb_right_hand'], training=False)

        # state = torch.cat([example_obs_dict['right_arm_states'], example_obs_dict['left_arm_states'], example_obs_dict['right_hand_states'], example_obs_dict['left_hand_states'], example_obs_dict['waist_joints_states']], dim=2)
        state = torch.cat([example_obs_dict['qrcode_detected'], example_obs_dict['qrcode_detected'], example_obs_dict['qrcode_detected'], example_obs_dict['qrcode_detected']], dim=2)

        obs_part_length = [head_feature.shape[1]]
        example_embeddings = torch.cat([head_feature, left_hand_feature, right_hand_feature, state], dim=2)
        assert len(example_embeddings.shape) == 3
        assert example_embeddings.shape[0] == 1
        # assert example_embeddings.shape[1] == sum(obs_part_length)
        
        return example_embeddings.shape, obs_part_length