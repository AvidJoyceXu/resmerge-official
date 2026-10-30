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

"""Frozen visual encoder for RGB-based residual observations.

The residual policy in simulation originally consumed privileged simulator
state (object-to-end-effector relative poses). This module provides the
image-based replacement used by the RGB observation mode, mirroring the
real-robot setup: a single third-person camera frame is passed through a
frozen ViT to obtain a `feature_dim`-dimensional embedding, and two
consecutive frames are concatenated.

The encoder is never trained: parameters are frozen and every forward runs
under ``torch.no_grad()``, so it acts as a fixed part of the observation
function rather than as part of the policy.
"""

from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet statistics, matching the preprocessing DINOv2 was trained with.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_VISUAL_ENCODER = "facebook/dinov2-small"


class FrozenVisualEncoder(nn.Module):
    """Frozen ViT image encoder producing one feature vector per frame.

    Args:
        model_path: HuggingFace model id or local path of the ViT to load.
        image_size: Spatial resolution the images are resized to before encoding.
        pooling: ``"cls"`` uses the class token, ``"mean"`` averages the patch
            tokens. The real-robot setup uses the class token.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_VISUAL_ENCODER,
        image_size: int = 224,
        pooling: str = "cls",
    ):
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"Invalid pooling: {pooling}")

        from transformers import AutoModel

        self.model_path = model_path
        self.image_size = image_size
        self.pooling = pooling

        self.vit = AutoModel.from_pretrained(model_path)
        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False

        self._feature_dim = self.vit.config.hidden_size

        self.register_buffer(
            "pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    @property
    def feature_dim(self) -> int:
        """Feature width of a single frame."""
        return self._feature_dim

    def train(self, mode: bool = True):
        # The encoder is frozen: keep it in eval mode regardless of the
        # surrounding module's train/eval state so norm layers stay fixed.
        return super().train(False)

    def _preprocess(self, images: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert raw ``[B, H, W, C]`` uint8 frames to normalized ``[B, 3, S, S]``."""
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(np.ascontiguousarray(images))
        # Some base-model preprocessors rewrite frames as [B, N_IMG, H, W, C];
        # a single view is equivalent to [B, H, W, C].
        if images.ndim == 5 and images.shape[1] == 1:
            images = images[:, 0]
        if images.ndim != 4:
            raise ValueError(
                f"Expected images of shape [B, H, W, C], got {tuple(images.shape)}"
            )

        # uint8 frames arrive in [0, 255]; float inputs are assumed to be in [0, 1].
        needs_rescale = not torch.is_floating_point(images)
        images = images.to(device=self.pixel_mean.device)
        images = images.permute(0, 3, 1, 2).float()
        if needs_rescale:
            images = images / 255.0

        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (images - self.pixel_mean) / self.pixel_std

    @torch.no_grad()
    def forward(self, images: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Encode a batch of frames into ``[B, feature_dim]`` float32 features."""
        pixel_values = self._preprocess(images)
        pixel_values = pixel_values.to(dtype=next(self.vit.parameters()).dtype)
        outputs = self.vit(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state  # [B, 1 + num_patches, D]
        features = hidden[:, 0] if self.pooling == "cls" else hidden[:, 1:].mean(dim=1)
        return features.float()

    @torch.no_grad()
    def encode_frames(
        self,
        prev_images: Union[torch.Tensor, np.ndarray],
        cur_images: Union[torch.Tensor, np.ndarray],
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Encode a stacked frame pair into ``[B, 2 * feature_dim]``.

        Both frames go through the same encoder in one batch so the two halves
        of the observation are guaranteed to share preprocessing.
        """
        prev_features = self._encode_chunked(prev_images, batch_size)
        cur_features = self._encode_chunked(cur_images, batch_size)
        return torch.cat([prev_features, cur_features], dim=-1)

    def _encode_chunked(self, images, batch_size: int) -> torch.Tensor:
        num_images = len(images)
        if num_images <= batch_size:
            return self(images)
        return torch.cat(
            [self(images[i : i + batch_size]) for i in range(0, num_images, batch_size)],
            dim=0,
        )
