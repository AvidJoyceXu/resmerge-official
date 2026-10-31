from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
import os

from psi_policy.common.normalize_util import get_image_range_normalizer
from psi_policy.common.streaming_replay_buffer import StreamingReplayBuffer
from psi_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from psi_policy.model.common.normalizer import LinearNormalizer
from psi_policy.dataset.base_dataset import BaseImageDataset
from psi_policy.common.nested_dict_util import obs_keys_by_type


def _zarr_path(rel: str, zarr_root: Optional[str]) -> str:
    rel = os.path.expanduser(rel)
    if zarr_root is None:
        return rel
    zarr_root = os.path.expanduser(zarr_root)
    return rel if zarr_root == "" else os.path.join(zarr_root, rel)


class PsiPolicyFrankaLeapSingleRgbDataset(BaseImageDataset):
    """Franka + Leap single-camera dataset backed by zarr.

    Expected source keys come from shape_meta when provided. For the current
    recorder->zarr pipeline, common keys are:
      - rgb: side_policy_256 or side_policy_256_{0,1,...}
      - low_dim: leap_pose, tcp_force, tcp_pose, tcp_torque, tcp_vel
      - action: action (22D)
    """

    def __init__(
        self,
        zarr_paths,
        horizon=1,
        n_obs_steps=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        image_size=(224, 224),
        keys=None,
        shape_meta: Optional[dict] = None,
        zarr_root: Optional[str] = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.zarr_paths = zarr_paths
        self.shape_meta = shape_meta

        derived = obs_keys_by_type(shape_meta, "rgb")
        self.rgb_camera_keys = list(derived) if derived else [
            "side_policy_256",
            "side_policy_256_0",
            "side_policy_256_1",
        ]

        
        if shape_meta is not None:
            self.state_keys = [
                k for k, v in shape_meta['obs'].items()
                if v.get('type') == 'low_dim'
            ]
        else:
            self.state_keys = [
                'leap_pose',
                'tcp_force',
                'tcp_pose',
                'tcp_torque',
                'tcp_vel',
            ]

        self.action_key = "action"

        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []

        for rel in zarr_paths:
            replay_buffer = StreamingReplayBuffer.copy_from_path(
                _zarr_path(rel, zarr_root), keys=keys
            )
            self.replay_buffers.append(replay_buffer)
            if derived:
                available_rgb_keys = self.rgb_camera_keys
            else:
                self.rgb_camera_keys = [
                    k for k in self.rgb_camera_keys if k in replay_buffer.keys()
                ]
                if not self.rgb_camera_keys:
                    raise KeyError(
                        "No rgb keys found in zarr. Expected one of "
                        "side_policy_256 / side_policy_256_0 / side_policy_256_1 "
                        "or provide shape_meta with rgb obs keys."
                    )
            missing_state = [k for k in self.state_keys if k not in replay_buffer.keys()]
            if missing_state:
                raise KeyError(f"Missing low_dim keys in zarr: {missing_state}")
            if self.action_key not in replay_buffer.keys():
                raise KeyError(f"Missing action key in zarr: {self.action_key}")

            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes,
                val_ratio=val_ratio,
                seed=seed
            )
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=max_train_episodes,
                seed=seed
            )
            self.train_masks.append(train_mask)

            key_first_k = {k: n_obs_steps for k in self.rgb_camera_keys}
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=train_mask,
                key_first_k=key_first_k
            )
            self.samplers.append(sampler)
            self.sampler_lens.append(len(sampler))

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

    def _process_image_batch(self, images):
        rgb = torch.from_numpy(images[..., :3]).float()
        rgb = rgb.permute(0, 3, 1, 2)
        rgb = F.interpolate(
            rgb / 255.0,
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )
        return rgb

    def _sample_to_data(self, sample):
        t_slice = slice(self.n_obs_steps)

        data = {'obs': {}}

        
        for key in self.state_keys:
            data['obs'][key] = torch.from_numpy(
                sample[key].astype(np.float32)[t_slice]
            )

        for cam_key in self.rgb_camera_keys:
            rgb_processed_frames = self._process_image_batch(sample[cam_key][t_slice])
            data['obs'][cam_key] = rgb_processed_frames

        data['action'] = torch.from_numpy(sample[self.action_key].astype(np.float32))
        return data

    def get_normalizer(self, mode='limits', **kwargs):
        
        state_data = {key: [] for key in self.state_keys}
        actions = []

        for rb in self.replay_buffers:
            for key in self.state_keys:
                state_data[key].append(rb[key])
            actions.append(rb[self.action_key].astype(np.float32))

        data = {}
        for key in self.state_keys:
            data[key] = np.concatenate(state_data[key], axis=0)
        data['action'] = np.concatenate(actions, axis=0)

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        for cam_key in self.rgb_camera_keys:
            normalizer[cam_key] = get_image_range_normalizer()
        return normalizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        curr_idx = idx
        for i, length in enumerate(self.sampler_lens):
            if curr_idx < length:
                sample = self.samplers[i].sample_sequence(curr_idx)
                break
            curr_idx -= length

        return self._sample_to_data(sample)

    def get_all_actions(self) -> torch.Tensor:
        parts = []
        for rb in self.replay_buffers:
            parts.append(rb[self.action_key].astype(np.float32))
        return torch.from_numpy(np.concatenate(parts, axis=0))

    def __len__(self):
        return sum(self.sampler_lens)