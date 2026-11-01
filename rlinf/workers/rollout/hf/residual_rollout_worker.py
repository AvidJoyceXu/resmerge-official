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

import copy
import gc
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.models import get_model
from rlinf.scheduler import Channel
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from rlinf.workers.rollout.hf.utils import init_real_obs


class ResidualRolloutWorker(MultiStepRolloutWorker):
    """
    Rollout Worker with residual policy support.

    Integrates base model (OpenVLA) and residual policy:
    - Base model generates base_action
    - Residual policy generates residual_action
    - Final action = base_action + res_scale * residual_action
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Residual policy configuration
        self.residual_cfg = cfg.get("residual_policy", {})
        self.use_residual = self.residual_cfg.get("enabled", False)

        if self.use_residual:
            self.res_scale = self.residual_cfg.get("res_scale", 0.1)
            self.prog_explore = self.residual_cfg.get("prog_explore", 10000)
            self.prog_explore_threshold = self.residual_cfg.get("prog_explore_threshold", 0)
            self.actor_input = cfg.get("network", {}).get("actor_input", "obs")
            self.critic_input = cfg.get("network", {}).get("critic_input", "res")

            # RGB observation mode: a frozen ViT turns the stacked camera
            # frames into the visual half of the residual observation. It lives
            # on the rollout worker rather than inside the policy so that the
            # actor worker neither instantiates nor shards it, and so that only
            # encoded features enter the replay buffer.
            self.visual_encoder_cfg = cfg.get("visual_encoder", {})
            self.use_visual_encoder = self.visual_encoder_cfg.get("enabled", False)
            self.visual_encoder = None

            # Base model will be loaded in init_worker
            self.base_model = None
            self.base_model_config = None
            self.base_model_type = None

            # Global step for progressive exploration
            self.global_step = 0

    def _get_base_model_kwargs(self, mode: str) -> dict:
        """Build kwargs for base_model.predict_action_batch based on its model type.

        OpenVLA / OpenVLA-OFT consume autoregressive sampling kwargs
        (do_sample, temperature, top_k, ...). OpenPI / Pi05 only take
        ``mode``, ``compute_values``, ``return_obs``.
        """
        if self.base_model_type in (
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ):
            kwargs = (
                self._train_sampling_params
                if mode == "train"
                else self._eval_sampling_params
            ).copy()
            kwargs["calulate_logprobs"] = False
            kwargs["calulate_values"] = False
            kwargs["return_obs"] = False
            return kwargs
        # OpenPI (pi0 / pi05), MLP, CNN, GR00T all take only mode-style kwargs.
        return {"mode": mode, "compute_values": False, "return_obs": False}

    def init_worker(self):
        """Initialize worker with residual policy support."""
        # Initialize residual policy (main model)
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.path = self.cfg.rollout.model.model_path

        self.hf_model = get_model(rollout_model_config)
        self.hf_model.eval()
        # Load base model if residual policy is enabled
        if self.use_residual:
            base_model_path = self.residual_cfg.get("base_model_path", None)
            if base_model_path is None:
                raise ValueError("base_model_path must be specified when residual_policy.enabled=True")

            # Create base model config (OpenVLA-OFT, Pi0/Pi0.5, ...)
            base_model_config = copy.deepcopy(self.cfg.actor.base_model)
            self.base_model = get_model(base_model_config)
            self.base_model.eval()
            self.base_model_config = base_model_config
            self.base_model_type = SupportedModel(base_model_config.model_type)

            # Freeze base model parameters
            for param in self.base_model.parameters():
                param.requires_grad = False

            print(
                f"[ResidualRolloutWorker] Base model ({self.base_model_type.value}) "
                f"loaded from {base_model_path}"
            )
            print(f"[ResidualRolloutWorker] Residual policy enabled with res_scale={self.res_scale}")

            if self.use_visual_encoder:
                from rlinf.models.embodiment.residual_policy.frozen_visual_encoder import (
                    DEFAULT_VISUAL_ENCODER,
                    FrozenVisualEncoder,
                )

                self.visual_encoder = FrozenVisualEncoder(
                    model_path=self.visual_encoder_cfg.get(
                        "model_path", DEFAULT_VISUAL_ENCODER
                    ),
                    image_size=self.visual_encoder_cfg.get("image_size", 224),
                    pooling=self.visual_encoder_cfg.get("pooling", "cls"),
                )
                self.visual_encoder.eval().to(self.device)
                print(
                    f"[ResidualRolloutWorker] Frozen visual encoder "
                    f"({self.visual_encoder.model_path}) enabled, "
                    f"feature_dim={self.visual_encoder.feature_dim}"
                )

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def _encode_visual_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Prepend frozen visual features to ``rl_flatten_obs`` in RGB obs mode.

        The env ships the stacked proprioception in ``rl_flatten_obs`` together
        with the previous camera frame; here the two frames are encoded and
        concatenated in front of it, giving
        ``[f(I_{t-1}), f(I_t), p_{t-1}, p_t]``. In privileged obs mode this is
        a no-op.
        """
        if self.visual_encoder is None or obs is None:
            return obs
        if obs.get("prev_main_images") is None:
            return obs

        visual_features = self.visual_encoder.encode_frames(
            obs["prev_main_images"], obs["main_images"]
        )
        proprio = obs["rl_flatten_obs"].to(
            device=visual_features.device, dtype=visual_features.dtype
        )
        obs["rl_flatten_obs"] = torch.cat([visual_features, proprio], dim=-1)
        # Consume the stacked frame so re-encoding the same dict is a no-op
        # and the raw frame stops being carried around once encoded.
        obs["prev_main_images"] = None
        return obs

    def predict(self, env_obs, mode="train", global_step=None, raw_env_obs=None):
        """
        Predict actions with residual policy support.

        Args:
            env_obs: Preprocessed environment observation (for residual policy)
            mode: "train" or "eval"
            global_step: Current global step for progressive exploration
            raw_env_obs: Raw environment observation (for base model)

        Returns:
            actions: Final actions [B, num_action_chunks, action_dim]
            result: Result dict with prev_logprobs, prev_values, forward_inputs
        """
        if global_step is not None:
            self.global_step = global_step

        if not self.use_residual:
            # Fallback to standard prediction
            return super().predict(env_obs, mode)

        # Use raw_env_obs for base model if provided, otherwise use env_obs
        # Base model needs raw env_obs with task_descriptions
        base_env_obs = raw_env_obs if raw_env_obs is not None else env_obs
        # Prepare sampling parameters for base model based on its type
        base_kwargs = self._get_base_model_kwargs(mode)

        # Get base action from base model
        with torch.no_grad():
            base_chunk_actions, base_result = self.base_model.predict_action_batch(
                env_obs=base_env_obs,
                **base_kwargs,
            )

        # Convert base actions to tensor and flatten
        base_actions_tensor = torch.from_numpy(base_chunk_actions).to(self.device)
        B, num_chunks, action_dim = base_actions_tensor.shape
        base_actions_flat = base_actions_tensor.reshape(B, -1)  # [B, num_chunks * action_dim]

        # Progressive exploration: mask residual actions
        if mode == "train":
            res_ratio = min(self.global_step / self.prog_explore, 1.0)
            enable_res_masks = np.random.rand(B) < res_ratio

            if self.global_step <= self.prog_explore_threshold:
                enable_res_masks = np.zeros(B, dtype=bool)
        else:
            enable_res_masks = np.ones(B, dtype=bool)  # Always enable in eval

        # Get residual action from residual policy
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
            SupportedModel.RESIDUAL_POLICY,
            SupportedModel.LORA_RESIDUAL_POLICY,
        ]:
            kwargs = {"mode": mode}

        kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        # Build input for residual policy based on actor_input mode
        if self.actor_input == "obs":
            residual_env_obs = env_obs
            residual_kwargs = kwargs
        elif self.actor_input == "obs_base_action":
            # Need to pass base_action to residual policy
            residual_env_obs = env_obs
            residual_kwargs = kwargs.copy()
            residual_kwargs["base_action"] = base_actions_flat.cpu().numpy()
        else:
            raise ValueError(f"Invalid actor_input: {self.actor_input}")

        with torch.no_grad():
            residual_chunk_actions, residual_result = self.hf_model.predict_action_batch(
                env_obs=residual_env_obs,
                base_action=base_actions_flat.cpu().numpy() if self.actor_input == "obs_base_action" else None,
                **residual_kwargs,
            )

        # Convert residual actions to tensor
        residual_actions_tensor = torch.from_numpy(residual_chunk_actions).to(self.device)

        # NOTE: Apply progressive exploration mask
        if mode == "train":
            residual_actions_tensor = residual_actions_tensor * torch.from_numpy(
                enable_res_masks.reshape(-1, 1, 1)
            ).to(self.device)

        # Compute final action: base_action + res_scale * residual_action
        final_actions = base_actions_tensor + self.res_scale * residual_actions_tensor
        final_actions_np = final_actions.cpu().numpy()

        # Compute residual action metrics for logging
        # 1. Residual action norm ratio (contribution to final action)
        res_norm = torch.norm(residual_actions_tensor.reshape(B, -1), dim=1)  # [B]
        base_norm = torch.norm(base_actions_flat, dim=1)  # [B]
        res_norm_ratio = res_norm / (base_norm + 1e-8)  # [B]

        # 2. Residual action enabled ratio (probability of enabling residual)
        if mode == "train":
            res_enabled_ratio = enable_res_masks.astype(np.float32).mean()  # scalar
        else:
            res_enabled_ratio = 1.0  # Always enabled in eval

        # 3. Residual norm ratio for enabled samples only
        if mode == "train" and np.any(enable_res_masks):
            enabled_res_norm = res_norm[torch.from_numpy(enable_res_masks).to(self.device)]
            enabled_base_norm = base_norm[torch.from_numpy(enable_res_masks).to(self.device)]
            res_norm_ratio_enabled = (enabled_res_norm / (enabled_base_norm + 1e-8)).mean().item()
        else:
            res_norm_ratio_enabled = res_norm_ratio.mean().item() if mode == "eval" else 0.0

        # Prepare forward_inputs based on critic_input mode
        forward_inputs = residual_result["forward_inputs"].copy()

        # Store actions in forward_inputs based on storage strategy
        if self.critic_input == "res" and self.actor_input == "obs":
            # Only store residual action
            forward_inputs["action"] = residual_result["forward_inputs"]["action"]
        else:
            forward_inputs["residual_action"] = residual_result["forward_inputs"]["action"]
            forward_inputs["base_action"] = base_actions_flat.cpu()
            # TODO: base_next_action will be added later when we have next_obs

        # Use residual policy's logprobs and values
        result = {
            "prev_logprobs": residual_result["prev_logprobs"],
            "prev_values": residual_result["prev_values"],
            "forward_inputs": forward_inputs,
            # Add residual metrics for logging
            "res_norm_ratio": res_norm_ratio.cpu(),  # [B] tensor
            "res_norm_ratio_enabled": res_norm_ratio_enabled,  # scalar
            "res_enabled_ratio": res_enabled_ratio,  # scalar
        }

        return final_actions_np, result

    def get_dones_and_rewards(
        self, env_output: dict[str, torch.Tensor], extracted_obs: dict[str, Any]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
        """
        Get dones and rewards from environment batch, with support for computing base_next_action.

        Overrides parent method to handle residual policy's need for raw env_obs (with task_descriptions)
        when calling predict() for bootstrap value computation.

        Args:
            env_output: Environment batch containing dones, rewards, and optionally final_obs
            extracted_obs: Preprocessed observation (residual policy format)

        Returns:
            Tuple of (dones, rewards, real_extracted_obs). dones and rewards are tensors.
        """
        # First step: no rewards yet, only dones
        real_extracted_obs = None
        if env_output["rewards"] is None:
            if hasattr(self.hf_model, "q_head"):
                real_extracted_obs = init_real_obs(extracted_obs)
            return (
                env_output["dones"].bool().cpu().contiguous(),
                None,
                real_extracted_obs,
            )

        dones = env_output["dones"].bool().cpu().contiguous()
        rewards = env_output["rewards"].cpu().contiguous()

        # Handle auto_reset: add bootstrap value to rewards for done episodes
        # Note: currently this is not correct for chunk-size>1 with partial reset
        if dones.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head"):
                final_obs = env_output["final_obs"]
                with torch.no_grad():
                    final_extracted_obs = self.hf_model.preprocess_env_obs(
                        self._encode_visual_obs(final_obs)
                    )
                    if hasattr(self.hf_model, "q_head"):
                        real_extracted_obs = init_real_obs(final_extracted_obs)
                    # For residual policy, predict() needs raw final_obs (with task_descriptions) for base model
                    if self.use_residual:
                        actions, result = self.predict(
                            env_obs=final_extracted_obs,
                            mode="train",
                            global_step=self.global_step,
                            raw_env_obs=final_obs  # Pass raw final_obs for base model
                        )
                    else:
                        actions, result = self.predict(final_extracted_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                    # Handle different shapes of _final_values
                    # _final_values can be [B, num_chunks, 1] or [B, num_chunks]
                    if _final_values.ndim == 3:
                        _final_values = _final_values[:, 0, :]  # [B, 1] or [B, action_dim]
                    elif _final_values.ndim == 2:
                        _final_values = _final_values[:, 0]  # [B]
                    # Ensure _final_values is 1D [B]
                    if _final_values.ndim > 1:
                        _final_values = _final_values.squeeze(-1)  # [B]

                    final_values = torch.zeros_like(_final_values)  # [bsz, ]
                    last_step_dones = dones[:, -1]  # [bsz, ]

                    final_values[last_step_dones] = _final_values[last_step_dones]

                    # Add bootstrap value to the last step of done episodes
                    rewards[:, -1] += self.cfg.algorithm.gamma * final_values.cpu()

        if real_extracted_obs is None and hasattr(self.hf_model, "q_head"):
            real_extracted_obs = init_real_obs(extracted_obs)

        return dones, rewards, real_extracted_obs

    async def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        """Generate rollouts with residual policy support."""
        if self.enable_offload:
            self.reload_model()

        self.buffer_list = [
            EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
            for _ in range(self.num_pipeline_stages)
        ]

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            last_extracted_obs = [None for i in range(self.num_pipeline_stages)]
            last_forward_inputs = [
                None for i in range(self.num_pipeline_stages)
            ]
            last_base_actions = [None for i in range(self.num_pipeline_stages)]

            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)

                    if last_forward_inputs[stage_id] is not None:
                        last_forward_inputs[stage_id] = self.update_intervene_actions(
                            env_output, last_forward_inputs[stage_id]
                        )

                    residual_extracted_obs = self.hf_model.preprocess_env_obs(
                        self._encode_visual_obs(env_output["obs"])
                    )
                    # Use residual_extracted_obs for get_dones_and_rewards (same as original codebase)
                    # This ensures compatibility with init_real_obs which expects residual policy format
                    dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                        env_output, residual_extracted_obs
                    )

                    # Predict actions with residual policy
                    # Pass raw env_output["obs"] for base model, residual_extracted_obs for residual policy
                    actions, result = self.predict(
                        env_obs=residual_extracted_obs,
                        mode="train",
                        global_step=self.global_step,
                        raw_env_obs=env_output["obs"]  # Raw obs with task_descriptions for base model
                    )

                    # NOTE: base_next_action should be added to last_forward_inputs (previous step's forward_inputs)
                    if last_forward_inputs[stage_id] is not None and self.use_residual and self.critic_input != "res":
                        last_forward_inputs[stage_id]["base_next_action"] = result["forward_inputs"]["base_action"]

                    forward_inputs_to_store = last_forward_inputs[stage_id]

                    res_norm_ratio = result.get("res_norm_ratio", None)
                    res_norm_ratio_enabled = result.get("res_norm_ratio_enabled", None)
                    res_enabled_ratio = result.get("res_enabled_ratio", None)

                    chunk_step_result = ChunkStepResult(
                        prev_logprobs=result["prev_logprobs"],
                        prev_values=result["prev_values"],
                        dones=dones,
                        truncations=env_output["truncations"],
                        terminations=env_output["terminations"],
                        rewards=rewards,
                        forward_inputs=forward_inputs_to_store,
                    )

                    # Add residual metrics to chunk_step_result if available
                    if res_norm_ratio is not None:
                        chunk_step_result.res_norm_ratio = res_norm_ratio
                    if res_norm_ratio_enabled is not None:
                        chunk_step_result.res_norm_ratio_enabled = res_norm_ratio_enabled
                    if res_enabled_ratio is not None:
                        chunk_step_result.res_enabled_ratio = res_enabled_ratio

                    self.buffer_list[stage_id].append_result(chunk_step_result)

                    if last_extracted_obs[stage_id] is not None and hasattr(
                        self.hf_model, "q_head"
                    ):
                        self.buffer_list[stage_id].add_transition(
                            last_extracted_obs[stage_id], real_extracted_obs
                        )
                    last_extracted_obs[stage_id] = residual_extracted_obs
                    last_forward_inputs[stage_id] = result["forward_inputs"]

                    self.send_chunk_actions(output_channel, actions)

                    # Update global step (will be set by runner)

            # Final step handling
            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                last_forward_inputs[stage_id] = self.update_intervene_actions(
                    env_output, last_forward_inputs[stage_id]
                )

                extracted_obs = self.hf_model.preprocess_env_obs(
                    self._encode_visual_obs(env_output["obs"])
                )
                dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                    env_output, extracted_obs
                )

                # For residual policy, predict() needs raw env_output["obs"] (with task_descriptions) for base model
                # NOTE: Call predict() BEFORE appending last_forward_inputs to buffer, so we can add base_next_action
                actions, result = self.predict(
                    env_obs=extracted_obs,
                    mode="train",
                    global_step=self.global_step,
                    raw_env_obs=env_output["obs"]  # Raw obs with task_descriptions for base model
                )

                # NOTE: Add base_next_action to last_forward_inputs (previous step's forward_inputs) if needed
                # This ensures consistency with normal step handling
                if last_forward_inputs[stage_id] is not None and self.use_residual and self.critic_input != "res":
                    last_forward_inputs[stage_id]["base_next_action"] = result["forward_inputs"]["base_action"]

                self.buffer_list[stage_id].dones.append(dones)
                self.buffer_list[stage_id].truncations.append(env_output["truncations"])
                self.buffer_list[stage_id].terminations.append(
                    env_output["terminations"]
                )
                self.buffer_list[stage_id].rewards.append(rewards)
                self.buffer_list[stage_id].forward_inputs.append(
                    put_tensor_device(last_forward_inputs[stage_id], "cpu")
                )

                if "prev_values" in result:
                    self.buffer_list[stage_id].prev_values.append(
                        result["prev_values"].cpu().contiguous()
                    )
                if hasattr(self.hf_model, "q_head"):
                    self.buffer_list[stage_id].add_transition(
                        last_extracted_obs[stage_id], real_extracted_obs
                    )

        for i in range(self.num_pipeline_stages):
            self.send_rollout_batch(actor_channel, i)

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        """Evaluate with residual policy support."""
        if self.enable_offload:
            self.reload_model()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(n_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    extracted_obs = self.hf_model.preprocess_env_obs(
                        self._encode_visual_obs(env_output["obs"])
                    )
                    # For residual policy, predict() needs raw env_output["obs"] (with task_descriptions) for base model
                    if self.use_residual:
                        actions, _ = self.predict(
                            env_obs=extracted_obs,
                            mode="eval",
                            global_step=self.global_step,
                            raw_env_obs=env_output["obs"]  # Raw obs with task_descriptions for base model
                        )
                    else:
                        actions, _ = self.predict(extracted_obs, mode="eval")
                    self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        """Offload models to CPU."""
        self.hf_model = self.hf_model.to("cpu")
        if self.use_residual and self.base_model is not None:
            self.base_model = self.base_model.to("cpu")
        if self.visual_encoder is not None:
            self.visual_encoder = self.visual_encoder.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        """Reload models to GPU."""
        self.hf_model = self.hf_model.to(self.device)
        if self.use_residual and self.base_model is not None:
            self.base_model = self.base_model.to(self.device)
        if self.visual_encoder is not None:
            self.visual_encoder = self.visual_encoder.to(self.device)

    def set_global_step(self, global_step):
        """Set global step for progressive exploration."""
        self.global_step = global_step
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
