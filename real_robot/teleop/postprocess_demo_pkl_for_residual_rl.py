#!/usr/bin/env python3
# Modified by the ResMerge contributors, 2026-10-31; see real_robot/THIRD_PARTY.md.
"""
Post-process demo pkl for residual RL training.

This script fills missing:
  1) observations["imle_preproj"] / next_observations["imle_preproj"]
  2) rewards

Images in the output ``--output_pkl``:
  - **Default**: RGB keys under ``observations`` / ``next_observations`` are **dropped**
    after computing ``imle_preproj`` (only ``state`` + ``imle_preproj`` kept) to save space.
  - With ``--keep_obs_images``: RGB (e.g. ``side_policy_256`` / ``wrist_1``) are **retained**
    in the written pickle alongside ``imle_preproj``.

Supported input layouts:
  - flat list[transition]
  - list[list[transition]] (trajectory list)
  - dict with key "transitions" (+ optional "meta")

See the root README.md for the supported IMLE feature-replay command.

"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import re
import sys
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


STATE_SLICES: list[tuple[str, slice]] = [
    ("leap_pose", slice(0, 16)),
    ("tcp_force", slice(16, 19)),
    ("tcp_pose", slice(19, 25)),
    ("tcp_torque", slice(25, 28)),
    ("tcp_vel", slice(28, 34)),
]


def _prepend_sys_path(path: str) -> None:
    path = os.path.abspath(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def _repo_root_from_this_file() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_precise_dex_root() -> str:
    return os.path.join(_repo_root_from_this_file(), "imle_dex_xy", "precise_dex")


def _split_hydra_overrides(argv: list[str]) -> tuple[list[str], list[str]]:
    hydra_key_re = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")
    hydra_overrides: list[str] = []
    rest: list[str] = []
    for tok in argv:
        if ("=" in tok) and (not tok.startswith("--")):
            key, _, _ = tok.partition("=")
            if "/" not in key and "\\" not in key and hydra_key_re.fullmatch(key):
                hydra_overrides.append(tok)
            else:
                rest.append(tok)
        else:
            rest.append(tok)
    return hydra_overrides, rest


def _is_transition(x: Any) -> bool:
    return isinstance(x, dict) and "observations" in x and "next_observations" in x


def _normalize_payload(raw: Any) -> tuple[list[list[dict[str, Any]]], str]:
    if isinstance(raw, dict) and isinstance(raw.get("transitions"), list):
        trans = raw["transitions"]
        if len(trans) == 0:
            return [], "dict_transitions_flat"
        if _is_transition(trans[0]):
            return [trans], "dict_transitions_flat"
        if isinstance(trans[0], list) and len(trans[0]) > 0 and _is_transition(trans[0][0]):
            return trans, "dict_transitions_trajectories"
        raise ValueError("Unsupported payload['transitions'] layout.")

    if isinstance(raw, list):
        if len(raw) == 0:
            return [], "list_flat"
        if _is_transition(raw[0]):
            return [raw], "list_flat"
        if isinstance(raw[0], list) and len(raw[0]) > 0 and _is_transition(raw[0][0]):
            return raw, "list_trajectories"
    raise ValueError("Unsupported pkl layout.")


def _transitions_from_zarr(
    zarr_path: str,
    *,
    stack_obs_num: int,
) -> list[list[dict[str, Any]]]:
    def _extract_current_frame(x: np.ndarray, base_dim: int) -> np.ndarray:
        """
        Select the current D-dimensional frame from flattened H*D or D low-dimensional Zarr
        fields to construct (H, 34) state.
        """
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        d = int(base_dim)
        if v.size == d:
            return v.astype(np.float32, copy=False)
        if v.size % d == 0:
            return v.reshape(-1, d)[-1].astype(np.float32, copy=False)
        raise ValueError(f"low_dim size {v.size} is not compatible with base_dim {d}")

    root = zarr.open(os.path.abspath(zarr_path), mode="r")
    if "data" not in root or "meta" not in root:
        raise ValueError("zarr must contain groups: data/ and meta/")
    data = root["data"]
    meta = root["meta"]
    if "episode_ends" not in meta:
        raise ValueError("zarr meta/episode_ends not found")

    required = ("action", "leap_pose", "tcp_force", "tcp_pose", "tcp_torque", "tcp_vel")
    for k in required:
        if k not in data:
            raise ValueError(f"zarr data/{k} not found")

    side_keys = []
    if "side_policy_256" in data:
        side_keys = ["side_policy_256"]
    else:
        side_keys = sorted([k for k in data.keys() if str(k).startswith("side_policy_256_")])
    if len(side_keys) == 0:
        raise ValueError("zarr data/side_policy_256 or side_policy_256_* not found")

    actions = np.asarray(data["action"])
    leap_pose = np.asarray(data["leap_pose"])
    tcp_force = np.asarray(data["tcp_force"])
    tcp_pose = np.asarray(data["tcp_pose"])
    tcp_torque = np.asarray(data["tcp_torque"])
    tcp_vel = np.asarray(data["tcp_vel"])
    imgs = [np.asarray(data[k]) for k in side_keys]

    T = int(actions.shape[0])
    if T == 0:
        return []
    episode_ends = np.asarray(meta["episode_ends"], dtype=np.int64).reshape(-1).tolist()
    if len(episode_ends) == 0:
        episode_ends = [T]

    trajectories: list[list[dict[str, Any]]] = []
    stack_n = int(max(1, int(stack_obs_num)))
    start = 0
    for ep_end in episode_ends:
        end = int(ep_end)
        if end <= start:
            continue
        ep_states_1d: list[np.ndarray] = []
        for tt in range(start, end):
            leap_t = _extract_current_frame(leap_pose[tt], 16)
            force_t = _extract_current_frame(tcp_force[tt], 3)
            pose_t = _extract_current_frame(tcp_pose[tt], 6)
            torque_t = _extract_current_frame(tcp_torque[tt], 3)
            vel_t = _extract_current_frame(tcp_vel[tt], 6)
            s_tt = np.concatenate(
                [leap_t, force_t, pose_t, torque_t, vel_t],
                axis=0,
            ).astype(np.float32, copy=False)
            ep_states_1d.append(s_tt)

        def _stack_state(ep_idx: int) -> np.ndarray:
            if len(ep_states_1d) == 0:
                raise ValueError("empty episode states")
            idx = int(max(0, min(ep_idx, len(ep_states_1d) - 1)))
            left = max(0, idx - stack_n + 1)
            seq = ep_states_1d[left : idx + 1]
            if len(seq) < stack_n:
                pad = [seq[0]] * (stack_n - len(seq))
                seq = pad + seq
            return np.stack(seq, axis=0).astype(np.float32, copy=False)

        ep_tr: list[dict[str, Any]] = []
        for t in range(start, end):
            t_next = t + 1 if (t + 1) < end else t
            ep_idx = t - start
            ep_next_idx = t_next - start
            state_t = _stack_state(ep_idx)
            state_next = _stack_state(ep_next_idx)

            if len(imgs) == 1:
                img_t = imgs[0][t]
                img_next = imgs[0][t_next]
            else:
                img_t = np.stack([arr[t] for arr in imgs], axis=0)
                img_next = np.stack([arr[t_next] for arr in imgs], axis=0)

            done = bool(t == end - 1)
            action_t = np.asarray(actions[t], dtype=np.float32).reshape(-1)
            action_next_base = (
                np.asarray(actions[t_next], dtype=np.float32).reshape(-1) if not done else action_t.copy()
            )
            tr = {
                "observations": {
                    "state": state_t,
                    "side_policy_256": np.asarray(img_t),
                    "leap_pose": np.asarray(leap_pose[t], dtype=np.float32),
                    "tcp_force": np.asarray(tcp_force[t], dtype=np.float32),
                    "tcp_pose": np.asarray(tcp_pose[t], dtype=np.float32),
                    "tcp_torque": np.asarray(tcp_torque[t], dtype=np.float32),
                    "tcp_vel": np.asarray(tcp_vel[t], dtype=np.float32),
                },
                "actions": action_t,
                "action_base": action_t.copy(),
                "next_action_base": action_next_base,
                "next_observations": {
                    "state": state_next,
                    "side_policy_256": np.asarray(img_next),
                    "leap_pose": np.asarray(leap_pose[t_next], dtype=np.float32),
                    "tcp_force": np.asarray(tcp_force[t_next], dtype=np.float32),
                    "tcp_pose": np.asarray(tcp_pose[t_next], dtype=np.float32),
                    "tcp_torque": np.asarray(tcp_torque[t_next], dtype=np.float32),
                    "tcp_vel": np.asarray(tcp_vel[t_next], dtype=np.float32),
                },
                "rewards": np.float32(0.0),
                "masks": np.float32(0.0 if done else 1.0),
                "dones": done,
            }
            ep_tr.append(tr)
        trajectories.append(ep_tr)
        start = end
    return trajectories


def _restore_payload_layout(original: Any, trajs: list[list[dict[str, Any]]], layout: str) -> Any:
    if layout == "dict_transitions_flat":
        out = dict(original)
        out["transitions"] = trajs[0] if len(trajs) > 0 else []
        return out
    if layout == "dict_transitions_trajectories":
        out = dict(original)
        out["transitions"] = trajs
        return out
    if layout == "list_flat":
        return trajs[0] if len(trajs) > 0 else []
    if layout == "list_trajectories":
        return trajs
    raise ValueError(f"Unknown layout: {layout}")


def _process_rgb_hwc_uint8(img: np.ndarray, out_hw: tuple[int, int]) -> torch.Tensor:
    rgb = torch.from_numpy(np.asarray(img[..., :3], dtype=np.float32))
    rgb = rgb.permute(2, 0, 1).unsqueeze(0)
    rgb = F.interpolate(rgb / 255.0, size=out_hw, mode="bilinear", align_corners=False)
    return rgb


def _build_policy_obs_from_transition(
    obs: dict[str, Any],
    *,
    rgb_keys: list[str],
    image_hw: tuple[int, int],
    stack_obs_num: int,
) -> dict[str, torch.Tensor]:
    side = np.asarray(obs["side_policy_256"])
    if side.ndim == 3:
        side = side[None, ...]
    if side.ndim != 4:
        raise ValueError(f"obs['side_policy_256'] should be 4D, got {side.shape}")
    if int(side.shape[0]) != int(len(rgb_keys)):
        raise ValueError(
            f"side_policy_256 first dim {side.shape[0]} != rgb keys count {len(rgb_keys)}"
        )

    out: dict[str, torch.Tensor] = {}
    for i, key in enumerate(rgb_keys):
        timg = _process_rgb_hwc_uint8(side[i], image_hw)
        out[key] = timg.unsqueeze(1).contiguous()

    for name, sl in STATE_SLICES:
        if name in obs:
            
            v = np.asarray(obs[name], dtype=np.float32).reshape(-1)
        else:
            st = np.asarray(obs["state"], dtype=np.float32)
            if st.ndim == 1:
                st = st.reshape(int(stack_obs_num), -1)
            v = np.asarray(st[:, sl], dtype=np.float32).reshape(-1)
        out[name] = torch.from_numpy(v).reshape(1, 1, -1)
    return out


def _load_policy(
    *,
    precise_dex_root: str,
    ckpt_path: str,
    hydra_overrides: list[str],
    device_override: str,
):
    precise_root = os.path.abspath(precise_dex_root)
    if not os.path.isdir(precise_root):
        raise FileNotFoundError(f"precise_dex_root not found: {precise_root}")
    _prepend_sys_path(precise_root)

    cwd_old = os.getcwd()
    try:
        os.chdir(precise_root)
        from utils import setup_path_variables

        setup_path_variables()
    finally:
        os.chdir(cwd_old)

    hydra_config_dir = os.path.abspath(os.path.join(precise_root, "psi_policy", "config"))
    with initialize_config_dir(version_base=None, config_dir=hydra_config_dir):
        cfg = compose(config_name="psi_policy_workspace", overrides=hydra_overrides)
    OmegaConf.resolve(cfg)

    cfg.optimizer.start_ckpt_path = ckpt_path
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    policy = workspace.model

    if device_override:
        dev = torch.device(device_override)
    else:
        dev = torch.device(cfg.training.device if cfg.training.get("device") else "cuda:0")
    policy.to(dev)
    policy.eval()

    image_shape = list(cfg.task.image_shape)
    _, h, w = image_shape
    image_hw = (int(h), int(w))
    obs_meta = cfg.task.shape_meta["obs"]
    rgb_keys = [k for k in obs_meta if obs_meta[k].get("type") == "rgb"]
    n_action_steps = int(cfg.n_action_steps)
    action_dim = int(cfg.task.shape_meta["action"]["shape"][0])
    return cfg, policy, dev, image_hw, rgb_keys, n_action_steps, action_dim


def _compute_imle_preproj(
    obs: dict[str, Any],
    *,
    policy: torch.nn.Module,
    device: torch.device,
    rgb_keys: list[str],
    image_hw: tuple[int, int],
    stack_obs_num: int,
    action_base: np.ndarray | None,
) -> np.ndarray:
    obs_t = _build_policy_obs_from_transition(
        obs,
        rgb_keys=rgb_keys,
        image_hw=image_hw,
        stack_obs_num=stack_obs_num,
    )
    obs_t = {k: v.to(device, non_blocking=True) for k, v in obs_t.items()}
    with torch.no_grad():
        nobs = policy.normalizer.normalize(obs_t)
        ab = None
        if action_base is not None:
            ab = torch.as_tensor(action_base, device=device, dtype=torch.float32).reshape(1, 1, -1)
        feat = policy.obs_encoder.forward_pre_projection(nobs, training=False, action_base=ab)
    return feat.detach().float().cpu().numpy().reshape(-1).astype(np.float32, copy=False)


def _make_selector_chosen_cand(
    cand_vec: np.ndarray,
    *,
    q_selector_action_dim: int,
) -> np.ndarray:
    cand = np.zeros((int(q_selector_action_dim),), dtype=np.float32)
    src = np.asarray(cand_vec, dtype=np.float32).reshape(-1)
    n = min(int(q_selector_action_dim), int(src.size))
    if n > 0:
        cand[:n] = src[:n]
    return cand


def _attach_q_selector_fields(
    out: dict[str, Any],
    *,
    q_num_candidates: int,
    q_selector_action_dim: int,
    chosen_cand_vec: np.ndarray,
    next_cand_vec: np.ndarray,
) -> None:
    chosen = _make_selector_chosen_cand(
        chosen_cand_vec,
        q_selector_action_dim=int(q_selector_action_dim),
    )
    k = int(q_num_candidates)
    d = int(q_selector_action_dim)
    next_cands = np.zeros((k, d), dtype=np.float32)
    next_cand = _make_selector_chosen_cand(
        next_cand_vec,
        q_selector_action_dim=int(q_selector_action_dim),
    )
    if k > 0:
        next_cands[:] = next_cand[None, :]
    h_scores = np.zeros((k,), dtype=np.float32)
    if k > 0:
        h_scores[:] = 1.0
    out["selector_chosen_cand"] = chosen
    out["selector_next_cands"] = next_cands
    out["selector_h_scores"] = h_scores


def _postprocess_transition(
    tr: dict[str, Any],
    *,
    policy: torch.nn.Module,
    device: torch.device,
    rgb_keys: list[str],
    image_hw: tuple[int, int],
    stack_obs_num: int,
    concat_action_base: bool,
    reward_mode: str,
    drop_obs_images: bool,
    add_q_selector_fields: bool,
    q_num_candidates: int,
    q_selector_action_dim: int,
    chosen_cand_vec: np.ndarray | None,
    next_cand_vec: np.ndarray | None,
) -> dict[str, Any]:
    out = copy.deepcopy(tr)

    action_base = None
    if concat_action_base:
        action_base = out.get("action_base")
        if action_base is None:
            act = np.asarray(out.get("actions"), dtype=np.float32).reshape(-1)
            action_base = np.zeros_like(act, dtype=np.float32)

    out["observations"]["imle_preproj"] = _compute_imle_preproj(
        out["observations"],
        policy=policy,
        device=device,
        rgb_keys=rgb_keys,
        image_hw=image_hw,
        stack_obs_num=stack_obs_num,
        action_base=action_base if concat_action_base else None,
    )
    out["next_observations"]["imle_preproj"] = _compute_imle_preproj(
        out["next_observations"],
        policy=policy,
        device=device,
        rgb_keys=rgb_keys,
        image_hw=image_hw,
        stack_obs_num=stack_obs_num,
        action_base=None,
    )

    if reward_mode == "keep_if_exists":
        if "rewards" not in out:
            out["rewards"] = np.float32(0.0)
    elif reward_mode == "zero":
        out["rewards"] = np.float32(0.0)
    elif reward_mode == "dones_as_terminal_reward":
        done = bool(out.get("dones", False))
        out["rewards"] = np.float32(1.0 if done else 0.0)
    elif reward_mode == "info_succeed":
        info = out.get("infos", {})
        succ = bool(info.get("succeed", False)) if isinstance(info, dict) else False
        out["rewards"] = np.float32(1.0 if succ else 0.0)
    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")

    if drop_obs_images:
        keep = {"state", "imle_preproj"}
        out["observations"] = {
            k: v for k, v in out["observations"].items() if k in keep
        }
        out["next_observations"] = {
            k: v for k, v in out["next_observations"].items() if k in keep
        }
    if add_q_selector_fields:
        if chosen_cand_vec is None or next_cand_vec is None:
            raise ValueError("chosen_cand_vec/next_cand_vec are required when add_q_selector_fields=True")
        _attach_q_selector_fields(
            out,
            q_num_candidates=int(q_num_candidates),
            q_selector_action_dim=int(q_selector_action_dim),
            chosen_cand_vec=chosen_cand_vec,
            next_cand_vec=next_cand_vec,
        )

    return out


def _compute_action_bases_for_traj(
    traj: list[dict[str, Any]],
    *,
    policy: torch.nn.Module,
    device: torch.device,
    rgb_keys: list[str],
    image_hw: tuple[int, int],
    stack_obs_num: int,
    latency_step: int,
    chunk_exec_steps: int,
    n_action_steps: int,
    action_dim: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Recompute action_base and next_action_base using the actor/evaluation chunk cadence and
    previous-trajectory reuse.
    """
    if len(traj) == 0:
        return [], []
    if chunk_exec_steps < 1:
        raise ValueError(f"chunk_exec_steps must be >=1, got {chunk_exec_steps}")

    prev_traj = torch.randn(
        (1, int(n_action_steps), int(action_dim)),
        device=device,
        dtype=torch.float32,
    )
    pred_t: torch.Tensor | None = None
    cached_pred_t: torch.Tensor | None = None
    steps_into_chunk = int(chunk_exec_steps)
    current_chunk_len = int(chunk_exec_steps)

    action_bases: list[np.ndarray] = []
    next_action_bases: list[np.ndarray] = []

    for t, tr in enumerate(traj):
        obs = tr["observations"]
        next_obs = tr["next_observations"]

        if steps_into_chunk >= current_chunk_len:
            if cached_pred_t is not None:
                pred_t = cached_pred_t
                cached_pred_t = None
            else:
                obs_t = _build_policy_obs_from_transition(
                    obs,
                    rgb_keys=rgb_keys,
                    image_hw=image_hw,
                    stack_obs_num=stack_obs_num,
                )
                obs_t = {k: v.to(device, non_blocking=True) for k, v in obs_t.items()}
                with torch.no_grad():
                    out = policy.bidirection_predict_action(
                        obs_t, None, prev_traj, latency_step=int(latency_step)
                    )
                pred_t = out["action"].detach()
            prev_traj = pred_t
            steps_into_chunk = 0
            current_chunk_len = min(
                int(chunk_exec_steps),
                int(pred_t.shape[1]),
                int(n_action_steps),
            )

        assert pred_t is not None
        j = int(steps_into_chunk)
        base_np = (
            pred_t[0, j]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
            .reshape(-1)
        )
        action_bases.append(base_np.copy())

        if (j + 1 < int(current_chunk_len)) and (j + 1 < int(pred_t.shape[1])):
            next_base_np = (
                pred_t[0, j + 1]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
                .reshape(-1)
            )
        else:
            obs_t_next = _build_policy_obs_from_transition(
                next_obs,
                rgb_keys=rgb_keys,
                image_hw=image_hw,
                stack_obs_num=stack_obs_num,
            )
            obs_t_next = {k: v.to(device, non_blocking=True) for k, v in obs_t_next.items()}
            with torch.no_grad():
                out_next = policy.bidirection_predict_action(
                    obs_t_next, None, prev_traj, latency_step=int(latency_step)
                )
            cached_pred_t = out_next["action"].detach()
            next_base_np = (
                cached_pred_t[0, 0]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
                .reshape(-1)
            )
        next_action_bases.append(next_base_np.copy())
        steps_into_chunk += 1

    return action_bases, next_action_bases


def _chunk_wrist_selector_features_np(
    cands_tka: np.ndarray,
    *,
    selector_chunk_len: int,
    wrist_dim: int = 6,
) -> np.ndarray:
    x = np.asarray(cands_tka, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"expect [K,T,A], got {x.shape}")
    k, t, a = x.shape
    wd = int(min(max(int(wrist_dim), 1), a))
    tl = int(max(int(selector_chunk_len), 1))
    out = np.zeros((k, tl, wd), dtype=np.float32)
    use_t = min(t, tl)
    out[:, :use_t, :] = x[:, :use_t, :wd]
    return out.reshape(k, tl * wd)


def _compute_selector_chunk_replay_for_traj(
    traj: list[dict[str, Any]],
    *,
    policy: torch.nn.Module,
    device: torch.device,
    rgb_keys: list[str],
    image_hw: tuple[int, int],
    stack_obs_num: int,
    latency_step: int,
    chunk_exec_steps: int,
    n_action_steps: int,
    action_dim: int,
    q_num_candidates: int,
    q_selector_action_dim: int,
    selector_chunk_size: int,
    selector_wrist_dim: int,
    selector_chosen_source: str,
) -> list[dict[str, Any]]:
    if len(traj) == 0:
        return []
    if chunk_exec_steps < 1:
        raise ValueError(f"chunk_exec_steps must be >=1, got {chunk_exec_steps}")

    prev_traj = torch.randn(
        (1, int(n_action_steps), int(action_dim)),
        device=device,
        dtype=torch.float32,
    )
    prev_selected_base_np: np.ndarray | None = None
    out_recs: list[dict[str, Any]] = []
    t = 0
    k_max = int(q_num_candidates)

    while t < len(traj):
        tr = traj[t]
        obs = tr["observations"]
        obs_t = _build_policy_obs_from_transition(
            obs,
            rgb_keys=rgb_keys,
            image_hw=image_hw,
            stack_obs_num=stack_obs_num,
        )
        obs_t = {k: v.to(device, non_blocking=True) for k, v in obs_t.items()}
        with torch.no_grad():
            if hasattr(policy, "predict_action_all"):
                out = policy.predict_action_all(
                    obs_t, None, prev_traj, latency_step=int(latency_step), num_candidates=int(k_max)
                )
            else:
                out = policy.bidirection_predict_action(
                    obs_t, None, prev_traj, latency_step=int(latency_step)
                )

        pred_t = out["action"].detach()
        prev_traj = pred_t
        base_np = (
            pred_t[0, 0].detach().float().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        )
        prev_selected_base_np = base_np.copy()

        chosen_cand_imle = np.zeros((int(q_selector_action_dim),), dtype=np.float32)
        h_scores_pad = np.zeros((int(k_max),), dtype=np.float32)

        all_norm = out.get("all_actions_normalized", None)
        all_unnorm = out.get("all_actions_unnormalized", None)
        hs = out.get("heuristic_scores", None)
        next_cands_pad = np.zeros((int(k_max), int(q_selector_action_dim)), dtype=np.float32)

        if all_norm is not None and all_unnorm is not None:
            all_norm_np = all_norm.detach().cpu().numpy().astype(np.float32, copy=False)
            all_unnorm_np = all_unnorm.detach().cpu().numpy().astype(np.float32, copy=False)
            if all_norm_np.shape[0] > k_max:
                all_norm_np = all_norm_np[:k_max]
            if all_unnorm_np.shape[0] > k_max:
                all_unnorm_np = all_unnorm_np[:k_max]

            cands_q_np = _chunk_wrist_selector_features_np(
                all_norm_np,
                selector_chunk_len=int(selector_chunk_size),
                wrist_dim=int(selector_wrist_dim),
            )
            cands_exec_np = all_unnorm_np[:, 0, :]

            
            dists = np.linalg.norm(cands_exec_np - base_np[None, :], axis=-1)
            chosen_idx = int(np.argmin(dists))
            chosen_cand_imle = cands_q_np[chosen_idx].astype(np.float32, copy=False)

            if hs is not None:
                hs_np = hs.detach().cpu().numpy().astype(np.float32).reshape(-1)
                hs_np = hs_np[: cands_q_np.shape[0]]
                h_scores_pad[: hs_np.shape[0]] = hs_np
            else:
                ref = prev_selected_base_np if prev_selected_base_np is not None else base_np
                hs_np = -np.linalg.norm(cands_exec_np - ref[None, :], axis=-1).astype(np.float32)
                h_scores_pad[: hs_np.shape[0]] = hs_np

        
        t_next_chunk = t + int(chunk_exec_steps)
        if t_next_chunk < len(traj):
            next_obs = traj[t_next_chunk]["observations"]
            obs_t_next = _build_policy_obs_from_transition(
                next_obs,
                rgb_keys=rgb_keys,
                image_hw=image_hw,
                stack_obs_num=stack_obs_num,
            )
            obs_t_next = {k: v.to(device, non_blocking=True) for k, v in obs_t_next.items()}
            with torch.no_grad():
                if hasattr(policy, "predict_action_all"):
                    out_next = policy.predict_action_all(
                        obs_t_next,
                        None,
                        prev_traj,
                        latency_step=int(latency_step),
                        num_candidates=int(k_max),
                    )
                else:
                    out_next = policy.bidirection_predict_action(
                        obs_t_next, None, prev_traj, latency_step=int(latency_step)
                    )
            all_norm_next = out_next.get("all_actions_normalized", None)
            if all_norm_next is not None:
                all_norm_next_np = all_norm_next.detach().cpu().numpy().astype(np.float32, copy=False)
                if all_norm_next_np.shape[0] > k_max:
                    all_norm_next_np = all_norm_next_np[:k_max]
                cands_next_np = _chunk_wrist_selector_features_np(
                    all_norm_next_np,
                    selector_chunk_len=int(selector_chunk_size),
                    wrist_dim=int(selector_wrist_dim),
                )
                n = min(cands_next_np.shape[0], next_cands_pad.shape[0])
                next_cands_pad[:n] = cands_next_np[:n]

        chunk_action = np.zeros((int(n_action_steps), int(action_dim)), dtype=np.float32)
        # Seed with predicted chunk first, then overwrite executed timesteps.
        t_seed = min(int(pred_t.shape[1]), int(n_action_steps))
        if t_seed > 0:
            chunk_action[:t_seed] = (
                pred_t[0, :t_seed]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
        max_fill = min(int(chunk_exec_steps), len(traj) - t, int(n_action_steps))
        reward_sum = 0.0
        done_flag = False
        for j in range(max_fill):
            trj = traj[t + j]
            a = np.asarray(trj.get("actions"), dtype=np.float32).reshape(-1)
            chunk_action[j, : min(a.size, int(action_dim))] = a[: int(action_dim)]
            reward_sum += float(np.asarray(trj.get("rewards", 0.0), dtype=np.float32))
            done_flag = bool(done_flag or trj.get("dones", False))

        
        chosen_cand_exec = np.zeros((int(q_selector_action_dim),), dtype=np.float32)
        wd = int(max(1, min(int(selector_wrist_dim), int(action_dim))))
        tl = int(max(1, int(selector_chunk_size)))
        exec_feat = np.zeros((tl, wd), dtype=np.float32)
        use_t = min(tl, chunk_action.shape[0])
        exec_feat[:use_t, :] = chunk_action[:use_t, :wd]
        flat_exec = exec_feat.reshape(-1)
        n_put = min(flat_exec.size, chosen_cand_exec.size)
        if n_put > 0:
            chosen_cand_exec[:n_put] = flat_exec[:n_put]
        chosen_cand = (
            chosen_cand_exec
            if str(selector_chosen_source) == "executed"
            else chosen_cand_imle
        )

        rec = {
            "observations": {
                "state": np.asarray(tr["observations"]["state"], dtype=np.float32),
                "imle_preproj": np.asarray(tr["observations"]["imle_preproj"], dtype=np.float32),
            },
            "actions": chunk_action.reshape(-1).astype(np.float32, copy=False),
            "next_observations": {
                "state": np.asarray(
                    traj[min(t + int(chunk_exec_steps), len(traj) - 1)]["observations"]["state"],
                    dtype=np.float32,
                ),
                "imle_preproj": np.asarray(
                    traj[min(t + int(chunk_exec_steps), len(traj) - 1)]["observations"]["imle_preproj"],
                    dtype=np.float32,
                ),
            },
            "rewards": np.float32(reward_sum),
            "masks": np.float32(1.0 - float(done_flag)),
            "dones": bool(done_flag),
            "selector_chosen_cand": chosen_cand.astype(np.float32, copy=False),
            "selector_next_cands": next_cands_pad.astype(np.float32, copy=False),
            "selector_h_scores": h_scores_pad.astype(np.float32, copy=False),
        }
        out_recs.append(rec)
        t += int(chunk_exec_steps)

    return out_recs


def main() -> None:
    hydra_overrides, argv_rest = _split_hydra_overrides(sys.argv[1:])

    p = argparse.ArgumentParser(description="Postprocess demo pkl for residual RL.")
    p.add_argument("--input_pkl", type=str, default="")
    p.add_argument("--input_zarr", type=str, default="")
    p.add_argument("--output_pkl", type=str, required=True, help="Output pickle for step-level replay")
    p.add_argument(
        "--output_selector_chunk_pkl",
        type=str,
        default="",
        help="Optional selector chunk replay output (one record per chunk).",
    )
    p.add_argument("--ckpt_path", type=str, required=True, help="IMLE ckpt for feature extraction")
    p.add_argument("--precise_dex_root", type=str, default=_default_precise_dex_root())
    p.add_argument("--stack_obs_num", type=int, default=2)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--latency_step", type=int, default=0, help="IMLE bidirection_predict_action latency_step")
    p.add_argument("--chunk_exec_steps", type=int, default=8, help="Steps executed per inference when recomputing offline base actions")
    p.add_argument(
        "--reward_mode",
        type=str,
        default="dones_as_terminal_reward",
        choices=("keep_if_exists", "zero", "dones_as_terminal_reward", "info_succeed"),
    )
    p.add_argument(
        "--sac_imle_feat_concat_action_base",
        action="store_true",
        help="Unsupported: leave disabled to use the maintained IMLE pre-projection feature layout.",
    )
    p.add_argument(
        "--keep_obs_images",
        action="store_true",
        dest="keep_obs_images",
        help=(
            "Keep RGB observation keys in observations/next_observations when writing --output_pkl (e.g. "
            "side_policy_256). Default False: remove images after extracting imle_preproj; retain state and imle_preproj."
        ),
    )
    p.add_argument(
        "--add_q_selector_fields",
        action="store_true",
        help="Also populate selector_chosen_cand, selector_next_cands and selector_h_scores for q-selector replay.",
    )
    p.add_argument(
        "--q_num_candidates",
        type=int,
        default=16,
        help="Number of candidates K for selector_next_cands / selector_h_scores when add_q_selector_fields is enabled.",
    )
    p.add_argument(
        "--q_selector_action_dim",
        type=int,
        default=96,
        help="Candidate dimension D; must match q_selector_action_dim in training.",
    )
    p.add_argument(
        "--selector_chunk_size",
        type=int,
        default=16,
        help="Executed-action chunk length for q-selector candidates (default 16).",
    )
    p.add_argument(
        "--selector_wrist_dim",
        type=int,
        default=6,
        help="Leading action dimensions used for selector chunk features (default 6).",
    )
    p.add_argument(
        "--selector_chosen_source",
        type=str,
        default="executed",
        choices=("executed", "imle_selected"),
        help="selector_chosen_cand source: executed=actual action chunk; imle_selected=IMLE/selector candidate.",
    )
    args = p.parse_args(argv_rest)

    if args.sac_imle_feat_concat_action_base:
        p.error(
            "Base-action feature concatenation is not supported; "
            "leave --sac_imle_feat_concat_action_base disabled."
        )
    if bool(args.input_pkl) == bool(args.input_zarr):
        raise ValueError("Specify exactly one input: --input_pkl or --input_zarr")
    input_path = os.path.abspath(args.input_pkl) if args.input_pkl else ""
    input_zarr = os.path.abspath(args.input_zarr) if args.input_zarr else ""
    output_path = os.path.abspath(args.output_pkl)
    ckpt_path = os.path.abspath(args.ckpt_path)

    if input_zarr:
        trajs = _transitions_from_zarr(input_zarr, stack_obs_num=int(args.stack_obs_num))
        raw = None
        layout = "zarr_to_flat_list"
    else:
        with open(input_path, "rb") as f:
            raw = pickle.load(f)
        trajs, layout = _normalize_payload(raw)

    _, policy, device, image_hw, rgb_keys, n_action_steps, action_dim = _load_policy(
        precise_dex_root=args.precise_dex_root,
        ckpt_path=ckpt_path,
        hydra_overrides=hydra_overrides,
        device_override=args.device,
    )

    out_trajs: list[list[dict[str, Any]]] = []
    out_selector_chunk: list[dict[str, Any]] = []
    total = sum(len(t) for t in trajs)
    done_n = 0
    action_mse_sum = 0.0
    action_mse_win_sum = 0.0
    action_mse_steps = 0
    for ti, traj in enumerate(trajs):
        base_seq, next_base_seq = _compute_action_bases_for_traj(
            traj,
            policy=policy,
            device=device,
            rgb_keys=rgb_keys,
            image_hw=image_hw,
            stack_obs_num=int(args.stack_obs_num),
            latency_step=int(args.latency_step),
            chunk_exec_steps=int(args.chunk_exec_steps),
            n_action_steps=int(n_action_steps),
            action_dim=int(action_dim),
        )
        if len(base_seq) != len(traj) or len(next_base_seq) != len(traj):
            raise RuntimeError("Base-action sequence length does not match the trajectory length")

        def _action_chunk_from_idx(idx: int) -> np.ndarray:
            wrist_dim = int(args.selector_wrist_dim)
            chunk_size = int(args.selector_chunk_size)
            vec = np.zeros((chunk_size * wrist_dim,), dtype=np.float32)
            write = 0
            for k in range(chunk_size):
                j = idx + k
                if j >= len(traj):
                    break
                a = np.asarray(traj[j].get("actions"), dtype=np.float32).reshape(-1)
                wrist = a[:wrist_dim] if a.size >= wrist_dim else np.pad(
                    a, (0, max(0, wrist_dim - a.size))
                )[:wrist_dim]
                vec[write : write + wrist_dim] = wrist
                write += wrist_dim
            return vec

        out_traj: list[dict[str, Any]] = []
        for tr_idx, tr in enumerate(traj):
            chosen_vec = _action_chunk_from_idx(tr_idx) if bool(args.add_q_selector_fields) else None
            next_vec = (
                _action_chunk_from_idx(
                    min(tr_idx + int(args.chunk_exec_steps), len(traj) - 1)
                )
                if bool(args.add_q_selector_fields)
                else None
            )
            out_tr = _postprocess_transition(
                tr,
                policy=policy,
                device=device,
                rgb_keys=rgb_keys,
                image_hw=image_hw,
                stack_obs_num=int(args.stack_obs_num),
                concat_action_base=bool(args.sac_imle_feat_concat_action_base),
                reward_mode=str(args.reward_mode),
                drop_obs_images=not bool(args.keep_obs_images),
                add_q_selector_fields=bool(args.add_q_selector_fields),
                q_num_candidates=int(args.q_num_candidates),
                q_selector_action_dim=int(args.q_selector_action_dim),
                chosen_cand_vec=chosen_vec,
                next_cand_vec=next_vec,
            )
            out_tr["actions"] = np.asarray(tr.get("actions"), dtype=np.float32).reshape(-1)
            out_tr["action_base"] = np.asarray(base_seq[tr_idx], dtype=np.float32).reshape(-1)
            out_tr["next_action_base"] = np.asarray(next_base_seq[tr_idx], dtype=np.float32).reshape(-1)

            a_exec = np.asarray(out_tr["actions"], dtype=np.float32).reshape(-1)
            a_pred = np.asarray(out_tr["action_base"], dtype=np.float32).reshape(-1)
            use_dim = min(int(a_exec.size), int(a_pred.size))
            mse_step = 0.0
            if use_dim > 0:
                pred_t = torch.as_tensor(a_pred[:use_dim], dtype=torch.float32).view(1, 1, -1)
                gt_t = torch.as_tensor(a_exec[:use_dim], dtype=torch.float32).view(1, 1, -1)
                
                mse_step = float(F.mse_loss(pred_t, gt_t).item())
            action_mse_sum += mse_step
            action_mse_win_sum += mse_step
            action_mse_steps += 1

            done_n += 1
            if done_n % 100 == 0:
                win_denom = float(min(100, action_mse_steps))
                mse_win = action_mse_win_sum / max(win_denom, 1.0)
                mse_avg = action_mse_sum / max(float(action_mse_steps), 1.0)
                print(
                    f"[postprocess][action_mse] step={done_n}/{total} "
                    f"last={mse_step:.8f} win100_avg={mse_win:.8f} global_avg={mse_avg:.8f}"
                )
                action_mse_win_sum = 0.0
            if done_n % 500 == 0:
                print(f"[postprocess] {done_n}/{total} transitions")
            out_traj.append(out_tr)
        out_trajs.append(out_traj)
        print(f"[postprocess] trajectory {ti + 1}/{len(trajs)} done ({len(traj)} steps)")

        if args.output_selector_chunk_pkl:
            selector_traj: list[dict[str, Any]] = []
            for tr_raw, tr_out in zip(traj, out_traj):
                tr_sel = copy.deepcopy(tr_out)
                
                tr_sel["observations"] = copy.deepcopy(tr_raw["observations"])
                tr_sel["next_observations"] = copy.deepcopy(tr_raw["next_observations"])
                
                tr_sel["observations"]["imle_preproj"] = np.asarray(
                    tr_out["observations"]["imle_preproj"], dtype=np.float32
                )
                tr_sel["next_observations"]["imle_preproj"] = np.asarray(
                    tr_out["next_observations"]["imle_preproj"], dtype=np.float32
                )
                selector_traj.append(tr_sel)

            chunk_recs = _compute_selector_chunk_replay_for_traj(
                selector_traj,
                policy=policy,
                device=device,
                rgb_keys=rgb_keys,
                image_hw=image_hw,
                stack_obs_num=int(args.stack_obs_num),
                latency_step=int(args.latency_step),
                chunk_exec_steps=int(args.chunk_exec_steps),
                n_action_steps=int(n_action_steps),
                action_dim=int(action_dim),
                q_num_candidates=int(args.q_num_candidates),
                q_selector_action_dim=int(args.q_selector_action_dim),
                selector_chunk_size=int(args.selector_chunk_size),
                selector_wrist_dim=int(args.selector_wrist_dim),
                selector_chosen_source=str(args.selector_chosen_source),
            )
            out_selector_chunk.extend(chunk_recs)

    if layout == "zarr_to_flat_list":
        out_payload = [tr for ep in out_trajs for tr in ep]
    else:
        out_payload = _restore_payload_layout(raw, out_trajs, layout)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(out_payload, f)

    if args.output_selector_chunk_pkl:
        out_sel = os.path.abspath(args.output_selector_chunk_pkl)
        os.makedirs(os.path.dirname(out_sel) or ".", exist_ok=True)
        with open(out_sel, "wb") as f:
            pickle.dump(out_selector_chunk, f)

    print("=" * 72)
    if input_zarr:
        print(f"Input(zarr)      : {input_zarr}")
    else:
        print(f"Input(pkl)       : {input_path}")
    print(f"Output           : {output_path}")
    if args.output_selector_chunk_pkl:
        print(f"Output(selector) : {os.path.abspath(args.output_selector_chunk_pkl)}")
        print(f"Selector chunks  : {len(out_selector_chunk)}")
    print(f"Layout           : {layout}")
    print(f"Transitions      : {total}")
    print(f"reward_mode      : {args.reward_mode}")
    print(f"concat_action_base: {bool(args.sac_imle_feat_concat_action_base)}")
    print(f"keep_obs_images  : {bool(args.keep_obs_images)}  (retain RGB observation keys in output pickle)")
    print(f"add_q_selector_fields: {bool(args.add_q_selector_fields)}")
    if bool(args.add_q_selector_fields):
        print(f"q_num_candidates : {int(args.q_num_candidates)}")
        print(f"q_selector_action_dim: {int(args.q_selector_action_dim)}")
        print(f"selector_chunk_size: {int(args.selector_chunk_size)}")
        print(f"selector_chosen_source: {args.selector_chosen_source}")
    print("=" * 72)


if __name__ == "__main__":
    main()

