#!/usr/bin/env python3
"""
Real-robot IMLE evaluation for Franka + LEAP. Observations follow pkl2zarr: stacked RGB
streams, normalized measured LEAP joints, and TCP state slices. The policy supplies
22-dimensional float32 actions. See README.md for commands. After reset, r starts and q quits;
space labels success and f labels failure on the next step. SpaceMouse intervention is opt-in.
LEAP uses its Python API directly. Successful episodes overwrite the checkpoint evaluation
video when camera frames are available.
"""

from __future__ import annotations

import argparse
import os
import queue
import re
import sys
import time
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import hydra
import cv2
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


_GREEN = "\033[92m" if sys.stdout.isatty() else ""
_RESET = "\033[0m" if sys.stdout.isatty() else ""


class _UserQuit(Exception):
    """Raised when the user presses q during an episode."""


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
STATE_SLICES: list[tuple[str, slice]] = [
    ("leap_pose", slice(0, 16)),
    ("tcp_force", slice(16, 19)),
    ("tcp_pose", slice(19, 25)),
    ("tcp_torque", slice(25, 28)),
    ("tcp_vel", slice(28, 34)),
]


def _default_precise_dex_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "imle_dex", "precise_dex"))


def _prepend_sys_path(path: str) -> None:
    path = os.path.abspath(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def _default_imle_examples_root() -> str:
    """Return imle_dex/examples, containing action_scaling.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "imle_dex", "examples"))


_prepend_sys_path(_default_imle_examples_root())
from action_scaling import normalize_leap_hand_physical  # noqa: E402

_HYDRA_KEY_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")


def _split_hydra_overrides(argv: list[str]) -> tuple[list[str], list[str]]:
    hydra_overrides: list[str] = []
    rest: list[str] = []
    for tok in argv:
        if ("=" in tok) and (not tok.startswith("--")):
            key, _, _ = tok.partition("=")
            if "/" not in key and "\\" not in key and _HYDRA_KEY_RE.fullmatch(key):
                hydra_overrides.append(tok)
            else:
                rest.append(tok)
        else:
            rest.append(tok)
    return hydra_overrides, rest


def _stack_leap_pose_online(leap_hw_hist: list[np.ndarray]) -> np.ndarray:
    """Match pkl2zarr hand-history stacking, with the current sample at the end."""
    h = len(leap_hw_hist)
    if h == 0:
        raise ValueError("leap_hw_hist empty")
    obs_horizon = h
    leap_seq = [normalize_leap_hand_physical(x) for x in leap_hw_hist]
    i = obs_horizon - 1
    start = max(0, i - obs_horizon + 1)
    hist = leap_seq[start : i + 1]
    if len(hist) < obs_horizon:
        hist = [leap_seq[0]] * (obs_horizon - len(hist)) + hist
    return np.stack(hist, axis=0).reshape(-1).astype(np.float32)


def _slice_tcp_fields_from_state(state_stack: np.ndarray) -> dict[str, np.ndarray]:
    """Extract fields from the (H, 34) state emitted by ChunkingWrapper."""
    out: dict[str, np.ndarray] = {}
    for name, sl in STATE_SLICES:
        if name == "leap_pose":
            continue
        part = np.asarray(state_stack[:, sl], dtype=np.float32).reshape(-1)
        out[name] = part
    return out


def _process_rgb_hwc_uint8(
    img: np.ndarray, out_hw: tuple[int, int]
) -> torch.Tensor:
    """Match dataset image preprocessing; return (1, 3, H, W)."""
    rgb = torch.from_numpy(np.asarray(img[..., :3], dtype=np.float32))  # H,W,3
    rgb = rgb.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    rgb = F.interpolate(
        rgb / 255.0,
        size=out_hw,
        mode="bilinear",
        align_corners=False,
    )
    return rgb


def build_policy_obs(
    obs: dict[str, Any],
    *,
    rgb_keys: list[str],
    image_hw: tuple[int, int],
    n_obs_steps: int,
    stack_obs_num: int,
    leap_hist: deque[np.ndarray],
) -> dict[str, torch.Tensor]:
    """Build the observation dictionary expected by the training-time Timm encoder."""
    if n_obs_steps != 1:
        raise NotImplementedError("Only cfg.n_obs_steps == 1 is supported by this Franka dataset configuration")

    side = np.asarray(obs["side_policy_256"])
    if side.ndim == 3:
        side = side[None, ...]
    if side.ndim != 4 or int(side.shape[0]) != int(len(rgb_keys)):
        raise ValueError(
            f"Expected {len(rgb_keys)} RGB streams in the first side_policy_256 dimension, got {side.shape}"
        )

    obs_dict: dict[str, torch.Tensor] = {}
    for i, key in enumerate(rgb_keys):
        timg = _process_rgb_hwc_uint8(side[i], image_hw)
        obs_dict[key] = timg.unsqueeze(1).contiguous()  # (1, 1, 3, H, W)

    st = np.asarray(obs["state"], dtype=np.float32)
    if st.ndim == 1:
        st = st.reshape(stack_obs_num, -1)
    tcp = _slice_tcp_fields_from_state(st)

    if len(leap_hist) < stack_obs_num:
        raise ValueError("Insufficient leap_hist entries; initialize the deque with the current leap_hw before calling")
    leap_pose = _stack_leap_pose_online(list(leap_hist))

    for k, v in tcp.items():
        obs_dict[k] = torch.from_numpy(v.astype(np.float32)).reshape(1, 1, -1)

    obs_dict["leap_pose"] = torch.from_numpy(leap_pose.astype(np.float32)).reshape(1, 1, -1)
    return obs_dict


# ============================ robot ========================================

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_IMLE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "imle_dex"))
_DEFAULT_LEAP_API = os.path.normpath(os.path.join(_HERE, "..", "LEAP_Hand_API", "python"))


def _setup_imle_paths(imle_root: str) -> None:
    root = os.path.abspath(imle_root)
    _prepend_sys_path(os.path.join(root, "examples"))
    _prepend_sys_path(os.path.join(root, "serl_robot_infra"))
    _prepend_sys_path(os.path.join(root, "serl_launcher"))


class _DummyVideoCapture:
    def __init__(self, width: int, height: int, name: str = "") -> None:
        self._shape = (height, width, 3)

    def read(self) -> np.ndarray:
        return np.zeros(self._shape, dtype=np.uint8)

    def close(self) -> None:
        pass


def _install_dummy_realsense_cameras(exp_name: str) -> None:
    import importlib
    from collections import OrderedDict

    from franka_env.envs.franka_env import FrankaEnv

    importlib.import_module(f"experiments.{exp_name}.wrapper")

    def _dummy_init_cameras(self: Any, name_serial_dict: dict[str, Any] | None = None) -> None:
        if getattr(self, "cap", None) is not None:
            self.close_cameras()
        self.cap = OrderedDict()
        for cam_name, kwargs in (name_serial_dict or {}).items():
            dim = kwargs.get("dim", (1280, 720))
            w, h = int(dim[0]), int(dim[1])
            self.cap[cam_name] = _DummyVideoCapture(width=w, height=h, name=cam_name)

    FrankaEnv.init_cameras = _dummy_init_cameras  # type: ignore[assignment]
    for cls in FrankaEnv.__subclasses__():
        cls.init_cameras = _dummy_init_cameras  # type: ignore[assignment]
    print("[CAMERAS] --no_cameras: using black placeholder images")


class _StoppedFrankaListener:
    def stop(self) -> None:
        pass

    def join(self, timeout: float | None = None) -> None:
        pass


def _stop_franka_pynput_listener(env: Any) -> None:
    base = getattr(env, "unwrapped", env)
    listener = getattr(base, "listener", None)
    if listener is None:
        return
    try:
        listener.stop()
        listener.join(timeout=2.0)
    except Exception as e:
        print(f"[KEYBOARD] Could not stop FrankaEnv listener: {e}")
        return
    time.sleep(0.5)
    try:
        setattr(base, "listener", _StoppedFrankaListener())
    except Exception:
        pass
    print(
        "[KEYBOARD] FrankaEnv ESC listener stopped (avoid multiple pynput / X11 RECORD hooks)."
    )


def _start_r_q_listener() -> tuple[queue.Queue[str], Any]:
    """Global r/q keyboard listener using pynput."""
    from pynput import keyboard

    cmd_queue: queue.Queue[str] = queue.Queue()

    def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
        try:
            if hasattr(key, "char") and key.char:
                c = key.char.lower()
                if c in ("r", "q"):
                    cmd_queue.put(c)
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return cmd_queue, listener


def _wait_for_r_after_reset(cmd_queue: queue.Queue[str]) -> bool:
    """Wait after reset: r starts; q returns False to quit."""
    print("[KEY] Reset complete: press [r] to start this episode or [q] to quit.")
    while True:
        try:
            while True:
                c = cmd_queue.get_nowait()
                if c == "q":
                    print("[KEY] q received: quitting.")
                    return False
                if c == "r":
                    print("[KEY] r received: starting this episode.")
                    return True
        except queue.Empty:
            pass
        time.sleep(0.05)


def _quit_pending(cmd_queue: queue.Queue[str]) -> bool:
    try:
        while True:
            if cmd_queue.get_nowait() == "q":
                return True
    except queue.Empty:
        return False


def _fill_leap_hist_after_reset(
    leap_hist: deque[np.ndarray],
    stack_obs_num: int,
    read_leap_hw: Any,
) -> None:
    lp0 = read_leap_hw()
    if lp0 is None:
        lp0 = np.zeros(16, dtype=np.float32)
    lp0 = np.asarray(lp0, dtype=np.float32).reshape(16)
    leap_hist.clear()
    for _ in range(int(stack_obs_num)):
        leap_hist.append(lp0.copy())


def _extract_side_policy_frame(obs: dict[str, Any]) -> np.ndarray | None:
    """Extract the latest side_policy_256 RGB frame as uint8 HWC."""
    if "side_policy_256" not in obs:
        return None
    frame = np.asarray(obs["side_policy_256"])
    if frame.ndim == 4:
        
        frame = frame[-1]
    if frame.ndim != 3 or frame.shape[-1] < 3:
        return None
    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _extract_demo_raw_frame(env: Any) -> np.ndarray | None:
    """Return the latest full-resolution demo frame from FrankaEnv, or None."""
    base = getattr(env, "unwrapped", env)
    rec = getattr(base, "recording_frames", None)
    if not isinstance(rec, list) or len(rec) == 0:
        return None
    latest = rec[-1]
    if not isinstance(latest, dict) or "demo" not in latest:
        return None
    frame = np.asarray(latest["demo"])
    if frame.ndim != 3 or frame.shape[-1] < 3:
        return None
    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def run_robot(
    cfg: OmegaConf,
    policy: torch.nn.Module,
    *,
    exp_name: str,
    imle_root: str,
    stack_obs_num: int,
    fake_env: bool,
    no_cameras: bool,
    no_leap: bool,
    leap_api_root: str,
    use_spacemouse: bool,
    use_keyboard_success: bool,
    max_episodes: int,
    max_episode_steps: int,
    control_hz: float,
    image_hw: tuple[int, int],
    latency_step: int = 0,
    chunk_exec_steps: int = 1,
) -> None:
    _setup_imle_paths(imle_root)
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    from experiments.mappings import CONFIG_MAPPING

    if exp_name not in CONFIG_MAPPING:
        raise ValueError(f"Unknown exp_name={exp_name!r}")

    train_cfg = CONFIG_MAPPING[exp_name]()

    if no_cameras:
        _install_dummy_realsense_cameras(exp_name)

    leap_node: Any = None
    if not no_leap:
        api_root = os.path.abspath(leap_api_root)
        _prepend_sys_path(api_root)
        try:
            from main import LeapNode  # LEAP_Hand_API/python/main.py

            leap_node = LeapNode()
            print(f"[LEAP] LeapNode initialized (hardware on, api={api_root}).")
        except Exception as e:
            raise RuntimeError(
                f"Could not import and initialize LeapNode from {api_root}."
                f"Check --leap_api_root or use --no_leap. Original error: {e}"
            ) from e

    env = train_cfg.get_environment(
        fake_env=fake_env,
        save_video=False,
        stack_obs_num=int(stack_obs_num),
        leap_node=leap_node,
        leap_send_commands=bool(leap_node is not None),
        use_spacemouse=True,
        use_keyboard_success=bool(use_keyboard_success),
    )
    _stop_franka_pynput_listener(env)

    if use_keyboard_success:
        print(
            "[ROBOT] Keyboard labels: space=success, f=failure; applied on the next env.step, ending the episode"
        )

    cmd_queue, key_listener = _start_r_q_listener()

    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)
    action_dim = int(cfg.task.shape_meta["action"]["shape"][0])
    latency_step = int(latency_step)
    chunk_exec_steps = int(chunk_exec_steps)
    if chunk_exec_steps < 1:
        raise ValueError(f"chunk_exec_steps must be >= 1, got {chunk_exec_steps}")
    if n_action_steps <= 8 + latency_step:
        raise ValueError(
            "bidirection_predict_action requires "
            f"n_action_steps ({n_action_steps}) > 8 + latency_step ({8 + latency_step})。"
            "Increase the trajectory length with a Hydra override, e.g. n_action_steps=40."
        )
    policy.eval()
    device = policy.device

    leap_hist: deque[np.ndarray] = deque(maxlen=int(stack_obs_num))

    obs_meta = cfg.task.shape_meta["obs"]
    rgb_keys = [k for k in obs_meta if obs_meta[k].get("type") == "rgb"]
    n_rgb = len(rgb_keys)
    if int(stack_obs_num) != int(n_rgb):
        raise ValueError(
            f"--stack_obs_num={stack_obs_num} does not match the {n_rgb} RGB streams in Hydra shape_meta."
        )

    def read_leap_hw() -> np.ndarray | None:
        if leap_node is None:
            return None
        try:
            pos, _, _ = leap_node.pos_vel_eff_srv()
            return np.asarray(pos, dtype=np.float32).reshape(16)
        except Exception:
            try:
                return np.asarray(leap_node.read_pos(), dtype=np.float32).reshape(16)
            except Exception:
                return None

    obs, _info = env.reset()
    _fill_leap_hist_after_reset(leap_hist, stack_obs_num, read_leap_hw)

    dt = 1.0 / max(float(control_hz), 1e-3)
    ckpt_path = os.path.abspath(str(cfg.optimizer.start_ckpt_path))
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_parent_dir = os.path.dirname(ckpt_dir)
    eval_dir = os.path.join(ckpt_parent_dir, "eval")
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    eval_video_path = os.path.join(eval_dir, f"{ckpt_name}_eval.mp4")
    print(
        f"[ROBOT] max_episodes={max_episodes} max_episode_steps={max_episode_steps} "
        f"control_hz={control_hz} stack_obs_num={stack_obs_num} "
        f"chunk_exec_steps={chunk_exec_steps} latency_step={latency_step} device={device}"
    )
    print(
        f"[ROBOT] Save side_policy evaluation video only on success, overwriting {eval_video_path}"
    )
    try:
        if not _wait_for_r_after_reset(cmd_queue):
            return

        success_tally = 0
        inference_total_s = 0.0
        inference_calls = 0
        for ep_idx in range(int(max_episodes)):
            if _quit_pending(cmd_queue):
                print("[KEY] q received during rollout: quitting.")
                break

            ep_step = 0
            done = False
            trunc = False
            ep_done = False
            ep_succeed = False
            video_frames: list[np.ndarray] = []
            demo_video_frames: list[np.ndarray] = []
            prev_traj = torch.randn(
                (1, n_action_steps, action_dim), device=device, dtype=torch.float32
            )
            while ep_step < int(max_episode_steps):
                if _quit_pending(cmd_queue):
                    print("[KEY] q received during rollout: quitting.")
                    raise _UserQuit()

                t0 = time.time()

                lp = read_leap_hw()
                if lp is None:
                    lp = leap_hist[-1] if leap_hist else np.zeros(16, dtype=np.float32)
                lp = np.asarray(lp, dtype=np.float32).reshape(16)
                leap_hist.append(lp)

                obs_t = build_policy_obs(
                    obs,
                    rgb_keys=rgb_keys,
                    image_hw=image_hw,
                    n_obs_steps=n_obs_steps,
                    stack_obs_num=int(stack_obs_num),
                    leap_hist=leap_hist,
                )
                obs_t = {k: v.to(device, non_blocking=True) for k, v in obs_t.items()}

                inf_t0 = time.perf_counter()
                with torch.no_grad():
                    result = policy.bidirection_predict_action(
                        obs_t, None, prev_traj, latency_step=latency_step
                    )
                inference_total_s += time.perf_counter() - inf_t0
                inference_calls += 1
                pred_t = result["action"]
                pred_chunk = pred_t[0].detach().cpu().numpy().astype(np.float32)
                prev_traj = pred_t.detach()

                n_exec = min(
                    chunk_exec_steps,
                    int(pred_chunk.shape[0]),
                    int(max_episode_steps) - ep_step,
                )
                if _quit_pending(cmd_queue):
                    print("[KEY] q received during rollout: quitting.")
                    raise _UserQuit()
                for j in range(n_exec):
                    if j > 0:
                        if _quit_pending(cmd_queue):
                            print("[KEY] q received during rollout: quitting.")
                            raise _UserQuit()
                        t0 = time.time()
                        lp = read_leap_hw()
                        if lp is None:
                            lp = leap_hist[-1] if leap_hist else np.zeros(16, dtype=np.float32)
                        lp = np.asarray(lp, dtype=np.float32).reshape(16)
                        leap_hist.append(lp)

                    action_numpy = pred_chunk[j].reshape(action_dim)
                    obs, _rew, done, trunc, _info = env.step(action_numpy)
                    ep_done = bool(ep_done or done)
                    if _info.get("keyboard_failure"):
                        ep_succeed = False
                        print("[KEYBOARD] f: episode labeled as failure (succeed=False)")
                    elif _info.get("keyboard_success"):
                        ep_succeed = True
                        print(f"{_GREEN}[KEYBOARD] space: episode labeled as success (succeed=True){_RESET}")
                    elif done or trunc:
                        if "succeed" in _info:
                            ep_succeed = bool(_info["succeed"])
                        else:
                            ep_succeed = bool(
                                ep_succeed or _info.get("succeed", False)
                            )
                    else:
                        ep_succeed = bool(
                            ep_succeed or _info.get("succeed", False)
                        )
                    side_frame = _extract_side_policy_frame(obs)
                    if side_frame is not None:
                        video_frames.append(side_frame)
                    demo_frame = _extract_demo_raw_frame(env)
                    if demo_frame is not None:
                        demo_video_frames.append(demo_frame)

                    ep_step += 1

                    elapsed = time.time() - t0
                    if elapsed < dt:
                        time.sleep(dt - elapsed)

                    if done or trunc:
                        print(
                            f"[ROBOT] episode {ep_idx + 1}/{max_episodes} ended early "
                            f"(step={ep_step}/{max_episode_steps}) done={done} trunc={trunc}"
                        )
                        break

                if done or trunc:
                    break

            if ep_step >= int(max_episode_steps) and not (done or trunc):
                print(
                    f"[ROBOT] episode {ep_idx + 1}/{max_episodes} reached the per-episode step limit "
                    f"max_episode_steps={max_episode_steps}"
                )

            if ep_succeed:
                success_tally += 1

            total_completed = ep_idx + 1
            rate = success_tally / total_completed if total_completed else 0.0
            this_ok = "success" if ep_succeed else "failure"
            print(
                f"{_GREEN}[EPISODE] completed={total_completed} | "
                f"successes={success_tally} | success_rate={rate:.1%} | result={this_ok}{_RESET}"
            )
            if inference_calls > 0:
                avg_inf_ms = 1000.0 * inference_total_s / float(inference_calls)
                print(
                    f"[ROBOT] Mean inference time: {avg_inf_ms:.3f} ms "
                    f"(over {inference_calls} calls, accumulated across episodes)"
                )

            save_frames = demo_video_frames if len(demo_video_frames) > 0 else video_frames
            save_tag = "demo(raw-fullres)" if len(demo_video_frames) > 0 else "side_policy_256"

            if ep_succeed and len(save_frames) > 0:
                os.makedirs(eval_dir, exist_ok=True)
                h, w = save_frames[0].shape[:2]
                writer = cv2.VideoWriter(
                    eval_video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(control_hz),
                    (w, h),
                )
                for frm in save_frames:
                    writer.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
                writer.release()
                print(
                    f"{_GREEN}[VIDEO] Successful episode: saved {save_tag} video (overwrite): "
                    f"{eval_video_path}{_RESET}"
                )
            elif ep_succeed and len(save_frames) == 0:
                print(
                    "[ROBOT] Successful episode has no demo/side_policy frames; skipping video."
                )

            if ep_idx + 1 >= int(max_episodes):
                print(f"[ROBOT] Completed {max_episodes} episodes; exiting.")
                break

            obs, _info = env.reset()
            _fill_leap_hist_after_reset(leap_hist, stack_obs_num, read_leap_hw)
            if not _wait_for_r_after_reset(cmd_queue):
                break

    except _UserQuit:
        pass
    finally:
        try:
            key_listener.stop()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass


def main() -> None:
    argv = sys.argv[1:]
    hydra_overrides, argv_rest = _split_hydra_overrides(argv)

    parser = argparse.ArgumentParser(description="IMLE PsiPolicy: Franka + LEAP real-robot evaluation")
    # robot
    parser.add_argument("--exp_name", type=str, default="pick_place")
    parser.add_argument("--imle_root", type=str, default=_DEFAULT_IMLE_ROOT)
    parser.add_argument("--stack_obs_num", type=int, default=2)
    parser.add_argument("--fake_env", action="store_true")
    parser.add_argument("--no_cameras", action="store_true")
    parser.add_argument(
        "--no_leap",
        action="store_true",
        help="Do not connect LEAP hardware; use placeholder hand observations",
    )
    parser.add_argument(
        "--leap_api_root",
        type=str,
        default=_DEFAULT_LEAP_API,
        help="LEAP_Hand_API/python directory containing main.py.LeapNode.",
    )
    parser.add_argument(
        "--use_spacemouse",
        action="store_true",
        help="Enable SpaceMouse arm intervention on action[0:6]; disabled by default",
    )
    parser.add_argument(
        "--no_keyboard_success",
        action="store_true",
        help="Disable keyboard success/failure labels through use_keyboard_success=False",
    )
    parser.add_argument("--max_episodes", type=int, default=10, help="Maximum number of evaluation episodes")
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=250,
        help="Maximum env.step calls per episode, independent of environment truncation",
    )
    parser.add_argument("--control_hz", type=float, default=10.0)
    parser.add_argument(
        "--latency_step",
        type=int,
        default=0,
        help="Latency alignment steps for bidirection_predict_action; "
        "if >0, requires n_action_steps > 8 + latency_step (set with Hydra).",
    )
    parser.add_argument(
        "--chunk_exec_steps",
        type=int,
        default=1,
        help="Environment steps executed per predicted chunk, capped by the horizon and remaining episode steps.",
    )
    # common
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--precise_dex_root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv_rest)

    precise_root = os.path.abspath(args.precise_dex_root or _default_precise_dex_root())
    if not os.path.isdir(precise_root):
        raise FileNotFoundError(f"precise_dex_root not found: {precise_root}")

    _prepend_sys_path(precise_root)
    os.chdir(precise_root)
    from utils import setup_path_variables

    setup_path_variables()

    
    
    hydra_config_dir = os.path.abspath(os.path.join(precise_root, "psi_policy", "config"))
    if not os.path.isdir(hydra_config_dir):
        raise FileNotFoundError(f"Hydra configuration directory does not exist: {hydra_config_dir}")
    with initialize_config_dir(version_base=None, config_dir=hydra_config_dir):
        cfg = compose(config_name="psi_policy_workspace", overrides=hydra_overrides)
    OmegaConf.resolve(cfg)

    if args.ckpt_path:
        cfg.optimizer.start_ckpt_path = args.ckpt_path
    if cfg.optimizer.start_ckpt_path in (None, "", "null"):
        raise ValueError("Provide a checkpoint with --ckpt_path or optimizer.start_ckpt_path=...")

    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    policy = workspace.model

    if args.device:
        dev = torch.device(args.device)
    else:
        dev = torch.device(cfg.training.device if cfg.training.get("device") else "cuda:0")
    policy.to(dev)
    policy.eval()

    image_shape = list(cfg.task.image_shape)
    if len(image_shape) != 3:
        raise ValueError(f"Expected task.image_shape=[C,H,W], got {image_shape}")
    _, h224, w224 = image_shape
    image_hw = (int(h224), int(w224))

    run_robot(
        cfg,
        policy,
        exp_name=args.exp_name,
        imle_root=args.imle_root,
        stack_obs_num=int(args.stack_obs_num),
        fake_env=bool(args.fake_env),
        no_cameras=bool(args.no_cameras),
        no_leap=bool(args.no_leap),
        leap_api_root=str(args.leap_api_root),
        use_spacemouse=bool(args.use_spacemouse),
        use_keyboard_success=not bool(args.no_keyboard_success),
        max_episodes=int(args.max_episodes),
        max_episode_steps=int(args.max_episode_steps),
        control_hz=float(args.control_hz),
        image_hw=image_hw,
        latency_step=int(args.latency_step),
        chunk_exec_steps=int(args.chunk_exec_steps),
    )


if __name__ == "__main__":
    main()
