#!/usr/bin/env python3
"""
Record Franka + LEAP demos with manual keyboard success labeling.

Success criterion is keyboard driven (global hotkeys via pynput), not
info["succeed"] from classifier wrappers.

Output format:
- pickle payload: {"transitions": [...], "meta": {...}}
- h5 payload: episodes/ep_xxxx groups for replay/inspection.

Keyboard (global, focus not required on terminal):
  r : start recording a new trajectory (clear trajectory)
  s : save current trajectory as success (append to dataset, reset env if enabled)
  d : discard current trajectory
  q : quit and flush files
s
IMPORTANT: Even with classifier disabled, FrankaEnv may set done=True when the
TCP is within REWARD_THRESHOLD of TARGET_POSE, or when MAX_EPISODE_LENGTH is
reached. For keyboard-only success labeling, widen REWARD_THRESHOLD and/or raise
MAX_EPISODE_LENGTH in your task EnvConfig so you do not end episodes early by
accident.

Usage:
See the root README.md for the maintained pick_place collection command.
"""

from __future__ import annotations

import copy
import datetime
import os
import pickle as pkl
import queue
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths: override imle root with --imle_root if needed.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_LEAPHAND_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_DEFAULT_IMLE_ROOT = os.path.join(_LEAPHAND_ROOT, "imle_dex")
_DEFAULT_LEAP_API = os.path.join(_LEAPHAND_ROOT, "LEAP_Hand_API", "python")


def _prepend_sys_path(path: str) -> None:
    path = os.path.abspath(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def _setup_imle_paths(imle_root: str) -> None:
    root = os.path.abspath(imle_root)
    _prepend_sys_path(os.path.join(root, "examples"))
    _prepend_sys_path(os.path.join(root, "serl_robot_infra"))
    _prepend_sys_path(os.path.join(root, "serl_launcher"))


# Deferred imports after absl parses flags -- see main().


def _image_keys_to_strip(config: Any) -> list[str]:
    keys: list[str] = []
    for name in ("image_keys", "classifier_keys"):
        ks = getattr(config, name, None)
        if ks:
            keys.extend(list(ks))
    return list(dict.fromkeys(keys))


def _strip_images_from_obs(
    obs: dict[str, Any], keys_to_remove: list[str]
) -> dict[str, Any]:
    if not keys_to_remove:
        return copy.deepcopy(obs)
    out = {}
    for k, v in obs.items():
        if k in keys_to_remove:
            continue
        out[k] = copy.deepcopy(v) if hasattr(v, "copy") else v
    return out


def _prepare_obs_for_save(
    obs: dict[str, Any],
    save_images: bool,
    keys_to_strip: list[str],
) -> dict[str, Any]:
    if save_images:
        return copy.deepcopy(obs)
    return _strip_images_from_obs(obs, keys_to_strip)


def _safe_deepcopy_info(info: dict[str, Any]) -> dict[str, Any]:
    try:
        return copy.deepcopy(info)
    except Exception:
        out: dict[str, Any] = {}
        for k, v in info.items():
            try:
                if isinstance(v, np.ndarray):
                    out[k] = v.copy()
                elif isinstance(v, (np.floating, np.integer)):
                    out[k] = v.item()
                else:
                    out[k] = copy.deepcopy(v)
            except Exception:
                out[k] = str(v)
        return out


def _write_episode_h5(
    h5f: Any,
    ep_idx: int,
    episode_transitions: list[dict[str, Any]],
    save_images: bool,
    image_keys_present: list[str],
) -> None:
    grp = h5f.create_group(f"episodes/ep_{ep_idx:04d}")
    T = len(episode_transitions)
    grp.attrs["length"] = T

    actions = np.stack([np.asarray(t["actions"], dtype=np.float32) for t in episode_transitions])
    rewards = np.asarray([t["rewards"] for t in episode_transitions], dtype=np.float32)
    masks = np.asarray([t["masks"] for t in episode_transitions], dtype=np.float32)
    dones = np.asarray([int(t["dones"]) for t in episode_transitions], dtype=np.uint8)

    grp.create_dataset("actions", data=actions, compression="gzip", compression_opts=1)
    grp.create_dataset("rewards", data=rewards, compression="gzip", compression_opts=1)
    grp.create_dataset("masks", data=masks, compression="gzip", compression_opts=1)
    grp.create_dataset("dones", data=dones, compression="gzip", compression_opts=1)

    obs_state = np.stack([t["observations"]["state"] for t in episode_transitions])
    next_state = np.stack([t["next_observations"]["state"] for t in episode_transitions])
    grp.create_dataset("obs_state", data=obs_state, compression="gzip", compression_opts=1)
    grp.create_dataset("next_obs_state", data=next_state, compression="gzip", compression_opts=1)

    leap_pos = np.stack(
        [
            np.asarray(
                t.get("leap_hw_pos"),
                dtype=np.float32,
            )
            if t.get("leap_hw_pos") is not None
            else np.full((16,), np.nan, dtype=np.float32)
            for t in episode_transitions
        ]
    )
    leap_vel = np.stack(
        [
            np.asarray(t.get("leap_hw_vel"), dtype=np.float32)
            if t.get("leap_hw_vel") is not None
            else np.full((16,), np.nan, dtype=np.float32)
            for t in episode_transitions
        ]
    )
    leap_cur = np.stack(
        [
            np.asarray(t.get("leap_hw_cur"), dtype=np.float32)
            if t.get("leap_hw_cur") is not None
            else np.full((16,), np.nan, dtype=np.float32)
            for t in episode_transitions
        ]
    )
    grp.create_dataset("leap_hw_pos", data=leap_pos, compression="gzip", compression_opts=1)
    grp.create_dataset("leap_hw_vel", data=leap_vel, compression="gzip", compression_opts=1)
    grp.create_dataset("leap_hw_cur", data=leap_cur, compression="gzip", compression_opts=1)
    # Aliases to reuse the existing `replay_leaphand_h5.py` script.
    grp.create_dataset("hw_pos", data=leap_pos, compression="gzip", compression_opts=1)
    grp.create_dataset("hw_vel", data=leap_vel, compression="gzip", compression_opts=1)
    grp.create_dataset("hw_cur", data=leap_cur, compression="gzip", compression_opts=1)

    leap_cmd = np.stack(
        [
            np.asarray(t.get("leap_cmd"), dtype=np.float32)
            if t.get("leap_cmd") is not None
            else np.full((16,), np.nan, dtype=np.float32)
            for t in episode_transitions
        ]
    )
    grp.create_dataset("leap_cmd", data=leap_cmd, compression="gzip", compression_opts=1)
    grp.create_dataset("cmd_leap_q", data=leap_cmd, compression="gzip", compression_opts=1)

    if save_images and image_keys_present:
        obs_grp = grp.create_group("obs_images")
        next_grp = grp.create_group("next_obs_images")
        for key in image_keys_present:
            if key not in episode_transitions[0]["observations"]:
                continue
            obs_img = np.stack([t["observations"][key] for t in episode_transitions])
            next_img = np.stack([t["next_observations"][key] for t in episode_transitions])
            safe = key.replace("/", "_")
            obs_grp.create_dataset(
                safe,
                data=obs_img,
                compression="gzip",
                compression_opts=1,
            )
            next_grp.create_dataset(
                safe,
                data=next_img,
                compression="gzip",
                compression_opts=1,
            )


# ---------------------------------------------------------------------------
# absl flags (must be registered before app.run)
# ---------------------------------------------------------------------------
from absl import app as absl_app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name",
    None,
    "Experiment name registered in experiments.mappings.CONFIG_MAPPING.",
)
flags.DEFINE_string(
    "imle_root",
    _DEFAULT_IMLE_ROOT,
    "Path to imle_dex root (contains examples/).",
)
flags.DEFINE_string(
    "leap_api_root",
    _DEFAULT_LEAP_API,
    "Directory containing LEAP_Hand_API main.py (LeapNode).",
)
flags.DEFINE_boolean(
    "no_leap",
    False,
    "If True, do not connect to LEAP hardware (leap_* fields will be NaN in h5 / missing in pkl).",
)
flags.DEFINE_boolean(
    "skip_xr_probe",
    True,
    "Skip subprocess check for xrobotoolkit_sdk (127.0.0.1:60061). "
    "If the PICO bridge is down, the main process may abort — default is a safe probe.",
)
flags.DEFINE_integer("demos_needed", 20, "Stop after this many keyboard-success saves.")
flags.DEFINE_integer(
    "stack_obs_num",
    2,
    "Observation history length passed to config.get_environment (ChunkingWrapper).",
)
flags.DEFINE_boolean("fake_env", False, "If True, use fake_env=True for debugging.")
flags.DEFINE_boolean(
    "no_cameras",
    False,
    "Do not open RealSense: use black placeholder frames (no pyrealsense). "
    "Use when no camera is attached, another process holds the device (EBUSY), or you only need robot + LEAP. "
    "Consider --nosave_images to avoid storing black images in h5/pkl.",
)
flags.DEFINE_boolean("save_video", False, "Passed to get_environment.")
flags.DEFINE_boolean(
    "save_images",
    True,
    "If False, strip image keys from stored observations (env still may grab cameras). Use --nosave_images.",
)
flags.DEFINE_boolean(
    "reset_on_save_or_discard",
    True,
    "Call env.reset() after successful save (s) or discard (d).",
)
flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for pkl/h5. Default: ./demo_data_franka_leap under CWD.",
)


class _DummyVideoCapture:
    """Minimal stand-in for franka_env.camera.VideoCapture (BGR uint8, no threads)."""

    def __init__(self, width: int, height: int, name: str = "") -> None:
        self.name = name
        self._shape = (height, width, 3)

    def read(self) -> np.ndarray:
        return np.zeros(self._shape, dtype=np.uint8)

    def close(self) -> None:
        pass


def _install_dummy_realsense_cameras(exp_name: str) -> None:
    """Replace FrankaEnv.init_cameras so no RealSense / V4L device is opened."""
    import importlib
    from collections import OrderedDict

    from franka_env.envs.franka_env import FrankaEnv

    # Ensure experiment wrapper is imported so task-specific subclasses exist.
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

    print(
        "[CAMERAS] --no_cameras: RealSense disabled; observations use black image placeholders."
    )


class _StoppedFrankaListener:
    """Placeholder so FrankaEnv.close() can still call listener.stop()."""

    def stop(self) -> None:
        pass

    def join(self, timeout: float | None = None) -> None:
        pass


def _stop_franka_pynput_listener(env: Any) -> None:
    """
    FrankaEnv starts a pynput listener (ESC) via X11 RECORD.
    XrClient (Pico) must be initialized before FrankaEnv so RECORD is not active yet;
    after env is built we stop Franka's listener before starting this script's r/s/d/q listener.
    """
    base = getattr(env, "unwrapped", env)
    listener = getattr(base, "listener", None)
    if listener is None:
        return
    try:
        listener.stop()
        listener.join(timeout=2.0)
    except Exception as e:
        print(f"[KEYBOARD] Could not stop FrankaEnv ESC listener: {e}")
        return
    time.sleep(0.5)
    try:
        setattr(base, "listener", _StoppedFrankaListener())
    except Exception:
        pass
    print(
        "[KEYBOARD] FrankaEnv ESC listener stopped (avoid multiple pynput / X11 RECORD hooks)."
    )


def _xrobotoolkit_sdk_probe_succeeds() -> bool:
    """Probe xrobotoolkit init in a subprocess to avoid hard-crash."""
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import xrobotoolkit_sdk as xrt; xrt.init()"],
            timeout=25,
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return True
        tail = ((r.stderr or "") + (r.stdout or "")).strip().replace("\n", " ")
        if tail:
            print(f"[LEAP] XR SDK probe failed (exit {r.returncode}): {tail[:300]}")
        else:
            print(f"[LEAP] XR SDK probe failed (exit {r.returncode}).")
        return False
    except subprocess.TimeoutExpired:
        print("[LEAP] XR SDK probe timed out.")
        return False
    except Exception as e:
        print(f"[LEAP] XR SDK probe error: {e}")
        return False


def main(argv: list[str]) -> None:
    del argv  # absl leftover

    _setup_imle_paths(FLAGS.imle_root)

    from experiments.mappings import CONFIG_MAPPING

    if not FLAGS.exp_name:
        raise absl_app.UsageError("--exp_name is required.")
    if FLAGS.exp_name not in CONFIG_MAPPING:
        raise absl_app.UsageError(f"Unknown exp_name={FLAGS.exp_name!r}.")

    import h5py
    from pynput import keyboard
    from action_scaling import normalize_action_22d

    if FLAGS.no_cameras:
        _install_dummy_realsense_cameras(FLAGS.exp_name)

    config = CONFIG_MAPPING[FLAGS.exp_name]()
    strip_keys = _image_keys_to_strip(config)
    save_images = bool(FLAGS.save_images)

    pico_dir = os.path.dirname(os.path.abspath(__file__))
    if pico_dir not in sys.path:
        sys.path.insert(0, pico_dir)
    try:
        from pico_leap_dex import leap_joint_limits_from_config

        leap_lo, leap_hi = leap_joint_limits_from_config(config)
    except Exception:
        leap_lo, leap_hi = None, None

    controller: Any = None
    if not FLAGS.no_leap:
        xr_ok = FLAGS.skip_xr_probe or _xrobotoolkit_sdk_probe_succeeds()
        if not xr_ok:
            print(
                "[LEAP] Skipping Pico/LEAP: start the PICO PC bridge (SDK uses "
                "127.0.0.1:60061) or pass --no_leap to record Franka-only without this check."
            )
            controller = None
        else:
            # pico_leap_dex imports pybullet at module import time. Keep xrt.init()
            # before importing Pico controller modules.
            try:
                import xrobotoolkit_sdk as xrt

                xrt.init()
                print("[LEAP] xrobotoolkit_sdk: xrt.init() before PyBullet import.")

                import xrobotoolkit_teleop.common.xr_client as _xr_client_mod

                def _xr_client_init_no_second_sdk_init(self: Any) -> None:
                    print("XRoboToolkit SDK initialized.")

                _xr_client_mod.XrClient.__init__ = _xr_client_init_no_second_sdk_init  # type: ignore[assignment]

                from pico_leap_dex import PicoLeapDexController  # local module

                controller = PicoLeapDexController(
                    gui=False,
                    use_hardware=True,
                    urdf_path=None,
                    print_joints=False,
                    leap_joint_low=leap_lo,
                    leap_joint_high=leap_hi,
                )
                print("[LEAP] PicoLeapDexController initialized (hardware on).")
            except Exception as e:
                print(
                    f"[LEAP] Failed to init PicoLeapDexController: {e}. "
                    f"Continuing with --no_leap behavior (no leap control/record)."
                )
                controller = None

    try:
        env = config.get_environment(
            fake_env=FLAGS.fake_env,
            save_video=FLAGS.save_video,
            stack_obs_num=int(FLAGS.stack_obs_num),
        )
    except RuntimeError as e:
        err = str(e).lower()
        if not FLAGS.no_cameras and (
            "vidioc" in err or "device or resource busy" in err or "errno=16" in err
        ):
            raise RuntimeError(
                "RealSense failed to open (often EBUSY: another process uses the camera, or no camera). "
                "Re-run with --no_cameras to use black image placeholders (robot + LEAP unchanged), "
                "e.g. add:  --no_cameras --nosave_images"
            ) from e
        raise

    _stop_franka_pynput_listener(env)

    cmd_queue: queue.Queue[str] = queue.Queue()

    def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
        try:
            if hasattr(key, "char") and key.char:
                c = key.char.lower()
                if c in ("r", "s", "d", "q"):
                    cmd_queue.put(c)
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    out_dir = FLAGS.output_dir or os.path.join(os.getcwd(), "demo_data_franka_leap")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{FLAGS.exp_name}_{FLAGS.demos_needed}demos_{stamp}"
    run_dir = os.path.join(out_dir, base)
    os.makedirs(run_dir, exist_ok=True)
    pkl_path = os.path.join(run_dir, f"{base}.pkl")
    h5_path = os.path.join(run_dir, f"{base}.h5")

    transitions: list[dict[str, Any]] = []
    episode_id_counter = 0
    success_count = 0
    recording = False
    trajectory: list[dict[str, Any]] = []
    stop_requested = False
    step_number = 0

    obs, info = env.reset()
    print("=" * 72)
    print("Franka + LEAP recorder")
    print(f"  exp_name          : {FLAGS.exp_name}")
    print(f"  imle_root         : {FLAGS.imle_root}")
    print(f"  successes_needed  : {FLAGS.demos_needed}")
    print(f"  save_images       : {save_images}")
    print(f"  leap              : {'on' if controller else 'off'}")
    print(f"  output pkl        : {pkl_path}")
    print(f"  output h5         : {h5_path}")
    print("  keys: [r] start  [s] save success  [d] discard  [q] quit")
    print("=" * 72)

    def read_leap_hw_from_controller() -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if controller is None or getattr(controller, "leap_node", None) is None:
            return None, None, None
        try:
            pos, vel, cur = controller.leap_node.pos_vel_eff_srv()
            return (
                np.asarray(pos, dtype=np.float32),
                np.asarray(vel, dtype=np.float32),
                np.asarray(cur, dtype=np.float32),
            )
        except Exception:
            try:
                pos = np.asarray(controller.leap_node.read_pos(), dtype=np.float32)
                vel = np.asarray(controller.leap_node.read_vel(), dtype=np.float32)
                cur = np.asarray(controller.leap_node.read_cur(), dtype=np.float32)
                return pos, vel, cur
            except Exception as e:
                print(f"[LEAP] read failed: {e}")
                return None, None, None

    h5_f = h5py.File(h5_path, "w")
    h5_f.create_group("episodes")
    h5_f.attrs["exp_name"] = FLAGS.exp_name
    h5_f.attrs["created_at"] = datetime.datetime.now().isoformat()
    h5_f.attrs["stack_obs_num"] = int(FLAGS.stack_obs_num)
    h5_f.attrs["save_images"] = save_images
    h5_f.attrs["imle_root"] = FLAGS.imle_root

    def flush_episode_to_disk(ep_transitions: list[dict[str, Any]], ep_idx: int) -> None:
        # All non-state keys are treated as image / auxiliary tensors (SERLObsWrapper layout).
        image_keys_present = (
            [k for k in ep_transitions[0]["observations"].keys() if k != "state"]
            if save_images
            else []
        )
        _write_episode_h5(h5_f, ep_idx, ep_transitions, save_images, image_keys_present)

    def flush_episode_pkl(ep_transitions: list[dict[str, Any]], ep_idx: int) -> str:
        ep_meta = {
            "exp_name": FLAGS.exp_name,
            "imle_root": FLAGS.imle_root,
            "episode_id": ep_idx,
            "stack_obs_num": int(FLAGS.stack_obs_num),
            "save_images": save_images,
            "created_at": datetime.datetime.now().isoformat(),
        }
        ep_payload = {"transitions": copy.deepcopy(ep_transitions), "meta": ep_meta}
        ep_path = os.path.join(run_dir, f"{base}_ep{ep_idx:04d}.pkl")
        with open(ep_path, "wb") as f:
            pkl.dump(ep_payload, f)
        return ep_path

    try:
        while success_count < FLAGS.demos_needed and not stop_requested:
            # Drain keyboard queue
            while True:
                try:
                    cmd = cmd_queue.get_nowait()
                except queue.Empty:
                    break
                if cmd == "r":
                    trajectory = []
                    recording = True
                    print("[KEY] Recording started (trajectory cleared).")
                elif cmd == "s":
                    if not trajectory:
                        print("[KEY] Nothing to save; trajectory empty.")
                    else:
                        ep_id = episode_id_counter
                        episode_id_counter += 1
                        for t in trajectory:
                            t["episode_id"] = ep_id
                            t["keyboard_success"] = True
                        transitions.extend(trajectory)
                        flush_episode_to_disk(trajectory, ep_id)
                        ep_pkl_path = flush_episode_pkl(trajectory, ep_id)
                        h5_f.flush()
                        success_count += 1
                        print(
                            f"[SAVE] Success {success_count}/{FLAGS.demos_needed} "
                            f"(episode_id={ep_id}, T={len(trajectory)})."
                        )
                        print(f"[SAVE] Episode pkl: {ep_pkl_path}")
                        trajectory = []
                        recording = False
                        if FLAGS.reset_on_save_or_discard:
                            obs, info = env.reset()
                            step_number = 0
                elif cmd == "d":
                    print(f"[KEY] Discarded trajectory (had {len(trajectory)} transitions).")
                    trajectory = []
                    recording = False
                    if FLAGS.reset_on_save_or_discard:
                        obs, info = env.reset()
                        step_number = 0
                elif cmd == "q":
                    print("[KEY] Quit requested.")
                    stop_requested = True
                    break

            if stop_requested:
                break

            leap_cmd = None
            if controller is not None:
                # Drive leaphand with pico every loop, keeping it time-aligned
                # with the Franka transition recording.
                leap_result = controller.step(verbose=False)
                if (
                    leap_result is not None
                    and "leap_q" in leap_result
                    and getattr(controller, "_hw_ready", False)
                ):
                    leap_cmd = leap_result.get("leap_q", None)

            lp, lv, lc = read_leap_hw_from_controller()

            # Build 22D action: action[0:6] = arm, action[6:22] = LeapHand 16 joints.
            # Send zeros first so env advances one step; intervention fills arm dims.
            actions_22d = np.zeros(env.action_space.sample().shape, dtype=np.float32)
            next_obs, rew, done, truncated, step_info = env.step(actions_22d)
            step_number += 1
            if "intervene_action" in step_info:
                intervene = np.asarray(step_info["intervene_action"], dtype=np.float32)
                actions_22d[:6] = intervene[:6]
            if leap_cmd is not None:
                actions_22d[6:22] = np.asarray(leap_cmd, dtype=np.float32).reshape(16)

            obs_s = _prepare_obs_for_save(obs, save_images, strip_keys)
            next_s = _prepare_obs_for_save(next_obs, save_images, strip_keys)

            transition: dict[str, Any] = {
                "observations": obs_s,
                "actions": normalize_action_22d(actions_22d),
                "next_observations": next_s,
                "rewards": rew,
                "masks": float(1.0 - done),
                "dones": done,
                "infos": _safe_deepcopy_info(dict(step_info)),
            }
            if leap_cmd is not None:
                transition["leap_cmd"] = np.asarray(leap_cmd, dtype=np.float32)
            if lp is not None:
                transition["leap_hw_pos"] = lp
                transition["leap_hw_vel"] = lv
                transition["leap_hw_cur"] = lc

            if recording:
                print(f"[STEP] step_number={step_number}")
                trajectory.append(transition)

            obs = next_obs

            if done or truncated:
                print(
                    f"[ENV] Episode ended: done={done}, truncated={truncated}, "
                    f"info.succeed={step_info.get('succeed')}"
                )
                if recording:
                    print(
                        "[ENV] Hint: you were recording; use [d] to discard or [s] to save "
                        "before/after reset depending on intent."
                    )
                obs, info = env.reset()
                step_number = 0
                recording = False

    finally:
        try:
            h5_f.close()
        except Exception:
            pass
        try:
            listener.stop()
        except Exception:
            pass

    meta = {
        "exp_name": FLAGS.exp_name,
        "imle_root": FLAGS.imle_root,
        "demos_needed": FLAGS.demos_needed,
        "demos_saved": success_count,
        "stack_obs_num": int(FLAGS.stack_obs_num),
        "save_images": save_images,
        "strip_image_keys": strip_keys,
        "created_at": datetime.datetime.now().isoformat(),
    }
    payload = {"transitions": transitions, "meta": meta}
    with open(pkl_path, "wb") as f:
        # Keep manual recorder pkl style: transitions + meta payload.
        pkl.dump(payload, f)

    print("=" * 72)
    print(f"Wrote pkl: {pkl_path}  ({len(transitions)} transitions)")
    print(f"Wrote h5 : {h5_path}  ({success_count} episodes)")
    print("=" * 72)


if __name__ == "__main__":
    absl_app.run(main)
