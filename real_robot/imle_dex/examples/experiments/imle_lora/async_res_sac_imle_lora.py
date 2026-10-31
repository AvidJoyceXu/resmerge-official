#!/usr/bin/env python3
# Modified by the ResMerge contributors, 2026-10-31; see real_robot/THIRD_PARTY.md.
"""
Asynchronous residual SAC with frozen IMLE and a low-rank actor. Rollout executes base +
alpha*tanh(u) according to the residual schedule; replay stores executed actions and
current/next base actions. The learner updates actor, full-rank critic and temperature.
Evaluation can use the residual actor or IMLE base alone without connecting a learner. Human
labels: space=success, f=failure; r starts after reset and t resets again. Actor statistics use
a fixed 20-episode denominator. See README.md for the supported imle_preproj workflow and
launch commands.
"""

from __future__ import annotations

import datetime
import copy
import importlib.util
import json
import os
import pickle as pkl
import random
import sys
import threading
import time
from collections import deque
from typing import Any, Callable, Optional, Tuple


def _imle_root_from_this_file() -> str:
    """Infer imle_dex from this file, three directories above imle_lora."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _prepend_imle_sys_paths(imle_root: str) -> None:
    """Load the repository-local packages."""
    root = os.path.abspath(imle_root)
    
    for sub in ("octo", "serl_launcher", "serl_robot_infra", "examples"):
        p = os.path.join(root, sub)
        if not os.path.isdir(p):
            continue
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def _bootstrap_imle_sys_path() -> None:
    _prepend_imle_sys_paths(_imle_root_from_this_file())


_bootstrap_imle_sys_path()


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags

from flax.training import checkpoints
from hydra import compose, initialize_config_dir
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerServer
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.data.data_store import (
    MemoryEfficientReplayBufferDataStore,
    ReplayBufferDataStore,
    populate_data_store,
)
from serl_launcher.utils.launcher import (
    dict_observation_space_drop_keys,
    make_drq_residual_imle_pixel_agent,
    make_sac_residual_imle_pixel_agent,
    make_trainer_config,
    make_wandb_logger,
    strip_replay_batch_pixel_keys,
)


_lora_agent_path = os.path.join(os.path.dirname(__file__), "lora_agent.py")
_lora_spec = importlib.util.spec_from_file_location("lora_agent", _lora_agent_path)
lora_agent = importlib.util.module_from_spec(_lora_spec)
sys.modules[_lora_spec.name] = lora_agent
_lora_spec.loader.exec_module(lora_agent)
make_sac_residual_imle_preproj_lora_agent = (
    lora_agent.make_sac_residual_imle_preproj_lora_agent
)

from serl_launcher.utils.timer_utils import Timer

from franka_env.envs.wrappers import AugmentDictObsWrapper, KeyboardSuccessWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "pick_place", "Configuration name in experiments.mappings.")
flags.DEFINE_string("imle_root", "", "imle_dex root; inferred from this file by default.")
flags.DEFINE_string("ckpt_path", "", "IMLE checkpoint; alternatively set Hydra optimizer.start_ckpt_path=... .")
flags.DEFINE_string("precise_dex_root", "", "precise_dex root; defaults to the evaluation module location.")
flags.DEFINE_string("device", "", "PyTorch device; defaults to Hydra cfg.training.device or cuda:0.")
flags.DEFINE_integer("stack_obs_num", 2, "Observation stack length; must match the Hydra RGB stream count.")
flags.DEFINE_bool("fake_env", True, "Use the fake/placeholder environment.")
flags.DEFINE_bool("no_cameras", False, "Use black camera placeholders, matching the evaluation camera override.")
flags.DEFINE_bool("no_leap", False, "Do not connect LEAP hardware.")
flags.DEFINE_string(
    "leap_api_root",
    "",
    "LEAP_Hand_API/python path; defaults to the repository copy.",
)
flags.DEFINE_bool("use_spacemouse", True, "Enable SpaceMouse intervention.")
flags.DEFINE_bool(
    "use_keyboard_success",
    False,
    "Real actor only: enable KeyboardSuccessWrapper; space ends an episode successfully.",
)
flags.DEFINE_bool(
    "rl_wait_r_after_reset",
    False,
    "Actor only: after reset, wait for r to begin; t resets again while waiting.",
)
flags.DEFINE_bool(
    "rl_actor_human",
    False,
    "Actor preset: disable SpaceMouse, enable --use_keyboard_success and "
    "--rl_wait_r_after_reset for keyboard-controlled episodes.",
)
flags.DEFINE_bool(
    "hil",
    False,
    "Actor human intervention mode: enable SpaceMouse, keyboard labels and r-to-start after reset."
    "During SpaceMouse intervention, the executed human-mixed action (info['intervene_action']) "
    "replaces the policy action in replay. Mutually exclusive with --rl_actor_human and --eval_only.",
)
flags.DEFINE_integer("latency_step", 0, "Latency steps for IMLE bidirection_predict_action.")
flags.DEFINE_integer("chunk_exec_steps", 8, "Environment steps per IMLE prediction.")
flags.DEFINE_float("control_hz", 10.0, "Actor control frequency in Hz, matching evaluation timing.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_float(
    "target_entropy",
    np.nan,
    "SAC target entropy; NaN uses the algorithm default (-action_dim/2).",
)
flags.DEFINE_float(
    "alpha_max",
    1.0,
    "Maximum SAC temperature alpha; <=0 disables the cap.",
)
flags.DEFINE_float(
    "residual_alpha",
    np.nan,
    "Residual scale alpha in a_res = alpha*tanh(u); NaN uses "
    "TrainConfig.residual_alpha. Actor and learner must use the same value.",
)
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 8, "critic batch = batch_size * ratio。")
flags.DEFINE_integer("max_steps", 1_000_000, "Maximum actor environment interactions.")
flags.DEFINE_integer("replay_buffer_capacity", 1_000_000, "Replay buffer capacity.")
flags.DEFINE_multi_string(
    "demo_paths",
    [],
    "Offline demonstration pickle paths (repeatable); enables 50/50 demo/online sampling in the learner.",
)
flags.DEFINE_bool(
    "replay_omit_pixels",
    False,
    "For imle_preproj only: omit RGB from replay and use ReplayBuffer to reduce memory; "
    "enable consistently on actor and learner. Not supported in pixel mode.",
)
flags.DEFINE_integer(
    "random_steps",
    300,
    "Initial global environment steps using Gaussian u instead of the policy, forcing epsilon=1 "
    "and executing a_base+alpha*tanh(u) for residual exploration.",
)
flags.DEFINE_integer(
    "res_start_steps",
    0,
    "Initial global steps without residual actions (epsilon=0).",
)
flags.DEFINE_integer(
    "res_ramp_steps",
    1000,
    "Steps over which residual probability increases from 0 to 1, starting at res_start_steps.",
)
flags.DEFINE_integer("training_starts", 300, "Minimum replay size before learner updates begin.")
flags.DEFINE_integer(
    "critic_warmup_steps",
    2000,
    "Residual learner warmup counted in gradient updates, not environment steps: "
    "update only the critic for the first N steps; 0 disables warmup.",
)
flags.DEFINE_integer("steps_per_update", 30, "Environment steps between actor parameter synchronization requests.")
flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string(
    "checkpoint_path",
    None,
    "If checkpoint_period>0, this is the parent directory for a new run named after the W&B run; "
    "to resume, point to a previous run containing checkpoint_* or set --resume; "
    "existing checkpoint_* files also trigger resume automatically.",
)
flags.DEFINE_bool(
    "resume",
    False,
    "Explicit resume: --checkpoint_path must point to a previous run directory (e.g. "
    ".../serl_res_sac_lora_checkpoints/pick_place_2026.05.27-20.34.36）。",
)
flags.DEFINE_integer(
    "checkpoint_step",
    0,
    "Agent checkpoint update step to restore; 0 selects the latest checkpoint in the run.",
)
flags.DEFINE_integer(
    "buffer_save_period",
    0,
    "Online replay save interval in gradient steps; 0 follows checkpoint_period if positive.",
)
flags.DEFINE_bool("learner", False, "Run the learner process.")
flags.DEFINE_bool("actor", False, "Run the actor process.")
flags.DEFINE_string("ip", "localhost", "Learner IP。")
flags.DEFINE_bool("debug", False, "Disable W&B logging.")
flags.DEFINE_string(
    "wandb_project",
    "imle_resrl_lora",
    "W&B project name for the LoRA experiment.",
)
flags.DEFINE_string(
    "wandb_entity",
    "",
    "W&B team/user entity; defaults to WANDB_ENTITY, then the authenticated account.",
)
flags.DEFINE_enum(
    "wandb_mode",
    "online",
    ["online", "offline"],
    "W&B mode: online uploads; offline stores local logs for later wandb sync."
    "If using HTTP(S)_PROXY, exclude robot/local services through NO_PROXY to avoid routing local requests through the proxy.",
)
flags.DEFINE_multi_string(
    "hydra_override",
    [],
    "Hydra override, e.g. n_action_steps=40; repeat --hydra_override for multiple values.",
)
flags.DEFINE_enum(
    "sac_obs_mode",
    "pixel",
    ["pixel", "imle_preproj"],
    "pixel uses ResNet/proprio; imle_preproj uses frozen IMLE pre-projection features with optional base-action concatenation.",
)
flags.DEFINE_bool(
    "sac_imle_feat_concat_action_base",
    False,
    "Concatenate the current IMLE base action to pre-projection features; must match checkpoint dimensions.",
)
flags.DEFINE_enum(
    "sac_pixel_backend",
    "sac",
    ["sac", "drq"],
    "Pixel mode backend: sac uses SACResidualIMLEAgent with augmentation_function; "
    "drq uses DrQResidualIMLEAgent with internal cropping and the same residual losses.",
)
flags.DEFINE_integer(
    "lora_rank",
    16,
    "Actor low-rank dimension for imle_preproj mode. "
    "Typical values: 4, 8, 16, 32. Smaller ranks reduce actor parameters; the critic remains full-rank.",
)


flags.DEFINE_string(
    "eval_merged_pt",
    "",
    "Load this .pt file into actor parameters after agent creation using load_merged_for_actor; "
    "supports single-task 'params' and merged 'merged_params' formats. "
    "Requires --eval_only.",
)
flags.DEFINE_list(
    "eval_source_pts",
    [],
    "If --eval_merged_pt has no _jax_extras LayerNorm parameters, "
    "provide the source task_*.pt files for weighted LayerNorm recovery. "
    "Leave empty for a single-task export containing _jax_extras.",
)
flags.DEFINE_integer(
    "eval_episodes",
    20,
    "Episodes to evaluate before exiting; 0 means unlimited.",
)
flags.DEFINE_bool(
    "eval_only",
    False,
    "Actor evaluation only: no learner connection, replay upload or parameter synchronization. "
    "Usually combined with --eval_merged_pt or --eval_imle_base_only.",
)
flags.DEFINE_bool(
    "eval_imle_base_only",
    False,
    "With --eval_only, execute only frozen IMLE base actions without constructing a JAX LoRA actor. "
    "Mutually exclusive with --eval_merged_pt. Set EVAL_BASE_ONLY=1 in the actor launcher.",
)


def print_green(x: str) -> None:
    print("\033[92m {}\033[00m".format(x))


def print_yellow(x: str) -> None:
    print("\033[93m {}\033[00m".format(x))


_TRAINING_META_JSON = "training_meta.json"
_REPLAY_BUFFER_LATEST_PKL = "replay_buffer_latest.pkl"
_REPLAY_BUFFER_STEP_PKL = "replay_buffer_step{step}.pkl"
_REPLAY_BUFFER_SNAPSHOT_KEY = "snapshot_v1"
_buffer_save_lock = threading.Lock()
_buffer_save_thread: Optional[threading.Thread] = None


def _is_run_checkpoint_dir(path: str) -> bool:
    """Check whether the directory contains checkpoint_* files or training_meta.json."""
    if not path or not os.path.isdir(path):
        return False
    try:
        for name in os.listdir(path):
            if name.startswith("checkpoint_"):
                return True
    except OSError:
        return False
    return os.path.isfile(os.path.join(path, _TRAINING_META_JSON))


def _resolve_run_checkpoint_dir(
    ckpt_parent: str,
    run_name: str,
    *,
    resume_explicit: bool,
) -> Tuple[str, bool]:
    """Return the absolute run directory and whether this is a resumed run."""
    ckpt_parent = os.path.abspath(os.path.expanduser(ckpt_parent))
    if resume_explicit or _is_run_checkpoint_dir(ckpt_parent):
        os.makedirs(ckpt_parent, exist_ok=True)
        resumed = _is_run_checkpoint_dir(ckpt_parent)
        if resume_explicit and not resumed:
            print_yellow(
                f"[resume] No checkpoint_* in {ckpt_parent}; starting a new run there."
            )
        return ckpt_parent, resumed
    safe = run_name.replace(os.sep, "_").replace("\x00", "")
    run_ckpt = os.path.join(ckpt_parent, safe)
    os.makedirs(run_ckpt, exist_ok=True)
    return run_ckpt, False


def _parse_checkpoint_step(ckpt_path: str) -> int:
    name = os.path.basename(os.path.normpath(ckpt_path))
    if name.startswith("checkpoint_"):
        return int(name.split("_", 1)[1])
    raise ValueError(f"Cannot parse update step from checkpoint path: {ckpt_path!r}")


def _restore_agent_from_run_dir(agent, run_dir: str, step: int = 0):
    """Restore agent and optimizer state; return (agent, next_update_step)."""
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    latest = checkpoints.latest_checkpoint(run_dir)
    if latest is None:
        print_yellow(f"[resume] No checkpoint in {run_dir}; keeping initialized agent parameters.")
        return agent, 0
    if step > 0:
        ckpt = checkpoints.restore_checkpoint(run_dir, agent.state, step=step)
        ckpt_step = int(step)
    else:
        ckpt = checkpoints.restore_checkpoint(run_dir, agent.state)
        ckpt_step = _parse_checkpoint_step(latest)
    agent = agent.replace(state=ckpt)
    next_step = ckpt_step + 1
    print_green(
        f"[resume] Loaded agent at step={ckpt_step}; continuing from update_step={next_step}"
    )
    return agent, next_step


def _write_training_meta(run_dir: str, update_steps: int, agent) -> None:
    meta = {
        "update_steps": int(update_steps),
        "exp_name": str(FLAGS.exp_name),
        "lora_rank": int(FLAGS.lora_rank),
        "sac_obs_mode": str(FLAGS.sac_obs_mode),
        "replay_buffer_capacity": int(FLAGS.replay_buffer_capacity),
        "seed": int(FLAGS.seed),
        "residual_alpha": float(agent.config["residual_alpha"]),
        "base_checkpoint": str(FLAGS.ckpt_path),
    }
    path = os.path.join(run_dir, _TRAINING_META_JSON)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _buffer_chronological_indices(buf) -> np.ndarray:
    n = int(buf._size)
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n < int(buf._capacity):
        return np.arange(n, dtype=np.int64)
    start = int(buf._insert_index)
    cap = int(buf._capacity)
    return np.array([(start + i) % cap for i in range(n)], dtype=np.int64)


def _clear_replay_buffer(buf) -> None:
    buf._size = 0
    buf._insert_index = 0


def _subselect_dataset_dict(d: Any, indx: np.ndarray) -> Any:
    """
    Snapshot valid replay entries in chronological order as a NumPy tree, avoiding
    per-transition dictionaries.
    """
    if isinstance(d, np.ndarray):
        return np.asarray(d[indx]).copy()
    if isinstance(d, dict):
        return {k: _subselect_dataset_dict(v, indx) for k, v in d.items()}
    raise TypeError(f"unsupported dataset leaf type: {type(d)}")


def _write_dataset_prefix(dst: Any, src: Any, n: int) -> None:
    """Restore the first n snapshot entries into ring-buffer slots [0:n)."""
    if isinstance(dst, np.ndarray):
        dst[:n] = src[:n]
    elif isinstance(dst, dict):
        for k in dst:
            _write_dataset_prefix(dst[k], src[k], n)
    else:
        raise TypeError(f"unsupported dataset leaf type: {type(dst)}")


def _snapshot_replay_buffer(buf) -> Optional[dict]:
    lock = getattr(buf, "_lock", None)
    if lock is not None:
        lock.acquire()
    try:
        n = int(len(buf))
        if n <= 0:
            return None
        indx = _buffer_chronological_indices(buf)
        return {
            "format": _REPLAY_BUFFER_SNAPSHOT_KEY,
            "size": n,
            "capacity": int(buf._capacity),
            "insert_index": int(buf._insert_index),
            "dataset": _subselect_dataset_dict(buf.dataset_dict, indx),
        }
    finally:
        if lock is not None:
            lock.release()


def _restore_snapshot_into_buffer(buf, snap: dict) -> int:
    n = int(snap["size"])
    _write_dataset_prefix(buf.dataset_dict, snap["dataset"], n)
    cap = int(buf._capacity)
    buf._size = n
    buf._insert_index = (n % cap) if n >= cap else n
    return n


def _restore_legacy_transition_list(buf, transitions: list) -> int:
    _clear_replay_buffer(buf)
    for tr in transitions:
        buf.insert(tr)
    return len(buf)


def _buffer_save_worker(
    snap: dict,
    latest_path: str,
    stepped_path: str,
    t0: float,
) -> None:
    global _buffer_save_thread
    try:
        tmp = latest_path + ".tmp"
        with open(tmp, "wb") as f:
            pkl.dump(snap, f, protocol=pkl.HIGHEST_PROTOCOL)
        os.replace(tmp, latest_path)
        if stepped_path != latest_path:
            import shutil

            shutil.copy2(latest_path, stepped_path)
        dt = time.time() - t0
        print_green(f"[buffer] Saved {snap['size']} entries in {dt:.1f}s to {latest_path}")
    except Exception as e:
        print_yellow(f"[buffer] Save failed: {e}")
    finally:
        with _buffer_save_lock:
            _buffer_save_thread = None


def _save_online_replay_buffer(buf, run_dir: str, update_steps: int) -> None:
    """
    Write a compact replay snapshot in a background thread to avoid blocking learner updates.
    """
    global _buffer_save_thread
    with _buffer_save_lock:
        if _buffer_save_thread is not None and _buffer_save_thread.is_alive():
            print_yellow(
                f"[buffer] step={update_steps}: previous save still running; skipping to avoid blocking training"
            )
            return

    print(f"[buffer] step={update_steps}: snapshotting online buffer (len={len(buf)})...", flush=True)
    t0 = time.time()
    snap = _snapshot_replay_buffer(buf)
    if snap is None:
        print_yellow(f"[buffer] step={update_steps}: buffer empty; skipping.")
        return

    run_dir = os.path.abspath(run_dir)
    latest = os.path.join(run_dir, _REPLAY_BUFFER_LATEST_PKL)
    stepped = os.path.join(
        run_dir, _REPLAY_BUFFER_STEP_PKL.format(step=int(update_steps))
    )
    dt_snap = time.time() - t0
    print(
        f"[buffer] Snapshot of {snap['size']} entries ready ({dt_snap:.1f}s); writing in background...",
        flush=True,
    )

    th = threading.Thread(
        target=_buffer_save_worker,
        args=(snap, latest, stepped, t0),
        daemon=True,
    )
    with _buffer_save_lock:
        _buffer_save_thread = th
    th.start()


def _restore_online_replay_buffer(buf, run_dir: str) -> int:
    path = os.path.join(os.path.abspath(run_dir), _REPLAY_BUFFER_LATEST_PKL)
    if not os.path.isfile(path):
        print_yellow(f"[resume] {path} not found; starting with empty online replay.")
        return 0
    print(f"[resume] Loading replay buffer: {path}", flush=True)
    t0 = time.time()
    with open(path, "rb") as f:
        payload = pkl.load(f)

    lock = getattr(buf, "_lock", None)
    if lock is not None:
        lock.acquire()
    try:
        if isinstance(payload, dict) and payload.get("format") == _REPLAY_BUFFER_SNAPSHOT_KEY:
            n = _restore_snapshot_into_buffer(buf, payload)
        elif isinstance(payload, list):
            print_yellow(
                "[resume] Loading transition-list replay entries..."
            )
            n = _restore_legacy_transition_list(buf, payload)
        else:
            raise ValueError(f"Unrecognized replay buffer format: {type(payload)}")
    finally:
        if lock is not None:
            lock.release()
    print_green(f"[resume] Restored {n} online transitions in {time.time() - t0:.1f}s.")
    return n


def _effective_buffer_save_period() -> int:
    if int(FLAGS.buffer_save_period) > 0:
        return int(FLAGS.buffer_save_period)
    if int(FLAGS.checkpoint_period) > 0:
        return int(FLAGS.checkpoint_period)
    return 0


_EPISODE_SLIDE_LEN = 20


def _replay_omit_pixel_keys(env: gym.Env) -> tuple[str, ...]:
    """
    Remove all top-level camera keys except state/imle_preproj; using only configured
    image_keys can miss aliases such as demo and waste replay memory.
    """
    _keep = frozenset({"state", "imle_preproj"})
    return tuple(k for k in env.observation_space.spaces if k not in _keep)


def _clear_keyboard_success_pending(env: Any) -> None:
    """Clear pending keyboard success/failure events along the wrapper chain."""
    cur: Any = env
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, KeyboardSuccessWrapper):
            cur.clear_pending_space()
        cur = getattr(cur, "env", None)


def _wait_for_r_continue(
    env: Any,
    fle: Any,
    leap_hist: deque,
    stack_obs_num: int,
    read_leap_hw: Callable[[], Optional[np.ndarray]],
    obs: dict[str, Any],
) -> dict[str, Any]:
    """
    Wait for r to start rollout; t resets again and clears pending success labels to avoid
    carrying accidental input into the next episode.
    """
    from pynput import keyboard

    print_yellow(
        "Environment reset: press r to start rollout or t to reset again."
    )
    current_obs = obs

    while True:
        choice: list[str] = [""]
        done = threading.Event()

        def on_press(key):
            try:
                if key == keyboard.Key.esc:
                    print_yellow("Esc received; still waiting for r / t...")
                    return
                c = getattr(key, "char", None)
                if c is None:
                    return
                cl = c.lower()
                if cl == "r":
                    choice[0] = "r"
                    done.set()
                    return False
                if cl == "t":
                    choice[0] = "t"
                    done.set()
                    return False
            except Exception:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        try:
            done.wait()
        finally:
            try:
                listener.stop()
            except Exception:
                pass
            try:
                listener.join(timeout=2.0)
            except Exception:
                pass

        if choice[0] == "t":
            current_obs, _ = env.reset()
            fle._fill_leap_hist_after_reset(leap_hist, stack_obs_num, read_leap_hw)
            _clear_keyboard_success_pending(env)
            print_yellow("Reset complete; press r to start or t to reset again.")
            continue

        if choice[0] == "r":
            _clear_keyboard_success_pending(env)
            print_green("r received; continuing rollout.")
            return current_obs


def _jax_device_for_torch(torch_device: torch.device):
    """Select the JAX device matching PyTorch for same-accelerator DLPack exchange."""
    if torch_device.type == "cuda":
        idx = int(torch_device.index or 0)
        gpus = [d for d in jax.local_devices() if getattr(d, "platform", "") == "gpu"]
        if gpus and idx < len(gpus):
            return gpus[idx]
        if gpus:
            return gpus[0]
    return jax.local_devices("cpu")[0]


def _jax_array_to_torch(x: Any, *, torch_device: torch.device) -> torch.Tensor:
    """Convert a JAX array to a torch tensor with DLPack, without NumPy."""
    import jax.dlpack as jax_dlpack

    x = jnp.asarray(x, dtype=jnp.float32).reshape(-1)
    x = jax.device_put(x, _jax_device_for_torch(torch_device))
    capsule = jax_dlpack.to_dlpack(x)
    return torch.utils.dlpack.from_dlpack(capsule).reshape(-1).to(torch_device)


def _torch_tensor_to_jax(t: torch.Tensor) -> Any:
    """Convert a torch tensor to a JAX array with DLPack, without NumPy."""
    import jax.dlpack as jax_dlpack

    t = t.detach().contiguous()
    if t.dtype != torch.float32:
        t = t.float()
    capsule = torch.utils.dlpack.to_dlpack(t)
    return jax_dlpack.from_dlpack(capsule)


def _concat_batches(batch_a: Any, batch_b: Any) -> Any:
    """Concatenate replay batches along their leading dimension; tree structures must match."""
    return jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate([x, y], axis=0),
        batch_a,
        batch_b,
    )


_RES_IMLE_CTX: Optional[dict[str, Any]] = None


def _leaphand_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _load_eval_module():
    path = os.path.join(_leaphand_repo_root(), "teleop", "franka_leap_imle_eval.py")
    spec = importlib.util.spec_from_file_location("franka_leap_imle_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup_imle_sys_path(imle_root: str) -> None:
    """Apply the same package bootstrap using the optional FLAGS.imle_root override."""
    _prepend_imle_sys_paths(imle_root)


def _load_psi_policy(
    hydra_overrides: list[str],
    ckpt_path: str,
    precise_dex_root: str,
    device_override: str,
):
    """Compose Hydra configuration from precise_dex, matching the evaluation entry point."""
    precise_root = os.path.abspath(precise_dex_root)
    if not os.path.isdir(precise_root):
        raise FileNotFoundError(f"precise_dex_root: {precise_root}")
    cwd_old = os.getcwd()
    sys.path.insert(0, precise_root)
    try:
        os.chdir(precise_root)
        from utils import setup_path_variables

        setup_path_variables()
    finally:
        os.chdir(cwd_old)

    hydra_config_dir = os.path.join(precise_root, "psi_policy", "config")
    if not os.path.isdir(hydra_config_dir):
        raise FileNotFoundError(hydra_config_dir)

    with initialize_config_dir(version_base=None, config_dir=hydra_config_dir):
        cfg = compose(config_name="psi_policy_workspace", overrides=list(hydra_overrides))
    OmegaConf.resolve(cfg)

    if ckpt_path:
        cfg.optimizer.start_ckpt_path = ckpt_path
    if not cfg.optimizer.start_ckpt_path or str(cfg.optimizer.start_ckpt_path) in ("null", ""):
        raise ValueError("Provide an IMLE checkpoint with --ckpt_path or optimizer.start_ckpt_path=...")

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
    if len(image_shape) != 3:
        raise ValueError(f"Expected task.image_shape=[C,H,W], got {image_shape}")
    _, h224, w224 = image_shape
    image_hw = (int(h224), int(w224))
    return cfg, policy, dev, image_hw


def _probe_flat_imle_feat_dim(
    policy: torch.nn.Module,
    device: torch.device,
    fle: Any,
    obs: dict[str, Any],
    leap_hist: deque,
    *,
    rgb_keys: list[str],
    image_hw: tuple[int, int],
    n_obs_steps: int,
    stack_obs_num: int,
    concat_action_base: bool,
    action_dim: int,
) -> int:
    with torch.no_grad():
        obs_t = fle.build_policy_obs(
            obs,
            rgb_keys=rgb_keys,
            image_hw=image_hw,
            n_obs_steps=n_obs_steps,
            stack_obs_num=int(stack_obs_num),
            leap_hist=leap_hist,
        )
        obs_t = {k: v.to(device) for k, v in obs_t.items()}
        nobs = policy.normalizer.normalize(obs_t)
        ab = None
        if concat_action_base:
            ab = torch.zeros(1, 1, int(action_dim), device=device, dtype=torch.float32)
        ft = policy.obs_encoder.forward_pre_projection(
            nobs, training=False, action_base=ab
        )
        return int(ft.numel())


def _fill_imle_preproj_obs(
    obs: dict[str, Any],
    *,
    policy: torch.nn.Module,
    device: torch.device,
    fle: Any,
    leap_hist: deque,
    rgb_keys: list[str],
    image_hw: tuple[int, int],
    n_obs_steps: int,
    stack_obs_num: int,
    action_base_torch: Optional[torch.Tensor],
    read_leap_hw: Optional[Any] = None,
) -> None:
    with torch.no_grad():
        obs_t = fle.build_policy_obs(
            obs,
            rgb_keys=rgb_keys,
            image_hw=image_hw,
            n_obs_steps=n_obs_steps,
            stack_obs_num=int(stack_obs_num),
            leap_hist=leap_hist,
        )
        obs_t = {k: v.to(device) for k, v in obs_t.items()}
        nobs = policy.normalizer.normalize(obs_t)
        ab = None
        if action_base_torch is not None:
            ab = action_base_torch.to(device=device, dtype=torch.float32)
            if ab.ndim == 1:
                ab = ab.reshape(1, 1, -1)
            elif ab.ndim == 2:
                ab = ab[:, None, :]
        ft = policy.obs_encoder.forward_pre_projection(
            nobs, training=False, action_base=ab
        )
    obs["imle_preproj"] = (
        ft.detach().float().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    )


def _next_imle_action_base_np(
    next_obs: dict[str, Any],
    *,
    pred_t: torch.Tensor,
    j: int,
    prev_traj: torch.Tensor,
    policy: torch.nn.Module,
    device: torch.device,
    fle: Any,
    leap_hist: deque,
    rgb_keys: list[str],
    image_hw: tuple[int, int],
    n_obs_steps: int,
    stack_obs_num: int,
    latency_step: int,
    read_leap_hw: Callable[[], Optional[np.ndarray]],
    current_chunk_len: int,
) -> tuple[np.ndarray, Optional[torch.Tensor]]:
    """
    Return (next_action_base, cached_pred_t), reusing look-ahead predictions across chunk
    boundaries.
    """
    if (j + 1 < int(current_chunk_len)) and (j + 1 < pred_t.shape[1]):
        return (
            pred_t[0, j + 1]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
            .reshape(-1),
            None,
        )
    lh: deque[np.ndarray] = deque(list(leap_hist), maxlen=int(stack_obs_num))
    lp = read_leap_hw()
    if lp is None:
        lp = lh[-1] if lh else np.zeros(16, dtype=np.float32)
    lp = np.asarray(lp, dtype=np.float32).reshape(16)
    lh.append(lp)
    obs_t = fle.build_policy_obs(
        next_obs,
        rgb_keys=rgb_keys,
        image_hw=image_hw,
        n_obs_steps=n_obs_steps,
        stack_obs_num=int(stack_obs_num),
        leap_hist=lh,
    )
    obs_t = {k: v.to(device, non_blocking=True) for k, v in obs_t.items()}
    with torch.no_grad():
        out = policy.bidirection_predict_action(
            obs_t, None, prev_traj, latency_step=int(latency_step)
        )
    cached_pred_t = out["action"].detach()
    return (
        cached_pred_t[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
        .reshape(-1),
        cached_pred_t,
    )


def actor(agent: Optional[SACAgent], data_store, env, sampling_rng):
    global _RES_IMLE_CTX
    if _RES_IMLE_CTX is None:
        raise RuntimeError("Internal error: _RES_IMLE_CTX is not initialized")

    ctx = _RES_IMLE_CTX
    fle = ctx["eval_mod"]
    policy: torch.nn.Module = ctx["policy"]
    device: torch.device = ctx["device"]
    imle_cfg = ctx["imle_cfg"]
    read_leap_hw: Callable[[], Optional[np.ndarray]] = ctx["read_leap_hw"]
    stack_obs_num: int = ctx["stack_obs_num"]
    rgb_keys: list[str] = ctx["rgb_keys"]
    image_hw: tuple[int, int] = ctx["image_hw"]
    n_obs_steps: int = ctx["n_obs_steps"]
    n_action_steps: int = ctx["n_action_steps"]
    action_dim: int = ctx["action_dim"]
    latency_step: int = ctx["latency_step"]
    chunk_exec_steps: int = ctx["chunk_exec_steps"]
    sac_obs_mode: str = str(ctx.get("sac_obs_mode", "pixel"))
    sac_imle_feat_concat_action_base: bool = bool(
        ctx.get("sac_imle_feat_concat_action_base", False)
    )

    eval_only = bool(FLAGS.eval_only)
    eval_imle_base_only = bool(FLAGS.eval_imle_base_only)
    eval_episodes_cap = int(FLAGS.eval_episodes) if eval_only else 0
    if eval_only:
        if eval_imle_base_only:
            print_green(
                f"[eval_only] IMLE base-only: a_taken=a_base, without residual/LoRA; "
                f"episode limit: {eval_episodes_cap or 'unlimited'}"
            )
        else:
            print_green(
                f"[eval_only] No learner connection or replay writes; always a_base+alpha*tanh(u), "
                f"deterministic SAC; episode limit: {eval_episodes_cap or 'unlimited'}"
            )
        client = None
    else:
        client = TrainerClient(
            "actor_env",
            FLAGS.ip,
            make_trainer_config(),
            data_store,
            wait_for_server=True,
        )

        def update_params(params):
            nonlocal agent
            agent = agent.replace(state=agent.state.replace(params=params))

        client.recv_network_callback(update_params)

    
    _eval_returns = []
    _eval_lengths = []
    _eval_successes = []

    leap_hist: deque[np.ndarray] = deque(maxlen=int(stack_obs_num))
    obs, _ = env.reset()
    fle._fill_leap_hist_after_reset(leap_hist, stack_obs_num, read_leap_hw)
    if ctx.get("rl_wait_r_after_reset"):
        obs = _wait_for_r_continue(
            env, fle, leap_hist, stack_obs_num, read_leap_hw, obs
        )
    if sac_obs_mode == "imle_preproj" and not eval_imle_base_only:
        _fill_imle_preproj_obs(
            obs,
            policy=policy,
            device=device,
            fle=fle,
            leap_hist=leap_hist,
            rgb_keys=rgb_keys,
            image_hw=image_hw,
            n_obs_steps=n_obs_steps,
            stack_obs_num=stack_obs_num,
            action_base_torch=None,
            read_leap_hw=read_leap_hw,
        )

    prev_traj = torch.randn(
        (1, n_action_steps, action_dim), device=device, dtype=torch.float32
    )
    pred_t: Optional[torch.Tensor] = None
    cached_pred_t: Optional[torch.Tensor] = None
    steps_into_chunk = chunk_exec_steps
    current_chunk_len = chunk_exec_steps
    use_residual_this_chunk = False
    step = 0
    timer = Timer()

    
    _diag_window: deque[float] = deque(maxlen=200)
    _diag_overrun = 0
    _diag_residual_off_violation = 0

    slide_n = int(_EPISODE_SLIDE_LEN)
    ep_ret_acc = 0.0
    ep_len_acc = 0
    ep_slide_returns: deque[float] = deque(maxlen=slide_n)
    ep_slide_lengths: deque[float] = deque(maxlen=slide_n)
    ep_slide_success: deque[float] = deque(maxlen=slide_n)
    episode_i = 0
    dt = 1.0 / max(float(FLAGS.control_hz), 1e-6)
    ra = float(ctx.get("residual_alpha", 0.2))
    res_start_steps = int(ctx.get("res_start_steps", 0))
    res_ramp_steps = int(ctx.get("res_ramp_steps", 500_000))

    while step < FLAGS.max_steps:
        step_t0 = time.time()
        timer.tick("total")

        if steps_into_chunk >= current_chunk_len:
            lp = read_leap_hw()
            if lp is None:
                lp = leap_hist[-1] if leap_hist else np.zeros(16, dtype=np.float32)
            lp = np.asarray(lp, dtype=np.float32).reshape(16)
            leap_hist.append(lp)

            if cached_pred_t is not None:
                pred_t = cached_pred_t
                cached_pred_t = None
            else:
                obs_t = fle.build_policy_obs(
                    obs,
                    rgb_keys=rgb_keys,
                    image_hw=image_hw,
                    n_obs_steps=n_obs_steps,
                    stack_obs_num=int(stack_obs_num),
                    leap_hist=leap_hist,
                )
                obs_t = {k: v.to(device, non_blocking=True) for k, v in obs_t.items()}
                with torch.no_grad():
                    result = policy.bidirection_predict_action(
                        obs_t, None, prev_traj, latency_step=int(latency_step)
                    )
                pred_t = result["action"].detach()
            prev_traj = pred_t
            steps_into_chunk = 0
            current_chunk_len = min(
                int(chunk_exec_steps),
                int(pred_t.shape[1]),
                int(n_action_steps),
            )
            
            
            if eval_imle_base_only:
                use_residual_this_chunk = False
            elif eval_only:
                use_residual_this_chunk = True
            elif step < res_start_steps:
                use_residual_this_chunk = False
            else:
                ramp_step = step - res_start_steps
                if res_ramp_steps <= 0:
                    epsilon = 1.0
                else:
                    epsilon = min(
                        1.0, max(0.0, float(ramp_step) / float(res_ramp_steps))
                    )
                use_residual_this_chunk = bool(random.random() < epsilon)

        assert pred_t is not None
        j = steps_into_chunk
        if j > 0:
            lp = read_leap_hw()
            if lp is None:
                lp = leap_hist[-1] if leap_hist else np.zeros(16, dtype=np.float32)
            leap_hist.append(np.asarray(lp, dtype=np.float32).reshape(16))

        base_t = pred_t[0, j].reshape(-1).to(device=device, dtype=torch.float32)


        with timer.context("sample_actions"):
            if eval_imle_base_only:
                pass  
            elif eval_only:
                if agent is None:
                    raise RuntimeError("eval_only requires a LoRA agent unless eval_imle_base_only is enabled")
                sampling_rng, key_u = jax.random.split(sampling_rng)
                u_jax = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key_u,
                    argmax=True,
                )
                u_t = _jax_array_to_torch(u_jax, torch_device=device).reshape(-1)
            elif step < FLAGS.random_steps:
                u_t = torch.randn(
                    (action_dim,),
                    device=device,
                    dtype=torch.float32,
                )
            else:
                sampling_rng, key_u = jax.random.split(sampling_rng)
                u_jax = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key_u,
                    argmax=False,
                )
                u_t = _jax_array_to_torch(u_jax, torch_device=device).reshape(-1)

        with timer.context("step_env"):
            base_np = base_t.detach().cpu().numpy().astype(np.float32, copy=False)
            j_idx = int(steps_into_chunk)
            res_b = (
                ra * torch.tanh(u_t)
                if not eval_imle_base_only
                else torch.zeros_like(base_t)
            )
            if use_residual_this_chunk:
                final_t = base_t + res_b
            else:
                final_t = base_t.clone()
            final = final_t.detach().cpu().numpy().astype(np.float32, copy=False)
            
            if not use_residual_this_chunk:
                _delta = float(np.max(np.abs(final - base_np)))
                if _delta > 1e-6:
                    _diag_residual_off_violation += 1
                    if _diag_residual_off_violation <= 5:
                        print_yellow(
                            f"[diag][step={step}] residual=off but |final-base|_inf={_delta:.3e}"
                        )
            next_obs, reward, done, truncated, info = env.step(final, base_np)
            reward = np.asarray(reward, dtype=np.float32)
            rw = float(np.asarray(reward, dtype=np.float64).reshape(-1)[0])
            term = bool(done or truncated)
            info = dict(info)
            ep_ret_acc += rw
            ep_len_acc += 1
            if sac_obs_mode == "imle_preproj" and not eval_imle_base_only:
                _fill_imle_preproj_obs(
                    next_obs,
                    policy=policy,
                    device=device,
                    fle=fle,
                    leap_hist=leap_hist,
                    rgb_keys=rgb_keys,
                    image_hw=image_hw,
                    n_obs_steps=n_obs_steps,
                    stack_obs_num=stack_obs_num,
                    action_base_torch=None,
                    read_leap_hw=read_leap_hw,
                )
            if ctx.get("replay_omit_pixels") and sac_obs_mode == "imle_preproj":
                pk = frozenset(ctx["replay_pixel_keys"])
                obs_r = {k: v for k, v in obs.items() if k not in pk}
                next_r = {k: v for k, v in next_obs.items() if k not in pk}
            else:
                obs_r, next_r = obs, next_obs
            obs_r_store = copy.deepcopy(obs_r)
            next_r_store = copy.deepcopy(next_r)
            next_base_np, cached_pred_next = _next_imle_action_base_np(
                next_obs,
                pred_t=pred_t,
                j=j_idx,
                prev_traj=prev_traj,
                policy=policy,
                device=device,
                fle=fle,
                leap_hist=leap_hist,
                rgb_keys=rgb_keys,
                image_hw=image_hw,
                n_obs_steps=n_obs_steps,
                stack_obs_num=stack_obs_num,
                latency_step=latency_step,
                read_leap_hw=read_leap_hw,
                current_chunk_len=current_chunk_len,
            )
            if cached_pred_next is not None:
                cached_pred_t = cached_pred_next


            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")
            action_stored = final.copy()
            if "intervene_action" in info:
                action_stored = np.asarray(
                    info.pop("intervene_action"), dtype=np.float32
                ).copy()

            if not eval_only:
                data_store.insert(
                    dict(
                        observations=obs_r_store,
                        actions=action_stored,
                        action_base=base_np,
                        next_action_base=next_base_np,
                        next_observations=next_r_store,
                        rewards=reward,
                        masks=np.float32(1.0 - float(term)),
                        dones=term,
                    )
                )
            obs = next_obs
            steps_into_chunk += 1
            step += 1

            if bool(info.get("keyboard_success")):
                print_green(
                    "[keyboard] Space success signal received; this step ends the episode"
                )


            if term:
                succ = bool(info.get("succeed", False))
                kbd_succ = bool(info.get("keyboard_success"))
                ep_slide_returns.append(float(ep_ret_acc))
                ep_slide_lengths.append(float(ep_len_acc))
                ep_slide_success.append(1.0 if succ else 0.0)
                episode_i += 1
                inv = float(slide_n)
                slide_succ_rate = sum(ep_slide_success) / inv
                slide_mean_ret = sum(ep_slide_returns) / inv
                slide_mean_len = sum(ep_slide_lengths) / inv
                print(
                    f"[episode end] id={episode_i} global_env_step={step} "
                    f"len={ep_len_acc} return={ep_ret_acc:.4f} succeed={succ} "
                    f"keyboard_success={kbd_succ} | "
                    f"slide{slide_n}: succ_rate={slide_succ_rate:.4f} "
                    f"(one success contributes 1/{slide_n}={1.0 / inv:.2f}) "
                    f"ret_sum/{slide_n}={slide_mean_ret:.4f} len_sum/{slide_n}={slide_mean_len:.2f}"
                )
                print_green(
                    f"[episode] Sliding success rate (denominator={slide_n})= {slide_succ_rate:.4f}"
                )
                if eval_only:
                    _eval_returns.append(float(ep_ret_acc))
                    _eval_lengths.append(int(ep_len_acc))
                    _eval_successes.append(1.0 if succ else 0.0)
                    if eval_episodes_cap > 0 and episode_i >= eval_episodes_cap:
                        print_green(
                            f"[eval_only] Completed {episode_i} episodes; exiting rollout."
                        )
                        
                        step = FLAGS.max_steps + 1
                        break
                else:
                    client.request(
                        "send-stats",
                        {
                            "wandb_step": int(step),
                            
                            "train/success_rate": float(slide_succ_rate),
                            "train/path_length": float(ep_len_acc),
                            "train/episode_r": float(ep_ret_acc),
                            "train/episode_t": float(step),
                            "episode/slide_succ_rate": float(slide_succ_rate),
                            "episode/slide_mean_ret": float(slide_mean_ret),
                            "episode/slide_mean_len": float(slide_mean_len),
                            "episode/index": float(episode_i),
                        },
                    )
                ep_ret_acc = 0.0
                ep_len_acc = 0
                obs, _ = env.reset()
                fle._fill_leap_hist_after_reset(leap_hist, stack_obs_num, read_leap_hw)
                prev_traj = torch.randn(
                    (1, n_action_steps, action_dim), device=device, dtype=torch.float32
                )
                cached_pred_t = None
                steps_into_chunk = chunk_exec_steps
                use_residual_this_chunk = (
                    False
                    if eval_imle_base_only
                    else (True if eval_only else False)
                )
                if ctx.get("rl_wait_r_after_reset"):
                    obs = _wait_for_r_continue(
                        env, fle, leap_hist, stack_obs_num, read_leap_hw, obs
                    )
                if sac_obs_mode == "imle_preproj" and not eval_imle_base_only:
                    _fill_imle_preproj_obs(
                        obs,
                        policy=policy,
                        device=device,
                        fle=fle,
                        leap_hist=leap_hist,
                        rgb_keys=rgb_keys,
                        image_hw=image_hw,
                        n_obs_steps=n_obs_steps,
                        stack_obs_num=stack_obs_num,
                        action_base_torch=None,
                        read_leap_hw=read_leap_hw,
                    )

        elapsed = time.time() - step_t0
        
        _diag_window.append(float(elapsed))
        if elapsed > dt:
            _diag_overrun += 1
        if (step + 1) % 100 == 0 and len(_diag_window) > 0:
            _arr = np.asarray(_diag_window, dtype=np.float64)
            _p50 = float(np.percentile(_arr, 50))
            _p95 = float(np.percentile(_arr, 95))
            _pmax = float(_arr.max())
            print(
                f"[diag][step={step + 1}] step_work_s p50={_p50 * 1000:.1f}ms "
                f"p95={_p95 * 1000:.1f}ms max={_pmax * 1000:.1f}ms "
                f"(dt={dt * 1000:.0f}ms, overruns={_diag_overrun}, "
                f"residual_off_viol={_diag_residual_off_violation}, "
                f"use_residual_this_chunk={use_residual_this_chunk})"
            )
        if elapsed < dt:
            time.sleep(dt - elapsed)

        if not eval_only and step % FLAGS.steps_per_update == 0:
            client.update()

        if not eval_only and step % FLAGS.log_period == 0:
            client.request(
                "send-stats",
                {
                    "wandb_step": int(step),
                    "timer": timer.get_average_times(),
                },
            )

        timer.tock("total")

    
    if eval_only and _eval_returns:
        n = len(_eval_returns)
        mean_ret = float(np.mean(_eval_returns))
        mean_len = float(np.mean(_eval_lengths))
        succ_rate = float(np.mean(_eval_successes))
        print_green("=" * 60)
        print_green(f"[eval_only] SUMMARY  n_episodes={n}")
        print_green(f"  success_rate = {succ_rate:.4f}  ({sum(_eval_successes):.0f}/{n})")
        print_green(f"  mean_return  = {mean_ret:.4f}")
        print_green(f"  mean_length  = {mean_len:.2f}")
        print_green("=" * 60)
        
        _eval_out_base = (
            str(FLAGS.eval_merged_pt).strip()
            if FLAGS.eval_merged_pt
            else str(FLAGS.ckpt_path).strip()
        )
        if _eval_out_base:
            _out_dir = os.path.dirname(os.path.abspath(_eval_out_base))
            _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            _tag = os.path.basename(_eval_out_base).replace(".pt", "").replace(
                ".ckpt", ""
            )
            if FLAGS.eval_imle_base_only:
                _tag = f"imle_base_{_tag}"
            _out_path = os.path.join(_out_dir, f"eval_{_tag}_{_ts}.json")
            try:
                import json
                with open(_out_path, "w") as _f:
                    json.dump({
                        "eval_imle_base_only": bool(FLAGS.eval_imle_base_only),
                        "imle_ckpt": str(FLAGS.ckpt_path),
                        "merged_pt": str(FLAGS.eval_merged_pt or ""),
                        "source_pts": list(FLAGS.eval_source_pts or []),
                        "n_episodes": n,
                        "success_rate": succ_rate,
                        "mean_return": mean_ret,
                        "mean_length": mean_len,
                        "returns": _eval_returns,
                        "lengths": _eval_lengths,
                        "successes": _eval_successes,
                    }, _f, indent=2)
                print_green(f"[eval_only] Results saved: {_out_path}")
            except Exception as _e:
                print_yellow(f"[eval_only] Save failed (stdout unaffected): {_e}")


def learner(
    _rng,
    agent: SACAgent,
    replay_buffer,
    replay_iterator,
    demo_replay_buffer=None,
    demo_replay_iterator=None,
):
    ckpt_parent = (str(FLAGS.checkpoint_path).strip() if FLAGS.checkpoint_path else "")
    need_ckpt = int(FLAGS.checkpoint_period) > 0
    buf_save_period = _effective_buffer_save_period()
    if (need_ckpt or buf_save_period > 0) and not ckpt_parent:
        raise SystemExit(
            "Checkpoint/replay saving requires --checkpoint_path; "
            "use a parent directory for a new run, or a previous run directory with --resume/existing checkpoint_* files."
        )

    _we = str(FLAGS.wandb_entity or "").strip()
    wandb_logger = make_wandb_logger(
        project=str(FLAGS.wandb_project),
        description=FLAGS.exp_name,
        debug=FLAGS.debug,
        entity=(_we if _we else None),
        wandb_mode=(FLAGS.wandb_mode if not FLAGS.debug else None),
    )
    wb_run = getattr(wandb_logger, "run", None) if wandb_logger is not None else None

    run_name = getattr(wb_run, "name", None) if wb_run is not None else None
    if not run_name:
        ts = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        desc = str(FLAGS.exp_name or "").strip()
        run_name = f"{desc}_{ts}" if desc else ts

    if need_ckpt or buf_save_period > 0:
        run_ckpt, resumed = _resolve_run_checkpoint_dir(
            ckpt_parent,
            run_name,
            resume_explicit=bool(FLAGS.resume),
        )
        FLAGS.checkpoint_path = run_ckpt
        if resumed:
            print_green(f"[resume] Run directory: {run_ckpt}")
        else:
            print_green(f"[ckpt] New run directory (matching W&B run name): {run_ckpt}")

    start_update_step = 0
    if need_ckpt or buf_save_period > 0:
        run_dir = str(FLAGS.checkpoint_path)
        if _is_run_checkpoint_dir(run_dir):
            agent, start_update_step = _restore_agent_from_run_dir(
                agent, run_dir, step=int(FLAGS.checkpoint_step)
            )
            _restore_online_replay_buffer(replay_buffer, run_dir)
            if start_update_step > 0:
                _write_training_meta(run_dir, start_update_step - 1, agent)

    update_steps = start_update_step

    def stats_callback(req_type: str, payload: dict) -> dict:
        assert req_type == "send-stats", req_type
        if wandb_logger is not None:
            data = dict(payload)
            wb = data.pop("wandb_step", None)
            data["env_step"] = int(wb) if wb is not None else int(update_steps)
            if data:
                wandb_logger.log(data)
        return {}

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    online_ready = int(len(replay_buffer))
    demo_ready = int(len(demo_replay_buffer)) if demo_replay_buffer is not None else 0
    ready_total = online_ready + demo_ready

    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=ready_total,
        desc="Filling up replay buffer",
    )
    while (int(len(replay_buffer)) + demo_ready) < FLAGS.training_starts:
        pbar.update((int(len(replay_buffer)) + demo_ready) - pbar.n)
        time.sleep(1)
    pbar.update((int(len(replay_buffer)) + demo_ready) - pbar.n)
    pbar.close()

    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    timer = Timer()
    pbar = tqdm.tqdm(
        total=FLAGS.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    if start_update_step >= int(FLAGS.max_steps):
        print_yellow(
            f"[resume] start_update_step={start_update_step} >= max_steps={FLAGS.max_steps}，"
            "No gradient updates remaining; increase --max_steps to continue."
        )
    for _ in tqdm.tqdm(
        range(start_update_step, int(FLAGS.max_steps)),
        dynamic_ncols=True,
        desc="learner",
        initial=start_update_step,
        total=int(FLAGS.max_steps),
    ):
        with timer.context("sample_replay_buffer"):
            if demo_replay_iterator is None:
                batch = next(replay_iterator)
            else:
                
                
                if len(replay_buffer) > 0:
                    batch_online = next(replay_iterator)
                    batch_demo = next(demo_replay_iterator)
                    batch = _concat_batches(batch_online, batch_demo)
                else:
                    batch = next(demo_replay_iterator)
        with timer.context("train"):
            _cw_res = (
                int(FLAGS.critic_warmup_steps) > 0
                and update_steps < int(FLAGS.critic_warmup_steps)
            )
            _main_nets_res = (
                frozenset({"critic"})
                if _cw_res
                else frozenset({"actor", "critic", "temperature"})
            )
            agent, update_info = agent.update(batch, networks_to_update=_main_nets_res)
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(
                {
                    **update_info,
                    "update_step": int(update_steps),
                    "train/critic_warmup": 1.0 if _cw_res else 0.0,
                }
            )
            wandb_logger.log(
                {"timer": timer.get_average_times(), "update_step": int(update_steps)}
            )
            if demo_replay_buffer is not None:
                wandb_logger.log(
                    {
                        "replay/online_size": float(len(replay_buffer)),
                        "replay/demo_size": float(len(demo_replay_buffer)),
                        "update_step": int(update_steps),
                    },
                )

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            print(
                f"[ckpt] step={update_steps}: saving agent checkpoint...",
                flush=True,
            )
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path,
                agent.state,
                step=update_steps,
                keep=20,
                overwrite=True,
            )
            _write_training_meta(FLAGS.checkpoint_path, update_steps, agent)
            print(f"[ckpt] step={update_steps}: agent checkpoint saved", flush=True)
            
            if FLAGS.sac_obs_mode == "imle_preproj":
                _merge_path = os.path.join(
                    FLAGS.checkpoint_path,
                    f"task_{FLAGS.exp_name}_step{int(update_steps)}_for_lora_merge.pt",
                )
                try:
                    lora_agent.save_for_merge(
                        agent,
                        _merge_path,
                        task_id=str(FLAGS.exp_name),
                        additional_info={
                            "global_step": int(update_steps),
                            "lora_rank": int(FLAGS.lora_rank),
                            "residual_alpha": float(agent.config["residual_alpha"]),
                            "base_checkpoint": str(FLAGS.ckpt_path),
                        },
                    )
                except Exception as _e:
                    print(f"[imle_lora] save_for_merge failed (training continues): {_e}")


                _states_path = os.path.join(
                    FLAGS.checkpoint_path,
                    f"states_imle_preproj_step{int(update_steps)}.npy",
                )
                try:
                    _accum = []
                    _target_n = 2048
                    for _ in range(16):  
                        _b = next(replay_iterator)
                        _obs = _b.get("observations") if hasattr(_b, "get") else _b["observations"]
                        _ip = _obs.get("imle_preproj") if hasattr(_obs, "get") else None
                        if _ip is None:
                            break
                        _accum.append(np.asarray(jax.device_get(_ip), dtype=np.float32))
                        if sum(a.shape[0] for a in _accum) >= _target_n:
                            break
                    if _accum:
                        _arr = np.concatenate(_accum, axis=0)[:_target_n]
                        if _arr.ndim > 2:
                            _arr = _arr.reshape(_arr.shape[0], -1)
                        np.save(_states_path, _arr)
                        print(
                            f"[imle_lora] Saved states.npy: shape={_arr.shape} to {_states_path}"
                        )
                except Exception as _e:
                    print(f"[imle_lora] states.npy save failed (training continues): {_e}")

            if buf_save_period > 0 and update_steps % buf_save_period == 0:
                try:
                    _save_online_replay_buffer(
                        replay_buffer, FLAGS.checkpoint_path, update_steps
                    )
                except Exception as _e:
                    print_yellow(f"[buffer] Save failed (training continues): {_e}")

        pbar.update(len(replay_buffer) - pbar.n)
        update_steps += 1


def main(argv):
    del argv  
    global _RES_IMLE_CTX

    if not os.environ.get("NO_PROXY"):
        os.environ["NO_PROXY"] = (
            "localhost,127.0.0.1,::1,192.168.0.0/16,10.0.0.0/8,172.16.0.0/12"
        )
    _no_proxy_extra = (os.environ.get("NO_PROXY_EXTRA") or "").strip().strip(",")
    if _no_proxy_extra:
        os.environ["NO_PROXY"] = f"{os.environ['NO_PROXY']},{_no_proxy_extra}"

    if not (FLAGS.learner ^ FLAGS.actor):
        raise SystemExit("Specify exactly one of --learner or --actor")
    if FLAGS.eval_merged_pt and not (FLAGS.actor and FLAGS.eval_only):
        raise SystemExit("--eval_merged_pt requires --actor and --eval_only")

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    assert FLAGS.batch_size % len(devices) == 0

    imle_root = FLAGS.imle_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    _setup_imle_sys_path(imle_root)

    fle = _load_eval_module()
    repo = _leaphand_repo_root()
    leap_api = FLAGS.leap_api_root or os.path.join(repo, "LEAP_Hand_API", "python")

    from experiments.mappings import CONFIG_MAPPING

    if FLAGS.exp_name not in CONFIG_MAPPING:
        raise ValueError(f"Unknown exp_name={FLAGS.exp_name!r}")
    train_cfg = CONFIG_MAPPING[FLAGS.exp_name]()
    target_entropy = (
        None if np.isnan(FLAGS.target_entropy) else float(FLAGS.target_entropy)
    )
    residual_alpha = (
        float(getattr(train_cfg, "residual_alpha", 0.05))
        if np.isnan(FLAGS.residual_alpha)
        else float(FLAGS.residual_alpha)
    )

    if not np.isfinite(residual_alpha) or residual_alpha <= 0:
        raise SystemExit("residual_alpha must be finite and positive")

    if FLAGS.replay_omit_pixels and FLAGS.sac_obs_mode != "imle_preproj":
        raise SystemExit(
            "--replay_omit_pixels requires --sac_obs_mode=imle_preproj; "
            "pixel mode needs camera frames in replay."
        )
    if FLAGS.sac_pixel_backend == "drq" and FLAGS.sac_obs_mode != "pixel":
        raise SystemExit(
            "--sac_pixel_backend=drq requires --sac_obs_mode=pixel."
        )
    alpha_max = None if float(FLAGS.alpha_max) <= 0 else float(FLAGS.alpha_max)
    _, sampling_rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed))

    if FLAGS.learner:
        env = train_cfg.get_environment(
            fake_env=True,
            save_video=False,
            stack_obs_num=int(FLAGS.stack_obs_num),
            leap_node=None,
            leap_send_commands=False,
            use_spacemouse=False,
            include_imle_action_base=bool(FLAGS.sac_imle_feat_concat_action_base),
        )
        fle._stop_franka_pynput_listener(env)

        if FLAGS.sac_obs_mode == "imle_preproj":
            if not str(FLAGS.ckpt_path or "").strip():
                raise SystemExit(
                    "The imle_preproj learner requires --ckpt_path to load IMLE and infer the feature dimension."
                )
            precise_l = FLAGS.precise_dex_root or fle._default_precise_dex_root()
            imle_cfg_l, policy_l, device_l, image_hw_l = _load_psi_policy(
                list(FLAGS.hydra_override),
                FLAGS.ckpt_path,
                precise_l,
                FLAGS.device,
            )
            obs_meta_l = imle_cfg_l.task.shape_meta["obs"]
            rgb_keys_l = [k for k in obs_meta_l if obs_meta_l[k].get("type") == "rgb"]
            n_obs_steps_l = int(imle_cfg_l.n_obs_steps)
            action_dim_l = int(imle_cfg_l.task.shape_meta["action"]["shape"][0])
            leap_hist_l: deque[np.ndarray] = deque(maxlen=int(FLAGS.stack_obs_num))
            obs_probe_l, _ = env.reset()
            fle._fill_leap_hist_after_reset(
                leap_hist_l, int(FLAGS.stack_obs_num), lambda: None
            )
            dim_l = _probe_flat_imle_feat_dim(
                policy_l,
                device_l,
                fle,
                obs_probe_l,
                leap_hist_l,
                rgb_keys=rgb_keys_l,
                image_hw=image_hw_l,
                n_obs_steps=n_obs_steps_l,
                stack_obs_num=int(FLAGS.stack_obs_num),
                concat_action_base=bool(FLAGS.sac_imle_feat_concat_action_base),
                action_dim=action_dim_l,
            )
            _preproj_extra = {
                "imle_preproj": gym.spaces.Box(
                    low=-1e10,
                    high=1e10,
                    shape=(dim_l,),
                    dtype=np.float32,
                ),
            }
            env = AugmentDictObsWrapper(env, _preproj_extra)
            obs0, _ = env.reset()
            fle._fill_leap_hist_after_reset(
                leap_hist_l, int(FLAGS.stack_obs_num), lambda: None
            )
            agent = make_sac_residual_imle_preproj_lora_agent(
                FLAGS.seed,
                obs0,
                env.action_space.sample(),
                residual_alpha=residual_alpha,
                alpha_max=alpha_max,
                image_keys=tuple(train_cfg.image_keys),
                discount=float(train_cfg.discount),
                reward_bias=float(getattr(train_cfg, "reward_bias", 0.0) or 0.0),
                target_entropy=target_entropy,
                lora_rank=int(FLAGS.lora_rank),
            )
        else:
            obs0, _ = env.reset()
            if FLAGS.sac_pixel_backend == "drq":
                agent = make_drq_residual_imle_pixel_agent(
                    FLAGS.seed,
                    obs0,
                    env.action_space.sample(),
                    residual_alpha=residual_alpha,
                    alpha_max=alpha_max,
                    image_keys=tuple(train_cfg.image_keys),
                    encoder_type=train_cfg.encoder_type,
                    discount=float(train_cfg.discount),
                    reward_bias=float(getattr(train_cfg, "reward_bias", 0.0) or 0.0),
                    target_entropy=target_entropy,
                )
            else:
                agent = make_sac_residual_imle_pixel_agent(
                    FLAGS.seed,
                    obs0,
                    env.action_space.sample(),
                    residual_alpha=residual_alpha,
                    alpha_max=alpha_max,
                    image_keys=tuple(train_cfg.image_keys),
                    encoder_type=train_cfg.encoder_type,
                    discount=float(train_cfg.discount),
                    reward_bias=float(getattr(train_cfg, "reward_bias", 0.0) or 0.0),
                    target_entropy=target_entropy,
                )
        agent = jax.device_put(jax.tree_util.tree_map(jnp.array, agent), sharding.replicate())
        pixel_keys_t = tuple(train_cfg.image_keys)
        _replay_residual = True
        if FLAGS.sac_obs_mode == "imle_preproj" and FLAGS.replay_omit_pixels:
            pixel_keys_t = _replay_omit_pixel_keys(env)
            replay_buffer = ReplayBufferDataStore(
                dict_observation_space_drop_keys(env.observation_space, pixel_keys_t),
                env.action_space,
                FLAGS.replay_buffer_capacity,
                include_action_base=_replay_residual,
            )
            replay_pp = None
        else:
            replay_buffer = MemoryEfficientReplayBufferDataStore(
                env.observation_space,
                env.action_space,
                capacity=FLAGS.replay_buffer_capacity,
                image_keys=train_cfg.image_keys,
                include_action_base=_replay_residual,
            )
            replay_pp = (
                (lambda b: strip_replay_batch_pixel_keys(b, pixel_keys_t))
                if FLAGS.sac_obs_mode == "imle_preproj"
                else None
            )
        replay_iterator = replay_buffer.get_iterator(
            sample_args={"batch_size": FLAGS.batch_size * FLAGS.critic_actor_ratio},
            device=sharding.replicate(),
            batch_postprocess=replay_pp,
        )
        demo_replay_buffer = None
        demo_replay_iterator = None
        if FLAGS.demo_paths:
            demo_paths = [str(p).strip() for p in FLAGS.demo_paths if str(p).strip()]
            if len(demo_paths) == 0:
                raise SystemExit("--demo_paths resolved to an empty list; check arguments.")
            for pth in demo_paths:
                if not os.path.isfile(pth):
                    raise FileNotFoundError(f"Demo pickle does not exist: {pth}")

            if FLAGS.sac_obs_mode == "imle_preproj" and FLAGS.replay_omit_pixels:
                demo_replay_buffer = ReplayBufferDataStore(
                    dict_observation_space_drop_keys(env.observation_space, pixel_keys_t),
                    env.action_space,
                    FLAGS.replay_buffer_capacity,
                    include_action_base=_replay_residual,
                )
                demo_pp = None
            else:
                demo_replay_buffer = MemoryEfficientReplayBufferDataStore(
                    env.observation_space,
                    env.action_space,
                    capacity=FLAGS.replay_buffer_capacity,
                    image_keys=train_cfg.image_keys,
                    include_action_base=_replay_residual,
                )
                demo_pp = (
                    (lambda b: strip_replay_batch_pixel_keys(b, pixel_keys_t))
                    if FLAGS.sac_obs_mode == "imle_preproj"
                    else None
                )

            populate_data_store(demo_replay_buffer, demo_paths)
            if len(demo_replay_buffer) == 0:
                raise SystemExit("Demo buffer is empty; check the pickle contents.")

            full_bs = int(FLAGS.batch_size * FLAGS.critic_actor_ratio)
            demo_bs = full_bs // 2
            online_bs = full_bs - demo_bs
            if demo_bs <= 0 or online_bs <= 0:
                raise SystemExit(
                    "batch_size * critic_actor_ratio is too small for 50/50 sampling."
                )

            replay_iterator = replay_buffer.get_iterator(
                sample_args={"batch_size": online_bs},
                device=sharding.replicate(),
                batch_postprocess=replay_pp,
            )
            demo_replay_iterator = demo_replay_buffer.get_iterator(
                sample_args={"batch_size": demo_bs},
                device=sharding.replicate(),
                batch_postprocess=demo_pp,
            )
            print_green(
                "[learner] Demonstration replay enabled: sampling online/demo 50/50 "
                f"(online_bs={online_bs}, demo_bs={demo_bs}, demo_size={len(demo_replay_buffer)})"
            )
        _RES_IMLE_CTX = None
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        _h_cfg = int(getattr(train_cfg, "H", 500_000))
        print_green(
            f"starting learner loop | H={_h_cfg} | "
            "SACResidualIMLE + replay(actions=a_taken, action_base, next_action_base)"
        )
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            replay_iterator,
            demo_replay_buffer=demo_replay_buffer,
            demo_replay_iterator=demo_replay_iterator,
        )
        return

    leap_node = None
    if not FLAGS.no_leap:
        if leap_api not in sys.path:
            sys.path.insert(0, leap_api)
        try:
            from main import LeapNode

            leap_node = LeapNode()
        except Exception as e:
            raise RuntimeError(
                f"Could not create LeapNode (use --no_leap to disable). Original error: {e}"
            ) from e

    if FLAGS.no_cameras:
        fle._install_dummy_realsense_cameras(FLAGS.exp_name)

    rl_one = bool(FLAGS.rl_actor_human)
    hil = bool(FLAGS.hil)
    
    if hil and rl_one:
        raise SystemExit("--hil and --rl_actor_human are mutually exclusive: SpaceMouse intervention versus keyboard-only control.")
    if hil and bool(FLAGS.eval_only):
        raise SystemExit(
            "--hil and --eval_only are mutually exclusive; evaluation has no learner/replay destination."
        )
    if FLAGS.eval_imle_base_only:
        if not FLAGS.eval_only:
            raise SystemExit("--eval_imle_base_only requires --eval_only.")
        if str(FLAGS.eval_merged_pt or "").strip():
            raise SystemExit(
                "--eval_imle_base_only and --eval_merged_pt are mutually exclusive."
            )
        if FLAGS.sac_obs_mode != "imle_preproj":
            raise SystemExit(
                "--eval_imle_base_only requires --sac_obs_mode=imle_preproj."
            )
    
    if hil:
        use_sm = True
        use_kbd = True
        wait_r = True
    elif rl_one or bool(FLAGS.eval_only):
        use_sm = False
        use_kbd = True
        wait_r = True
    else:
        use_sm = bool(FLAGS.use_spacemouse)
        use_kbd = bool(FLAGS.use_keyboard_success)
        wait_r = bool(FLAGS.rl_wait_r_after_reset)
    if use_kbd and FLAGS.fake_env:
        print_yellow(
            "[actor] fake_env=True disables KeyboardSuccessWrapper; "
            "use --fake_env=false for space-key success labels."
        )
    if hil:
        print_green(
            "[actor] HIL: SpaceMouse intervention; space=success, f=failure; r=start and t=reset after reset."
            " During intervention, info['intervene_action'] replaces the policy action in replay."
        )
    elif rl_one:
        print_green(
            "[actor] Keyboard mode: SpaceMouse disabled; space=success, f=failure; r=start, t=reset."
        )
    elif bool(FLAGS.eval_only):
        if FLAGS.eval_imle_base_only:
            print_green(
                "[actor] IMLE base-only evaluation: no LoRA residual; "
                "after reset: r=start, space=success, f=failure, t=reset again."
            )
        else:
            print_green(
                "[actor] Evaluation uses the online RL keyboard workflow: "
                "after reset: r=start, space=success, f=failure, t=reset again."
            )

    env = train_cfg.get_environment(
        fake_env=FLAGS.fake_env,
        save_video=False,
        stack_obs_num=int(FLAGS.stack_obs_num),
        leap_node=leap_node,
        leap_send_commands=leap_node is not None,
        use_spacemouse=use_sm,
        use_keyboard_success=use_kbd,
        include_imle_action_base=bool(FLAGS.sac_imle_feat_concat_action_base),
    )
    fle._stop_franka_pynput_listener(env)

    precise = FLAGS.precise_dex_root or fle._default_precise_dex_root()
    imle_cfg, policy, device, image_hw = _load_psi_policy(
        list(FLAGS.hydra_override), FLAGS.ckpt_path, precise, FLAGS.device
    )

    n_action_steps = int(imle_cfg.n_action_steps)
    latency_step = int(FLAGS.latency_step)
    if n_action_steps <= 8 + latency_step:
        raise ValueError(
            f"n_action_steps ({n_action_steps}) must exceed 8 + latency_step ({8 + latency_step}); "
            "set --hydra_override=n_action_steps=40 or another compatible horizon."
        )

    n_obs_steps = int(imle_cfg.n_obs_steps)
    action_dim = int(imle_cfg.task.shape_meta["action"]["shape"][0])
    obs_meta = imle_cfg.task.shape_meta["obs"]
    rgb_keys = [k for k in obs_meta if obs_meta[k].get("type") == "rgb"]
    if int(FLAGS.stack_obs_num) != len(rgb_keys):
        raise ValueError(
            f"--stack_obs_num={FLAGS.stack_obs_num} does not match the {len(rgb_keys)} RGB streams."
        )

    def read_leap_hw() -> Optional[np.ndarray]:
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

    leap_hist_probe: deque[np.ndarray] = deque(maxlen=int(FLAGS.stack_obs_num))
    obs_probe, _ = env.reset()
    fle._fill_leap_hist_after_reset(
        leap_hist_probe, int(FLAGS.stack_obs_num), read_leap_hw
    )
    _eval_imle_base_only = bool(FLAGS.eval_imle_base_only)
    if FLAGS.sac_obs_mode == "imle_preproj" and not _eval_imle_base_only:
        dim = _probe_flat_imle_feat_dim(
            policy,
            device,
            fle,
            obs_probe,
            leap_hist_probe,
            rgb_keys=rgb_keys,
            image_hw=image_hw,
            n_obs_steps=n_obs_steps,
            stack_obs_num=int(FLAGS.stack_obs_num),
            concat_action_base=bool(FLAGS.sac_imle_feat_concat_action_base),
            action_dim=action_dim,
        )
        _preproj_extra_a = {
            "imle_preproj": gym.spaces.Box(
                low=-1e10,
                high=1e10,
                shape=(dim,),
                dtype=np.float32,
            ),
        }
        env = AugmentDictObsWrapper(env, _preproj_extra_a)

    obs0, _ = env.reset()

    agent: Optional[SACAgent] = None
    if not _eval_imle_base_only:
        if FLAGS.sac_obs_mode == "imle_preproj":
            agent = make_sac_residual_imle_preproj_lora_agent(
                FLAGS.seed,
                obs0,
                env.action_space.sample(),
                residual_alpha=residual_alpha,
                alpha_max=alpha_max,
                image_keys=tuple(train_cfg.image_keys),
                discount=float(train_cfg.discount),
                reward_bias=float(getattr(train_cfg, "reward_bias", 0.0) or 0.0),
                target_entropy=target_entropy,
                lora_rank=int(FLAGS.lora_rank),
            )
        else:
            if FLAGS.sac_pixel_backend == "drq":
                agent = make_drq_residual_imle_pixel_agent(
                    FLAGS.seed,
                    obs0,
                    env.action_space.sample(),
                    residual_alpha=residual_alpha,
                    alpha_max=alpha_max,
                    image_keys=tuple(train_cfg.image_keys),
                    encoder_type=train_cfg.encoder_type,
                    discount=float(train_cfg.discount),
                    reward_bias=float(getattr(train_cfg, "reward_bias", 0.0) or 0.0),
                    target_entropy=target_entropy,
                )
            else:
                agent = make_sac_residual_imle_pixel_agent(
                    FLAGS.seed,
                    obs0,
                    env.action_space.sample(),
                    residual_alpha=residual_alpha,
                    alpha_max=alpha_max,
                    image_keys=tuple(train_cfg.image_keys),
                    encoder_type=train_cfg.encoder_type,
                    discount=float(train_cfg.discount),
                    reward_bias=float(getattr(train_cfg, "reward_bias", 0.0) or 0.0),
                    target_entropy=target_entropy,
                )
        agent = jax.device_put(
            jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
        )
    else:
        print_green(
            "[eval] IMLE base-only: skipping the JAX LoRA actor; loading only the PyTorch checkpoint"
        )

    
    if FLAGS.eval_merged_pt:
        if FLAGS.sac_obs_mode != "imle_preproj":
            raise SystemExit(
                "--eval_merged_pt requires imle_preproj mode; pixel actors have a different architecture."
            )
        _src_pts = [p for p in (FLAGS.eval_source_pts or []) if str(p).strip()]
        print_green(
            f"[eval] Loading LoRA parameters: {FLAGS.eval_merged_pt} "
            f"(LN sources: {_src_pts or 'embedded _jax_extras'})"
        )
        agent = lora_agent.load_merged_for_actor(
            agent,
            merged_pt_path=str(FLAGS.eval_merged_pt),
            source_pt_paths_for_ln=(_src_pts or None),
        )
        
        agent = jax.device_put(jax.tree_util.tree_map(jnp.array, agent), sharding.replicate())
        print_green(
            "[eval] Actor parameters loaded; residual is always applied without epsilon-greedy/random exploration"
        )

    _RES_IMLE_CTX = {
        "eval_mod": fle,
        "policy": policy,
        "device": device,
        "imle_cfg": imle_cfg,
        "read_leap_hw": read_leap_hw,
        "stack_obs_num": int(FLAGS.stack_obs_num),
        "rgb_keys": rgb_keys,
        "image_hw": image_hw,
        "n_obs_steps": n_obs_steps,
        "n_action_steps": n_action_steps,
        "action_dim": action_dim,
        "latency_step": latency_step,
        "chunk_exec_steps": int(FLAGS.chunk_exec_steps),
        "sac_obs_mode": str(FLAGS.sac_obs_mode),
        "sac_imle_feat_concat_action_base": bool(
            FLAGS.sac_imle_feat_concat_action_base
        ),
        "replay_omit_pixels": bool(FLAGS.replay_omit_pixels),
        "replay_pixel_keys": (
            _replay_omit_pixel_keys(env)
            if FLAGS.replay_omit_pixels and FLAGS.sac_obs_mode == "imle_preproj"
            else tuple(train_cfg.image_keys)
        ),
        "residual_alpha": residual_alpha,
        "res_start_steps": int(FLAGS.res_start_steps),
        "res_ramp_steps": int(FLAGS.res_ramp_steps),
        "rl_wait_r_after_reset": wait_r,
        "eval_imle_base_only": _eval_imle_base_only,
    }
    sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
    data_store = QueuedDataStore(2000)
    if FLAGS.eval_only:
        if _eval_imle_base_only:
            print_green(
                f"starting eval IMLE base-only | ckpt={FLAGS.ckpt_path} | "
                f"no LoRA / no residual / no replay"
            )
        else:
            print_green(
                f"starting eval actor | α={_RES_IMLE_CTX['residual_alpha']} | "
                f"{FLAGS.sac_obs_mode}: always a_base+α·tanh(u), deterministic SAC, no replay"
            )
    else:
        print_green(
            f"starting online actor | res_start={_RES_IMLE_CTX['res_start_steps']} | "
            f"res_ramp={_RES_IMLE_CTX['res_ramp_steps']} | "
            f"α={_RES_IMLE_CTX['residual_alpha']} | "
            f"{FLAGS.sac_obs_mode}: ε-greedy residual + full-action replay"
        )
    actor(agent, data_store, env, sampling_rng)


if __name__ == "__main__":
    app.run(main)
