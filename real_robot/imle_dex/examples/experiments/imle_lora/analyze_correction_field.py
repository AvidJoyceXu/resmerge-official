#!/usr/bin/env python3
# Modified by the ResMerge contributors, 2026-10-31; see real_robot/THIRD_PARTY.md.
"""
Offline merge-feasibility analysis for IMLE LoRA residual actors. Adapted from
libero-sac/continual/eval/analyze_correction_vector_field.py. Query both actors on shared
imle_preproj states and compute paper RFC = DC * MC for residual alpha*tanh(mean_u).
Also report absolute log norm ratio and the fraction of low-direction-agreement states.
Read states from states_npy or
feature-enriched demo_pkl. Output plots, raw arrays and a merge recommendation report.
See README.md for commands.
"""

from __future__ import annotations


import importlib.util
import os
import sys


def _imle_root_from_this_file() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _prepend_imle_sys_paths(imle_root: str) -> None:
    root = os.path.abspath(imle_root)
    for sub in ("octo", "serl_launcher", "serl_robot_infra", "examples"):
        p = os.path.join(root, sub)
        if not os.path.isdir(p):
            continue
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


_prepend_imle_sys_paths(_imle_root_from_this_file())


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")  

import datetime
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from absl import app, flags



_lora_agent_path = os.path.join(os.path.dirname(__file__), "lora_agent.py")
_lora_spec = importlib.util.spec_from_file_location("lora_agent", _lora_agent_path)
lora_agent = importlib.util.module_from_spec(_lora_spec)
sys.modules["lora_agent"] = lora_agent  
_lora_spec.loader.exec_module(lora_agent)


# ============================================================================
# Flags
# ============================================================================
FLAGS = flags.FLAGS

flags.DEFINE_string("task_i_pt", "", "Task i exported actor or merged .pt file (required).")
flags.DEFINE_string("task_j_pt", "", "Task j exported actor or merged .pt file (required).")
flags.DEFINE_list(
    "source_pts_i", [],
    "Source actor exports for LayerNorm recovery if task_i is a merged file without _jax_extras.",
)
flags.DEFINE_list(
    "source_pts_j", [],
    "Source actor exports for LayerNorm recovery if task_j is a merged file without _jax_extras.",
)

flags.DEFINE_string(
    "states_npy", "",
    "Float32 state array shaped (N,obs_dim); specify this or demo_pkl, not both.",
)
flags.DEFINE_string(
    "demo_pkl", "",
    "SERL replay pickle containing observations with imle_preproj features.",
)
flags.DEFINE_integer("max_states", 2000, "Maximum states to analyze; larger datasets are randomly subsampled.")

flags.DEFINE_float("residual_alpha", 0.05, "Residual scale used during training: alpha*tanh(mean_u).")
flags.DEFINE_float("tau", 0.5, "Minimum paper RFC for the merge recommendation.")
flags.DEFINE_enum("decision_rule", "rfc", ["rfc", "legacy"],
                  "Use paper RFC, or the legacy three-threshold feasibility diagnostic.")

flags.DEFINE_float(
    "delta_dir", 0.3,
    "States with direction cosine below delta_dir are counted as low-agreement states.",
)
flags.DEFINE_float(
    "threshold_dir", 0.5,
    "Minimum mean direction cosine for the merge recommendation.",
)
flags.DEFINE_float(
    "threshold_scale", 1.0,
    "Maximum mean absolute log norm ratio (1.0 corresponds to a scale ratio around e).",
)
flags.DEFINE_float(
    "threshold_bad", 0.10,
    "Maximum fraction of low-agreement states.",
)

flags.DEFINE_string("output_dir", "./correction_field_out", "Output directory for analysis artifacts.")
flags.DEFINE_integer("seed", 0, "Random seed for state subsampling and agent initialization.")
flags.DEFINE_integer("batch_size", 64, "Batch size for actor inference.")


# ============================================================================

# ============================================================================

def _params_dict_from_ckpt(ckpt: dict) -> dict:
    """
    Read single-task params or merged_params from a checkpoint.
    """
    if "params" in ckpt:
        return ckpt["params"]
    if "merged_params" in ckpt:
        return ckpt["merged_params"]
    raise KeyError(
        f"Checkpoint has neither params nor merged_params; keys={list(ckpt.keys())}"
    )


def build_agent_from_lora_pt(
    lora_pt_path: str,
    residual_alpha: float,
    seed: int,
    source_pt_paths_for_ln: Optional[List[str]] = None,
):
    """
    Construct an agent using dimensions from export metadata and load actor weights. No robot
    environment or IMLE base model is needed. Merged exports without _jax_extras require
    source exports to recover LayerNorm.
    """
    ckpt = torch.load(lora_pt_path, map_location="cpu", weights_only=False)
    params = _params_dict_from_ckpt(ckpt)
    meta = params["meta"]
    obs_dim = int(meta["obs_dim"])
    action_dim = int(meta["action_dim"])
    rank = int(meta["rank"])

    
    sample_obs = {"imle_preproj": np.zeros((1, obs_dim), dtype=np.float32)}
    sample_action = np.zeros((action_dim,), dtype=np.float32)

    agent = lora_agent.make_sac_residual_imle_preproj_lora_agent(
        seed=int(seed),
        sample_obs=sample_obs,
        sample_action=sample_action,
        residual_alpha=float(residual_alpha),
        lora_rank=rank,
    )

    ln_sources = source_pt_paths_for_ln or None
    if ln_sources is None and "merged_params" in ckpt and "_jax_extras" not in ckpt:
        raise ValueError(
            f"{lora_pt_path} is a merged export without _jax_extras; "
            "provide the original task exports through --source_pts_i / --source_pts_j."
        )

    
    agent = lora_agent.load_merged_for_actor(
        agent,
        lora_pt_path,
        source_pt_paths_for_ln=ln_sources,
    )
    return agent, obs_dim, action_dim, rank


# ============================================================================

# ============================================================================

def collect_states_from_npy(npy_path: str, max_states: int, seed: int) -> np.ndarray:
    arr = np.load(npy_path)
    if arr.ndim != 2:
        raise ValueError(f"{npy_path} must have shape (N,obs_dim), got {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    if arr.shape[0] > max_states:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(arr.shape[0], max_states, replace=False)
        arr = arr[idx]
    print(f"[states] from npy: {npy_path}  → shape={arr.shape}")
    return arr


def collect_states_from_pkl(
    pkl_path: str, max_states: int, seed: int
) -> np.ndarray:
    """
    Extract imle_preproj from a list of SERL transitions or trajectory dictionaries. Both
    current and next observations contribute samples. Missing features require offline feature
    preprocessing first.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"{pkl_path} must contain a nonempty list.")

    samples: List[np.ndarray] = []

    def _maybe_collect_from_obs(obs_obj):
        if isinstance(obs_obj, dict):
            v = obs_obj.get("imle_preproj")
            if v is not None:
                arr = np.asarray(v, dtype=np.float32).reshape(-1)
                samples.append(arr)

    for item in data:
        if not isinstance(item, dict):
            continue
        obs = item.get("observations") or item.get("obs")
        if obs is None:
            continue
        if isinstance(obs, dict):
            _maybe_collect_from_obs(obs)
        elif isinstance(obs, list):
            for o in obs:
                _maybe_collect_from_obs(o)
        
        n_obs = item.get("next_observations") or item.get("next_obs")
        if isinstance(n_obs, dict):
            _maybe_collect_from_obs(n_obs)
        elif isinstance(n_obs, list):
            for o in n_obs:
                _maybe_collect_from_obs(o)

    if len(samples) == 0:
        raise SystemExit(
            f"{pkl_path} contains no imle_preproj observations. "
            f"Use --states_npy with saved learner feature samples, "
            f"or generate feature-enriched replay using the frozen IMLE base checkpoint."
        )

    arr = np.stack(samples, axis=0)
    if arr.shape[0] > max_states:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(arr.shape[0], max_states, replace=False)
        arr = arr[idx]
    print(f"[states] from pkl: {pkl_path}; found {len(samples)}, using {arr.shape[0]}")
    return arr


# ============================================================================

# ============================================================================

def query_residual_actions(
    agent,
    states: np.ndarray,
    residual_alpha: float,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Query deterministic unsquashed policy means with sample_actions(argmax=True), then return
    alpha*tanh(mean_u) for every state.
    """
    out: List[np.ndarray] = []
    for i in range(0, len(states), batch_size):
        batch = states[i : i + batch_size]
        obs_batch = {"imle_preproj": jnp.asarray(batch)}
        
        
        u_mean = agent.sample_actions(obs_batch, argmax=True)
        u_mean_np = np.asarray(jax.device_get(u_mean), dtype=np.float32)
        a_res = float(residual_alpha) * np.tanh(u_mean_np)
        out.append(a_res)
    return np.concatenate(out, axis=0)


# ============================================================================

# ============================================================================

def compute_direction_consistency(a1: np.ndarray, a2: np.ndarray, eps: float = 1e-8) -> float:
    n1 = float(np.linalg.norm(a1))
    n2 = float(np.linalg.norm(a2))
    if n1 < eps and n2 < eps:
        return 1.0
    if n1 < eps or n2 < eps:
        return 0.0
    return float(np.clip(np.dot(a1, a2) / (n1 * n2), -1.0, 1.0))


def compute_scale_consistency(a1: np.ndarray, a2: np.ndarray, eps: float = 1e-8) -> float:
    n1 = float(np.linalg.norm(a1))
    n2 = float(np.linalg.norm(a2))
    if n2 < eps:
        if n1 < eps:
            return 0.0
        return float(np.log(n1 / eps))
    return float(abs(np.log((n1 / n2) + eps)))


def compute_rfc(a_i: np.ndarray, a_j: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    """Paper equations: mean stabilized direction times mean bounded magnitude agreement."""
    u, v = np.asarray(a_i, dtype=np.float64), np.asarray(a_j, dtype=np.float64)
    n_u, n_v = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    dc = float(np.mean(np.sum(u * v, axis=1) / (n_u * n_v + eps)))
    mc = float(np.mean(2 * np.minimum(n_u, n_v) / (n_u + n_v + eps)))
    return {"dc": dc, "mc": mc, "rfc": dc * mc}


def pointwise_probe(
    a_i: np.ndarray, a_j: np.ndarray, states: np.ndarray
) -> List[Dict[str, Any]]:
    assert len(a_i) == len(a_j) == len(states)
    results: List[Dict[str, Any]] = []
    for k in range(len(states)):
        results.append({
            "state_idx": k,
            "a1": a_i[k].copy(),
            "a2": a_j[k].copy(),
            "dir": compute_direction_consistency(a_i[k], a_j[k]),
            "log_scale": compute_scale_consistency(a_i[k], a_j[k]),
            "a1_norm": float(np.linalg.norm(a_i[k])),
            "a2_norm": float(np.linalg.norm(a_j[k])),
        })
    return results


def safety_aggregation(results: List[Dict[str, Any]], delta_dir: float) -> Dict[str, Any]:
    dirs = np.array([r["dir"] for r in results], dtype=np.float64)
    log_scales = np.array([r["log_scale"] for r in results], dtype=np.float64)
    a1_norms = np.array([r["a1_norm"] for r in results], dtype=np.float64)
    a2_norms = np.array([r["a2_norm"] for r in results], dtype=np.float64)
    dangerous_mask = dirs < delta_dir
    p_bad = float(np.mean(dangerous_mask))
    return {
        "num_states": len(results),
        "delta_dir": float(delta_dir),
        "mean_dir": float(dirs.mean()),
        "std_dir": float(dirs.std()),
        "median_dir": float(np.median(dirs)),
        "min_dir": float(dirs.min()),
        "max_dir": float(dirs.max()),
        "mean_log_scale": float(log_scales.mean()),
        "std_log_scale": float(log_scales.std()),
        "median_log_scale": float(np.median(log_scales)),
        "min_log_scale": float(log_scales.min()),
        "max_log_scale": float(log_scales.max()),
        "dangerous_count": int(dangerous_mask.sum()),
        "p_bad": p_bad,
        "dangerous_mean_dir": float(dirs[dangerous_mask].mean()) if dangerous_mask.any() else 0.0,
        "dangerous_mean_log_scale": float(log_scales[dangerous_mask].mean()) if dangerous_mask.any() else 0.0,
        "mean_a1_norm": float(a1_norms.mean()),
        "mean_a2_norm": float(a2_norms.mean()),
        "dirs": dirs,
        "log_scales": log_scales,
        "dangerous_mask": dangerous_mask,
    }


def merge_decision(
    aggregation: Dict[str, Any], thresholds: Dict[str, float], decision_rule: str = "rfc"
) -> Dict[str, Any]:
    mean_dir = aggregation["mean_dir"]
    mean_log_scale = aggregation["mean_log_scale"]
    p_bad = aggregation["p_bad"]
    reasons: List[str] = []
    if not all(np.isfinite(v) for v in [mean_dir, mean_log_scale, p_bad]):
        reasons.append("nonfinite analysis statistics")
    if decision_rule == "rfc":
        if not np.isfinite(aggregation["rfc"]):
            reasons.append("nonfinite RFC")
        elif aggregation["rfc"] < thresholds["tau"]:
            reasons.append(f"RFC={aggregation['rfc']:.4f} < tau={thresholds['tau']}")
    elif mean_dir < thresholds["threshold_dir"]:
        reasons.append(
            f"overall direction misalignment: mean_dir={mean_dir:.4f} "
            f"< {thresholds['threshold_dir']}"
        )
    if decision_rule == "legacy" and mean_log_scale > thresholds["threshold_scale"]:
        reasons.append(
            f"scale mismatch too large: mean_log_scale={mean_log_scale:.4f} "
            f"> {thresholds['threshold_scale']}"
        )
    if decision_rule == "legacy" and p_bad > thresholds["threshold_bad"]:
        reasons.append(
            f"too many dangerous states: {p_bad*100:.2f}% > "
            f"{thresholds['threshold_bad']*100:.2f}%"
        )
    return {
        "can_merge": len(reasons) == 0,
        "decision_rule": decision_rule,
        "dc": aggregation["dc"],
        "mc": aggregation["mc"],
        "rfc": aggregation["rfc"],
        "reasons": reasons,
        "mean_dir": mean_dir,
        "mean_log_scale": mean_log_scale,
        "p_bad": p_bad,
        "thresholds": thresholds,
    }


# ============================================================================

# ============================================================================

def visualize_correction_field(
    aggregation: Dict[str, Any],
    decision: Dict[str, Any],
    output_path: str,
    title_suffix: str = "",
) -> None:
    dirs = aggregation["dirs"]
    log_scales = aggregation["log_scales"]
    dangerous_mask = aggregation["dangerous_mask"]
    safe_mask = ~dangerous_mask

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    ax1.hist(dirs, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax1.axvline(aggregation["mean_dir"], color="red", linestyle="--", linewidth=2,
                label=f"Mean: {aggregation['mean_dir']:.4f}")
    ax1.axvline(aggregation["delta_dir"], color="orange", linestyle="--", linewidth=2,
                label=f"Danger Δ: {aggregation['delta_dir']}")
    ax1.set_xlabel("Direction Consistency (Cosine)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title(f"Direction Consistency Distribution\n{title_suffix}", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.hist(log_scales, bins=50, alpha=0.7, color="coral", edgecolor="black")
    ax2.axvline(aggregation["mean_log_scale"], color="red", linestyle="--", linewidth=2,
                label=f"Mean: {aggregation['mean_log_scale']:.4f}")
    ax2.set_xlabel("|log(||a_i|| / ||a_j||)|", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title(f"Scale Consistency Distribution\n{title_suffix}", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.scatter(dirs[safe_mask], log_scales[safe_mask], alpha=0.5, color="green",
                label="Safe", s=20)
    ax3.scatter(dirs[dangerous_mask], log_scales[dangerous_mask], alpha=0.7, color="red",
                label="Dangerous", s=30)
    ax3.axvline(aggregation["delta_dir"], color="orange", linestyle="--", linewidth=2,
                label=f"Danger Δ")
    ax3.set_xlabel("Direction Consistency", fontsize=12)
    ax3.set_ylabel("Log Scale Consistency", fontsize=12)
    ax3.set_title(f"Direction vs Scale Scatter\n{title_suffix}", fontsize=13)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.axis("off")
    can = "CAN MERGE" if decision["can_merge"] else "CANNOT MERGE"
    txt = [
        f"Decision: {can}",
        f"Rule: {decision['decision_rule']}",
        f"DC={aggregation['dc']:.4f}, MC={aggregation['mc']:.4f}",
        f"RFC={aggregation['rfc']:.4f}, tau={decision['thresholds']['tau']}",
        "",
        "Statistics:",
        f"  N states         = {aggregation['num_states']}",
        f"  mean_dir         = {aggregation['mean_dir']:.4f}",
        f"  median_dir       = {aggregation['median_dir']:.4f}",
        f"  std_dir          = {aggregation['std_dir']:.4f}",
        f"  mean_log_scale   = {aggregation['mean_log_scale']:.4f}",
        f"  median_log_scale = {aggregation['median_log_scale']:.4f}",
        f"  mean ||a_i||     = {aggregation['mean_a1_norm']:.4f}",
        f"  mean ||a_j||     = {aggregation['mean_a2_norm']:.4f}",
        f"  dangerous        = {aggregation['dangerous_count']} / "
        f"{aggregation['num_states']} ({aggregation['p_bad']*100:.2f}%)",
        "",
        "Legacy diagnostic thresholds:",
        f"  dir  >= {decision['thresholds']['threshold_dir']}",
        f"  scale <= {decision['thresholds']['threshold_scale']}",
        f"  p_bad <= {decision['thresholds']['threshold_bad']*100:.1f}%",
    ]
    if decision["reasons"]:
        txt.append("")
        txt.append("Failure reasons:")
        for r in decision["reasons"]:
            txt.append(f"  • {r}")
    ax4.text(
        0.02, 0.98, "\n".join(txt), fontsize=11, family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor=("lightgreen" if decision["can_merge"] else "mistyrose"),
                  alpha=0.7),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] saved: {output_path}")


# ============================================================================
# main
# ============================================================================

def main(argv):
    del argv

    if FLAGS.max_states <= 0 or FLAGS.batch_size <= 0:
        raise SystemExit("max_states and batch_size must be positive")
    if not np.isfinite(FLAGS.residual_alpha) or FLAGS.residual_alpha <= 0:
        raise SystemExit("residual_alpha must be finite and positive")
    if not -1 <= FLAGS.tau <= 1:
        raise SystemExit("tau must be in [-1,1]")
    if not all(np.isfinite(v) for v in [FLAGS.delta_dir, FLAGS.threshold_dir,
                                       FLAGS.threshold_scale, FLAGS.threshold_bad]):
        raise SystemExit("Analysis thresholds must be finite")
    if not FLAGS.task_i_pt or not FLAGS.task_j_pt:
        raise SystemExit("Provide both --task_i_pt and --task_j_pt.")
    if bool(FLAGS.states_npy) == bool(FLAGS.demo_pkl):
        raise SystemExit("Specify exactly one of --states_npy and --demo_pkl.")

    np.random.seed(int(FLAGS.seed))

    src_i = list(FLAGS.source_pts_i) if FLAGS.source_pts_i else None
    src_j = list(FLAGS.source_pts_j) if FLAGS.source_pts_j else None

    
    print("\n=== Building agent i ===")
    agent_i, obs_dim_i, act_dim_i, rank_i = build_agent_from_lora_pt(
        FLAGS.task_i_pt, FLAGS.residual_alpha, FLAGS.seed, source_pt_paths_for_ln=src_i
    )
    print(f"  task_i: obs_dim={obs_dim_i} action_dim={act_dim_i} rank={rank_i}")

    print("\n=== Building agent j ===")
    agent_j, obs_dim_j, act_dim_j, rank_j = build_agent_from_lora_pt(
        FLAGS.task_j_pt, FLAGS.residual_alpha, FLAGS.seed + 1, source_pt_paths_for_ln=src_j
    )
    print(f"  task_j: obs_dim={obs_dim_j} action_dim={act_dim_j} rank={rank_j}")

    if obs_dim_i != obs_dim_j:
        raise SystemExit(
            f"Task observation dimensions differ ({obs_dim_i} vs {obs_dim_j}); "
            "use the same IMLE base checkpoint, observation stack and camera configuration."
        )
    if act_dim_i != act_dim_j:
        raise SystemExit(f"Task action dimensions differ ({act_dim_i} vs {act_dim_j}).")
    if rank_i != rank_j:
        
        print(f"[warn] Ranks differ: {rank_i} vs {rank_j}; comparison is possible, but merging requires compatible ranks.")

    
    print("\n=== Collecting states ===")
    if FLAGS.states_npy:
        states = collect_states_from_npy(FLAGS.states_npy, FLAGS.max_states, FLAGS.seed)
    else:
        states = collect_states_from_pkl(FLAGS.demo_pkl, FLAGS.max_states, FLAGS.seed)

    if states.shape[0] == 0 or not np.isfinite(states).all():
        raise SystemExit("State samples must be nonempty and finite")
    if states.shape[1] != obs_dim_i:
        raise SystemExit(
            f"State dimension {states.shape[1]} does not match agent obs_dim {obs_dim_i}; "
            "ensure the features were collected using the same IMLE checkpoint."
        )

    
    print("\n=== Querying residual actions ===")
    a_i = query_residual_actions(agent_i, states, FLAGS.residual_alpha, FLAGS.batch_size)
    a_j = query_residual_actions(agent_j, states, FLAGS.residual_alpha, FLAGS.batch_size)
    if not np.isfinite(a_i).all() or not np.isfinite(a_j).all():
        raise SystemExit("Actor outputs contain nonfinite residuals")
    print(f"  a_i shape={a_i.shape}, mean ||a_i||={np.linalg.norm(a_i, axis=1).mean():.4f}")
    print(f"  a_j shape={a_j.shape}, mean ||a_j||={np.linalg.norm(a_j, axis=1).mean():.4f}")

    
    print("\n=== Computing pointwise consistency ===")
    results = pointwise_probe(a_i, a_j, states)
    aggregation = safety_aggregation(results, delta_dir=float(FLAGS.delta_dir))
    aggregation.update(compute_rfc(a_i, a_j))

    
    thresholds = {
        "tau": float(FLAGS.tau),
        "threshold_dir": float(FLAGS.threshold_dir),
        "threshold_scale": float(FLAGS.threshold_scale),
        "threshold_bad": float(FLAGS.threshold_bad),
    }
    decision = merge_decision(aggregation, thresholds, FLAGS.decision_rule)

    
    print("\n" + "=" * 60)
    print("Correction Field Analysis Summary")
    print("=" * 60)
    print(f"  task_i : {os.path.basename(FLAGS.task_i_pt)}")
    print(f"  task_j : {os.path.basename(FLAGS.task_j_pt)}")
    print(f"  N      : {aggregation['num_states']}  (action_dim={act_dim_i}, rank_i={rank_i}, rank_j={rank_j})")
    print(f"  DC={aggregation['dc']:.4f}  MC={aggregation['mc']:.4f}  "
          f"RFC={aggregation['rfc']:.4f}  tau={FLAGS.tau}  rule={FLAGS.decision_rule}")
    print(f"  mean_dir          = {aggregation['mean_dir']:.4f}  "
          f"(threshold {thresholds['threshold_dir']})")
    print(f"  mean_log_scale    = {aggregation['mean_log_scale']:.4f}  "
          f"(threshold {thresholds['threshold_scale']})")
    print(f"  p_bad             = {aggregation['p_bad']*100:.2f}%  "
          f"(threshold {thresholds['threshold_bad']*100:.2f}%)")
    print(f"  mean ||a_i||      = {aggregation['mean_a1_norm']:.4f}")
    print(f"  mean ||a_j||      = {aggregation['mean_a2_norm']:.4f}")
    print(f"  >>> can_merge = {decision['can_merge']}")
    if decision["reasons"]:
        for r in decision["reasons"]:
            print(f"  - {r}")
    print("=" * 60)

    
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_i = os.path.basename(FLAGS.task_i_pt).replace(".pt", "")
    tag_j = os.path.basename(FLAGS.task_j_pt).replace(".pt", "")
    out_png = os.path.join(
        FLAGS.output_dir, f"correction_field_{tag_i}__vs__{tag_j}__{ts}.png"
    )
    visualize_correction_field(
        aggregation, decision, out_png,
        title_suffix=f"{tag_i} vs {tag_j}",
    )

    
    out_npz = out_png.replace(".png", ".npz")
    np.savez(
        out_npz,
        dirs=aggregation["dirs"],
        log_scales=aggregation["log_scales"],
        dangerous_mask=aggregation["dangerous_mask"],
        a_i=a_i, a_j=a_j, states=states,
        meta=np.array(
            [obs_dim_i, act_dim_i, rank_i, rank_j, aggregation["num_states"]], dtype=np.int64
        ),
    )
    print(f"[data] raw arrays saved: {out_npz}")
    report = dict(decision, task_i_pt=str(FLAGS.task_i_pt), task_j_pt=str(FLAGS.task_j_pt),
                  num_states=int(len(states)), residual_alpha=float(FLAGS.residual_alpha),
                  delta_dir=float(FLAGS.delta_dir),
                  states_source=str(FLAGS.states_npy or FLAGS.demo_pkl))
    out_json = out_png.replace(".png", ".json")
    with open(out_json, "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(f"[decision] saved: {out_json}")


if __name__ == "__main__":
    app.run(main)
