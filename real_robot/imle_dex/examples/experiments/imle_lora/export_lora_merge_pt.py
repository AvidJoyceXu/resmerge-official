#!/usr/bin/env python3
# Modified by the ResMerge contributors, 2026-10-31; see real_robot/THIRD_PARTY.md.
"""
Export an existing Flax actor checkpoint as a PyTorch .pt file for EVAL_MERGED_PT. Supply
run_dir, ckpt_path and matching lora_rank. checkpoint_step=0 selects the latest checkpoint;
output naming is automatic unless specified.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

_IMLE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for sub in ("serl_launcher", "serl_robot_infra", "examples"):
    p = os.path.join(_IMLE_ROOT, sub)
    if os.path.isdir(p):
        sys.path.insert(0, p)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

from types import SimpleNamespace
from flax.training import checkpoints

import lora_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Run directory containing checkpoint_*")
    ap.add_argument("--checkpoint_step", type=int, default=0, help="0 selects the latest checkpoint")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--ckpt_path", default="", help="Optional base checkpoint reference stored in export metadata")
    ap.add_argument("--exp_name", default="pick_place")
    ap.add_argument("--residual_alpha", type=float, default=0.05)
    ap.add_argument("--out", default="", help="Output .pt path; defaults to an automatically named file in run_dir")
    args = ap.parse_args()
    if not math.isfinite(args.residual_alpha) or args.residual_alpha <= 0:
        raise SystemExit("residual_alpha must be finite and positive")

    run_dir = os.path.abspath(args.run_dir)
    step = int(args.checkpoint_step)
    if step <= 0:
        latest = checkpoints.latest_checkpoint(run_dir)
        if latest is None:
            raise SystemExit(f"No checkpoint found: {run_dir}")
        step = int(os.path.basename(latest).split("_", 1)[1])

    # Read saved tensors directly: dimensions and optimizer layout belong to the run.
    state = checkpoints.restore_checkpoint(run_dir, target=None, step=step)
    if not isinstance(state, dict) or "params" not in state:
        raise ValueError("Expected an RL state checkpoint containing params")
    params = state["params"]
    actor = lora_agent._actor_policy_params(params)
    encoder = lora_agent._params_subtree(actor, "encoder")
    saved_rank = int(encoder["enc_lora_A"]["kernel"].shape[1])
    if saved_rank != args.lora_rank:
        raise ValueError(f"Checkpoint rank is {saved_rank}, but --lora_rank={args.lora_rank}")
    # save_for_merge only consumes state.params and config; no optimizer is rebuilt.
    agent = SimpleNamespace(state=SimpleNamespace(params=params),
                            config={"lora_rank": saved_rank})

    out = args.out or os.path.join(
        run_dir, f"task_{args.exp_name}_step{step}_for_lora_merge.pt"
    )
    lora_agent.save_for_merge(
        agent,
        out,
        task_id=str(args.exp_name),
        additional_info={"global_step": step, "lora_rank": saved_rank,
                         "base_checkpoint": args.ckpt_path,
                         "residual_alpha": float(args.residual_alpha)},
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
