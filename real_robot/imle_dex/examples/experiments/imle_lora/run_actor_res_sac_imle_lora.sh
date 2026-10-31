#!/usr/bin/env bash

set -euo pipefail
IMLE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPT="$(dirname "$0")/async_res_sac_imle_lora.py"
# Match the learner checkpoint, rank, observation stack and action timing.
LORA_RANK="${LORA_RANK:-16}"

EVAL_BASE_ONLY="${EVAL_BASE_ONLY:-0}"
EVAL_MERGED_PT="${EVAL_MERGED_PT:-}"
EVAL_SOURCE_PTS="${EVAL_SOURCE_PTS:-}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"

# Default HIL=1 enables SpaceMouse intervention. Set HIL=0 for keyboard-only or evaluation.
HIL="${HIL:-1}"

: "${CKPT_PATH:?Set CKPT_PATH to the IMLE checkpoint}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the RL output directory}"
LEARNER_IP="${LEARNER_IP:-localhost}"

export http_proxy https_proxy
if [ -z "${NO_PROXY:-}" ]; then
  export NO_PROXY="localhost,127.0.0.1,::1,192.168.0.0/16,10.0.0.0/8,172.16.0.0/12"
else
  export NO_PROXY="${NO_PROXY},localhost,127.0.0.1,::1"
fi
[ -n "${NO_PROXY_EXTRA:-}" ] && export NO_PROXY="${NO_PROXY},${NO_PROXY_EXTRA}"

MODE_ARGS=()
if [ "${HIL}" = "1" ]; then
  if [ -n "${EVAL_MERGED_PT}" ] || [ "${EVAL_BASE_ONLY}" = "1" ]; then
    echo "[run_actor] HIL=1 and EVAL_* are mutually exclusive." >&2
    exit 2
  fi
  MODE_ARGS+=(--hil)
  echo "[run_actor] HIL mode: SpaceMouse intervention; space=success, r=start; executed actions enter replay."
elif [ "${EVAL_BASE_ONLY}" = "1" ]; then
  if [ -n "${EVAL_MERGED_PT}" ]; then
    echo "[run_actor] EVAL_BASE_ONLY=1 and EVAL_MERGED_PT are mutually exclusive." >&2
    exit 2
  fi
  MODE_ARGS+=(--eval_only --eval_imle_base_only --eval_episodes="${EVAL_EPISODES}")
  echo "[run_actor] eval-only IMLE base: ckpt=${CKPT_PATH} episodes=${EVAL_EPISODES}"
  echo "[run_actor] Base-only actions without LoRA residual; keys: r/space/f/t"
elif [ -n "${EVAL_MERGED_PT}" ]; then
  MODE_ARGS+=(--eval_only --eval_merged_pt="${EVAL_MERGED_PT}" --eval_episodes="${EVAL_EPISODES}")
  if [ -n "${EVAL_SOURCE_PTS}" ]; then
    MODE_ARGS+=(--eval_source_pts="${EVAL_SOURCE_PTS}")
  fi
  echo "[run_actor] eval-only: merged_pt=${EVAL_MERGED_PT} episodes=${EVAL_EPISODES}"
  echo "[run_actor] Evaluation: always base+residual without exploration; keys: r/space/f/t"
else
  MODE_ARGS+=(--rl_actor_human --nouse_spacemouse)
fi

exec python3 "$SCRIPT" \
  --actor \
  --nofake_env \
  --ip="${LEARNER_IP}" \
  --exp_name=pick_place \
  --imle_root="${IMLE_ROOT}" \
  --ckpt_path="${CKPT_PATH}" \
  --stack_obs_num=2 \
  --latency_step=0 \
  --chunk_exec_steps=4 \
  --seed=0 \
  --target_entropy=-2 \
  --alpha_max=0.1 \
  --residual_alpha=0.05 \
  --batch_size=256 \
  --critic_actor_ratio=4 \
  --max_steps=50000 \
  --replay_buffer_capacity=1000000 \
  --random_steps=300 \
  --training_starts=300 \
  --steps_per_update=30 \
  --log_period=10 \
  --checkpoint_period=5000 \
  --checkpoint_path="${CHECKPOINT_PATH}" \
  --nodebug \
  --sac_obs_mode=imle_preproj \
  --replay_omit_pixels \
  --lora_rank="${LORA_RANK}" \
  --hydra_override=n_action_steps=16 \
  "${MODE_ARGS[@]}" \
  "$@"
