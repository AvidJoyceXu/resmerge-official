#!/usr/bin/env bash

set -euo pipefail
IMLE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPT="$(dirname "$0")/async_res_sac_imle_lora.py"
LORA_RANK="${LORA_RANK:-16}"

# Resume from a run containing checkpoint_* and replay_buffer_latest.pkl.
RESUME_RUN="${RESUME_RUN:-}"
RESUME_ARGS=()
if [ -n "${RESUME_RUN}" ]; then
  RESUME_ARGS=(--resume --checkpoint_path="${RESUME_RUN}")
fi

: "${CKPT_PATH:?Set CKPT_PATH to the IMLE checkpoint}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the RL output directory}"

export http_proxy https_proxy
if [ -z "${NO_PROXY:-}" ]; then
  export NO_PROXY="localhost,127.0.0.1,::1,192.168.0.0/16,10.0.0.0/8,172.16.0.0/12"
else
  export NO_PROXY="${NO_PROXY},localhost,127.0.0.1,::1"
fi
[ -n "${NO_PROXY_EXTRA:-}" ] && export NO_PROXY="${NO_PROXY},${NO_PROXY_EXTRA}"

exec python3 "$SCRIPT" \
  --learner \
  --ip=localhost \
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
  --max_steps=1000000 \
  --replay_buffer_capacity=100000 \
  --random_steps=300 \
  --training_starts=500 \
  --steps_per_update=30 \
  --log_period=10 \
  --checkpoint_period=5000 \
  --checkpoint_path="${RESUME_RUN:-${CHECKPOINT_PATH}}" \
  "${RESUME_ARGS[@]}" \
  --sac_obs_mode=imle_preproj \
  --replay_omit_pixels \
  --lora_rank="${LORA_RANK}" \
  --hydra_override=n_action_steps=16 \
  "$@"
