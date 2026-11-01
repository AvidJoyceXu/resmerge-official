#! /bin/bash
# Modified for the ResMerge source release: default to the released RGB Object task-6 recipe.
set -euo pipefail

# NOTE: fix mujoco egl error
export NVIDIA_DRIVER_CAPABILITIES="all"

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "${1:-}" ]; then
    CONFIG_NAME="libero_object_task6_lora_residual_sac_openvlaoft_rgb"
else
    CONFIG_NAME=$1
fi
if [ "$#" -gt 0 ]; then
    shift
fi

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD=(python "$SRC_FILE" --config-path "${EMBODIED_PATH}/config/"
    --config-name "$CONFIG_NAME" "runner.logger.log_path=${LOG_DIR}" "$@")
printf '%q ' "${CMD[@]}" > "$MEGA_LOG_FILE"
printf '\n' >> "$MEGA_LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$MEGA_LOG_FILE"
