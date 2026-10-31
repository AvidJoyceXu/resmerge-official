#!/usr/bin/env bash
# Modified by the ResMerge contributors, 2026-10-31; see real_robot/THIRD_PARTY.md.
set -euo pipefail
MODEL=imle
DATASET_ZARR="${DATASET_ZARR:-}"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) [[ $# -ge 2 ]] || { echo "--model requires a value" >&2; exit 1; }; MODEL="$2"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    --dataset) [[ $# -ge 2 ]] || { echo "--dataset requires a path" >&2; exit 1; }; DATASET_ZARR="$2"; shift 2 ;;
    --dataset=*) DATASET_ZARR="${1#*=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ "$MODEL" == imle ]] || { echo "This example supports IMLE only" >&2; exit 1; }
: "${DATASET_ZARR:?Set --dataset or DATASET_ZARR}"
# Resolve relative data paths before switching to the training source directory.
[[ -d "$DATASET_ZARR" ]] || { echo "Dataset directory not found: $DATASET_ZARR" >&2; exit 1; }
DATASET_ZARR="$(cd "$DATASET_ZARR" && pwd -P)"
export DATASET_ZARR
cd "$(dirname "$0")"
exec python train.py --config-name=psi_policy_workspace "task.dataset_name=$(basename "$DATASET_ZARR" .zarr)" "${ARGS[@]}"
