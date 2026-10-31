#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
MODEL=imle
DATASET_ZARR="${DATASET_ZARR:-}"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    --dataset) DATASET_ZARR="$2"; shift 2 ;;
    --dataset=*) DATASET_ZARR="${1#*=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ "$MODEL" == imle ]] || { echo "This example supports IMLE only" >&2; exit 1; }
: "${DATASET_ZARR:?Set --dataset or DATASET_ZARR}"
export DATASET_ZARR
exec python train.py --config-name=psi_policy_workspace "task.dataset_name=$(basename "$DATASET_ZARR" .zarr)" "${ARGS[@]}"
