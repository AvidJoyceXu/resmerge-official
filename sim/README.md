# 🧪 ResMerge in Simulation

[📄 Paper](https://openreview.net/forum?id=wLtzr7PNBI) · [🌐 Project](https://resmerge.github.io/) · [🏠 Main README](../README.md)

Run ResMerge with a frozen **OpenVLA-OFT** base policy on **LIBERO**, using RGB observations and proprioception. This guide follows the RGB experiment recipe.

| Suite | Task IDs, in arrival order | Base checkpoint |
|---|---|---|
| LIBERO Object | **1, 6, 7, 8, 9** | `Haozhan72/Openvla-oft-SFT-libero-object-traj1` |
| LIBERO Spatial | **0, 2, 3, 6, 7** | `Haozhan72/Openvla-oft-SFT-libero-spatial-traj1` |

This simulation release covers OpenVLA-OFT residual training, RFC analysis, merging, refinement and evaluation. The small-model experiments and sequential OpenVLA-OFT SFT baseline training are not included. Download the initialized base checkpoints below before training residuals.

Run all commands from the **repository root**. The commands use the released `*_rgb` configs; checkpoints from the older privileged-state configs are incompatible with this workflow.

<a id="installation"></a>

## 🛠️ Installation

Use a Linux host with NVIDIA GPUs, Docker and the NVIDIA Container Toolkit. Allow about 200 GB of disk space for the image, model assets and outputs. The reference runs used two GPUs for Object task 1 (64 training environments), and one GPU for each other task (32 environments). Reserve at least 25 GB of free GPU memory per GPU; peak usage depends on the rollout batch size.

From your checkout, start a container with the repository and model cache mounted at paths derived from the current directory:

```bash
mkdir -p hf logs results

docker run --rm -it --gpus all --shm-size 64g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e HF_HOME="$PWD/hf" \
  -v "$PWD:$PWD" -w "$PWD" \
  rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 \
  /bin/bash
```

`--gpus all` exposes all host GPUs. To select devices, replace it with, for example, `--gpus '"device=0,1"'` for Object task 1 or `--gpus '"device=2"'` for a single-GPU run. The configs use all GPUs visible inside their container. Run one experiment at a time per container; use separate containers for concurrent experiments.

Inside **each new container shell**, activate the environment and prepare the assets:

```bash
source switch_env openvla-oft
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export EMBODIED_PATH="$PWD/examples/embodiment"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python -m pip install -r requirements/resmerge-analysis.txt

export RESMERGE_OBJECT_BASE="$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Haozhan72/Openvla-oft-SFT-libero-object-traj1"))')"
export RESMERGE_SPATIAL_BASE="$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Haozhan72/Openvla-oft-SFT-libero-spatial-traj1"))')"
export RESMERGE_VISUAL_ENCODER="$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("facebook/dinov2-small"))')"
```

The image supplies OpenVLA-OFT, LIBERO and their runtime dependencies. Downloads are reused from `HF_HOME`. The configs read the exported checkpoint paths, so no machine-specific paths need to be edited. TensorBoard logging is enabled by default.

Check the active environment before launching a run:

```bash
python - <<'PY'
import torch
import transformers
import rlinf
from libero.libero import get_libero_path
from rlinf.envs.venv import SubprocVectorEnv
from rlinf.workers.rollout.hf.residual_rollout_worker import ResidualRolloutWorker
print("torch:", torch.__version__, "transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("RLinf:", rlinf.__file__)
print("LIBERO assets:", get_libero_path("assets"))
assert torch.cuda.is_available()
PY
```

The image's OpenVLA-OFT environment uses PyTorch `2.6.0+cu124` and Transformers `4.40.1`. The printed RLinf path should point into this checkout.

<a id="train"></a>

## 🧠 Train Task Experts

Train a single task:

```bash
bash examples/embodiment/run_embodiment.sh \
  libero_object_task6_lora_residual_sac_openvlaoft_rgb
```

For all ten experts, run the following sequentially in a container with enough GPU memory for Object task 1:

```bash
for suite in object spatial; do
  if [ "$suite" = object ]; then
    tasks=(1 6 7 8 9)
  else
    tasks=(0 2 3 6 7)
  fi
  for task in "${tasks[@]}"; do
    bash examples/embodiment/run_embodiment.sh \
      "libero_${suite}_task${task}_lora_residual_sac_openvlaoft_rgb" \
      "runner.logger.log_path=logs/rgb/${suite}_task${task}" || break 2
  done
done
```

The RGB configs train for **1,000 steps** and save every 500 steps. Use **`global_step_1000`** for the following experiments. With the loop above, checkpoints are saved at:

```text
logs/rgb/<suite>_task<task>/libero_<suite>_task<task>_single_trial_lora_openvlaoft_rgb/checkpoints/global_step_1000/actor/huggingface_model/
```

Extra Hydra overrides can be appended to either launcher. Each task keeps its own checkpoint; use the same base checkpoint and RGB encoder for all experts within a suite.

<a id="rfc"></a>

## 🔎 Compute RFC

Choose a pair from one suite. This example compares Object tasks 6 and 8; set `SUITE=spatial`, `TASK_I=6`, `TASK_J=7` for the Spatial pair used in the refinement example.

```bash
export SUITE=object
export TASK_I=6
export TASK_J=8
export TAU=0.20

checkpoint_for() {
  printf 'logs/rgb/%s_task%s/libero_%s_task%s_single_trial_lora_openvlaoft_rgb/checkpoints/global_step_1000/actor/huggingface_model' \
    "$SUITE" "$1" "$SUITE" "$1"
}
export CKPT_I="$(checkpoint_for "$TASK_I")"
export CKPT_J="$(checkpoint_for "$TASK_J")"
export MERGED="merged_policy_dir/${SUITE}_${TASK_I}_${TASK_J}_rgb"

if [ "$SUITE" = object ]; then
  RFC_CONFIG=configs/eval_lora_config_object.yaml
else
  RFC_CONFIG=configs/eval_lora_config.yaml
fi

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
python examples/embodiment/eval/correction_field_analysis/analyze_correction_vector_field.py \
  --config "$RFC_CONFIG" \
  --task_i "$TASK_I" --task_j "$TASK_J" \
  --checkpoint_i "$CKPT_I" --checkpoint_j "$CKPT_J" \
  --state_method base_rollout --max_demos 10 --num_probe_states 2000 \
  --obs_mode rgb --visual_encoder_path "$RESMERGE_VISUAL_ENCODER" \
  --tau "$TAU"
```

If you trained with the single-task command, set `CKPT_I` and `CKPT_J` to the corresponding directories under `logs/<timestamp>-<config>/` instead. RFC uses **2,000 probe states per task** from frozen-base rollouts, sampled at every simulator step. Ten episodes provide up to 2,400 states per task; increase `--max_demos` if the output reports fewer than 2,000. Check the reported per-task counts before comparing pairs.

Reports and plots are saved under `results/correction_field_analysis_task*/`. Use the reported **RFC = DC × MC** and `can_merge` decision. `TAU=0.20` is an example setting from the RGB threshold sweep; keep your selected threshold fixed throughout a sequence. To repeat the sweep, use `0.25 0.20 0.16 0.10 0.05 0.0`.

For continual adaptation, process the task order in the table above. Compare each new expert with every current expert, merge with the highest-RFC expert if it passes the threshold, and otherwise keep a separate expert. When comparing against an already merged expert, pass every covered task through `--probe_tasks`; for example, use `--probe_tasks 1 6 8` for task 8 versus an expert serving tasks 1 and 6.

<a id="merge"></a>

## 🔀 Merge

After the pair passes the RFC gate, merge its checkpoints:

```bash
python toolkits/merge_lora_policies/quick_merge.py \
  --checkpoint_paths "$CKPT_I" "$CKPT_J" \
  --output_path "$MERGED" --prune_ratio 0.2 --restore_norm \
  --obs_dim 846 --action_dim 7 --num_action_chunks 8 --actor_input obs
```

The dimensions match the released RGB configs for both suites. Save to a directory to produce safetensors for evaluation and refinement. For an incremental merge, replace one input with the selected existing expert checkpoint and merge it with the new task expert. The [merge toolkit](../toolkits/merge_lora_policies/README.md) also supports explicit weights.

<a id="refine"></a>

## ✨ Refine After Merging

Continue residual RL on all tasks served by the merged expert. For the **Spatial 6/7** example, first run the RFC and merge commands above with `SUITE=spatial`, `TASK_I=6`, and `TASK_J=7`, then run:

```bash
bash examples/embodiment/run_embodiment.sh \
  libero_spatial_task6_lora_residual_sac_openvlaoft_rgb \
  actor.model.model_path="$MERGED" \
  'env.train.specific_reset_id=[6,7]' \
  'env.eval.specific_reset_id=[6,7]' \
  runner.max_epochs=300 runner.save_interval=100 \
  runner.logger.log_path=logs/refine_spatial_6_7 \
  runner.logger.experiment_name=refine_spatial_6_7

export REFINED=logs/refine_spatial_6_7/refine_spatial_6_7/checkpoints/global_step_300/actor/huggingface_model
```

This starts a new run from the merged actor with a fresh optimizer and replay buffer. For another expert, use its suite's RGB config and the full list of tasks it serves. Evaluate the saved refined checkpoint separately from the merge-only checkpoint.

<a id="evaluate"></a>

## 📊 Evaluate

Evaluate an expert on every released task in its suite. Set `POLICY` to an individual, merged or refined checkpoint directory:

```bash
export POLICY="$MERGED"
# After the Spatial refinement example: export POLICY="$REFINED"

if [ "$SUITE" = object ]; then
  tasks=(1 6 7 8 9)
else
  tasks=(0 2 3 6 7)
fi
for task in "${tasks[@]}"; do
  bash examples/embodiment/eval_single_task.sh \
    "libero_${SUITE}_task${task}_lora_residual_sac_openvlaoft_rgb" \
    "$POLICY" "$task" || break
done
```

Results are saved under `results/single_eval_<timestamp>/`; consult `eval.log` for **`eval/success_once`**, with videos under the run's `video/eval/` directory. Repeat the loop for each expert checkpoint to collect a cross-task success matrix. For a continual sequence, evaluate the current expert bank after each stage.

To evaluate the frozen base policy, use any matching RGB residual checkpoint and disable its correction:

```bash
bash examples/embodiment/eval_single_task.sh \
  "libero_${SUITE}_task${TASK_I}_lora_residual_sac_openvlaoft_rgb" \
  "$CKPT_I" "$TASK_I" residual_policy.res_scale=0
```

The reference evaluation uses 32 environments, `max_trials_per_task=1`, and a fixed reset state. These are repeated episodes from **one initial state per task**. To evaluate across more initial states, append `env.eval.max_trials_per_task=50`; report that protocol separately from the reference setting.
