# ResMerge

**Residual Policy Learning and Merging for Continual Adaptation of Pre-trained Robot Policies**

[Project page](https://resmerge.github.io/)

ResMerge learns lightweight residual policies on top of a frozen robot policy and uses residual-field compatibility (RFC) to guide policy merging. Compatible residuals can be merged and further refined while keeping the base policy frozen.

## Release scope

This release contains the **OpenVLA-OFT simulation implementation** built on RLinf, with LIBERO Object and Spatial configurations:

- Residual policy training with SAC.
- Pairwise correction-field analysis and RFC computation.
- Residual checkpoint merging and post-merge refinement.
- Evaluation on individual tasks, including cross-task checks.

The release covers part of the paper's implementation. Privileged-state residuals are the paper's simulation setting; RGB residuals are an additional implementation. It does not provide ResMerge ViT-T or real-robot experiment recipes, or a complete set of checkpoints and artifacts for every paper result. The complete inherited RLinf runtime is retained. The commands below describe the available workflow; they have not been rerun as part of preparing this release.

## Environment

Use a Linux machine with NVIDIA GPUs, Docker, and the NVIDIA Container Toolkit. The provided configurations load a 7B base policy; memory requirements depend on the environment batch size and worker placement.

From the repository root, set a persistent model-cache directory and start the existing RLinf image:

```bash
export RESMERGE_DIR="$PWD"
export RESMERGE_HF=/absolute/path/to/huggingface-cache
mkdir -p "$RESMERGE_HF"

docker run -it --gpus all --shm-size 64g \
  --name resmerge \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e HF_HOME=/workspace/hf \
  -v "$RESMERGE_DIR:/workspace/RLinf" \
  -v "$RESMERGE_HF:/workspace/hf" \
  rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 \
  /bin/bash
```

Inside the container:

```bash
cd /workspace/RLinf
export PATH=/opt/venv/openvla-oft/bin:$PATH
export PYTHONPATH=/workspace/RLinf
export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export WANDB_MODE=offline

# Extra packages imported by the pairwise RFC analysis script.
python -m pip install -r requirements/resmerge-analysis.txt
```

Use the `openvla-oft` environment explicitly. For concurrent experiments, use a separate container for each run. Set GPU visibility through Docker or edit `cluster.component_placement`; the default `all` placement uses all GPUs visible to the container. Keep batch sizes consistent with the selected worker count.

## Base models and configuration

Download the base models used by the existing configurations:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

for model_id in [
    "Haozhan72/Openvla-oft-SFT-libero-object-traj1",
    "Haozhan72/Openvla-oft-SFT-libero-spatial-traj1",
]:
    print(model_id, snapshot_download(model_id))
PY
```

Set the printed snapshot paths for the suites you will use. These paths must exist inside the container:

```bash
export RESMERGE_OBJECT_BASE=/absolute/path/printed/for/object
export RESMERGE_SPATIAL_BASE=/absolute/path/printed/for/spatial
```

The released task and RFC configurations read the corresponding variable through `residual_policy.base_model_path`; `actor.base_model.model_path` refers to the same value. Set residual initialization and evaluation checkpoint paths separately. No residual checkpoint is loaded by default.

Two observation modes are available:

| Mode | LIBERO Object | LIBERO Spatial | Additional asset |
| --- | --- | --- | --- |
| Privileged state | 88 dimensions | 74 dimensions | None |
| RGB + proprioception | 846 dimensions | 846 dimensions | `facebook/dinov2-small` |

RGB configurations have the suffix `_rgb`. Their visual encoder defaults to the Hugging Face identifier `facebook/dinov2-small`; set `RESMERGE_VISUAL_ENCODER` to a downloaded local snapshot when needed. Use a matching observation mode, architecture, and base model throughout training, RFC analysis, merging, and evaluation. Privileged Object and Spatial residuals have different input dimensions and cannot be merged directly.

Included task configurations are Object tasks 0–9 in privileged mode, Object tasks 1/6/7/8/9 in RGB mode, and Spatial tasks 0/2/3/6/7 in both modes. Use the exact files under `examples/embodiment/config/`; configurations for other tasks are not supplied.

Base-model assets are distributed by their respective providers. A complete downloadable ResMerge residual-checkpoint collection is not bundled with this source release; the workflow below can use residuals trained locally.

## 1. Train task residuals

For example, train two LIBERO Object experts:

```bash
bash examples/embodiment/run_embodiment.sh \
  libero_object_task6_lora_residual_sac_openvlaoft

bash examples/embodiment/run_embodiment.sh \
  libero_object_task8_lora_residual_sac_openvlaoft
```

Run these sequentially, or use separate containers. Edit the selected YAML files first to set the training budget, base-model paths, worker placement, and logging. The shell wrapper accepts a configuration name and does not forward additional Hydra overrides.

Residual checkpoints are written under:

```text
logs/<run>/<experiment>/checkpoints/global_step_<step>/actor/huggingface_model/
```

For the following examples, replace the paths with the actual checkpoint directories:

```bash
export CKPT6=/absolute/path/to/task6/actor/huggingface_model
export CKPT8=/absolute/path/to/task8/actor/huggingface_model
export MERGED=/workspace/RLinf/merged_policy_dir/object_6_8
```

Paths must be accessible inside the container. The residual checkpoint is used together with its frozen base policy.

## 2. Compute RFC

With `RESMERGE_OBJECT_BASE` set, analyze a pair:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

python examples/embodiment/eval/correction_field_analysis/analyze_correction_vector_field.py \
  --config configs/eval_lora_config_object.yaml \
  --task_i 6 --task_j 8 \
  --checkpoint_i "$CKPT6" --checkpoint_j "$CKPT8" \
  --state_method base_rollout --max_demos 5 \
  --obs_mode privileged --tau 0.5
```

`--config` is resolved relative to the analysis script directory. Spatial analysis uses `configs/eval_lora_config.yaml` and `RESMERGE_SPATIAL_BASE`. `base_rollout` collects probe states using the frozen base policy and does not require demonstration HDF5 files. For RGB checkpoints, use `--obs_mode rgb` and `--visual_encoder_path /absolute/path/to/dinov2-small`. Use the same probe-state protocol when comparing pairs; `--max_demos 5` alone does not fix the number of probe states.

RFC combines correction direction and magnitude consistency. The example uses the analysis CLI's default threshold, `0.5`; it is not a claim about every paper experiment. Use the resulting score and your selected threshold to decide which pair to merge. The merge command below does not compute RFC or enforce the compatibility threshold automatically.

## 3. Merge compatible residuals

```bash
python toolkits/merge_lora_policies/quick_merge.py \
  --checkpoint_paths "$CKPT6" "$CKPT8" \
  --output_path "$MERGED" \
  --restore_norm
```

An output directory produces the safetensors checkpoint format used by the evaluation pipeline. For incremental merging, merge the current expert with the next selected residual, then recompute compatibility when needed.

## 4. Refine a merged expert

Initialize a new training run from the merged residual and train on the tasks it covers. This example uses a 600-epoch budget; choose the budget for your experiment.

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python examples/embodiment/train_embodied_agent.py \
  --config-path "$EMBODIED_PATH/config/" \
  --config-name libero_object_task6_lora_residual_sac_openvlaoft \
  actor.model.model_path="$MERGED" \
  'env.train.specific_reset_id=[6,8]' \
  runner.max_epochs=600 \
  runner.logger.log_path=/workspace/RLinf/logs/refine_object_6_8
```

This initializes the actor weights for a new run rather than resuming an old optimizer and replay buffer. Evaluate the resulting refined checkpoint separately from the merge-only checkpoint.

## 5. Evaluate

Evaluate the merged residual on each covered task:

```bash
bash examples/embodiment/eval_single_task.sh \
  libero_object_task6_lora_residual_sac_openvlaoft "$MERGED" 6

bash examples/embodiment/eval_single_task.sh \
  libero_object_task8_lora_residual_sac_openvlaoft "$MERGED" 8
```

Use a trained single-task checkpoint or the refined checkpoint in place of `$MERGED` to evaluate those policies. Outputs are saved under `results/single_eval_<timestamp>/`.

To check transfer, evaluate a trained residual on another task in the same suite and observation mode by changing the final task ID, for example:

```bash
bash examples/embodiment/eval_single_task.sh \
  libero_object_task6_lora_residual_sac_openvlaoft "$CKPT6" 8
```

**Evaluation protocol:** existing configurations use one fixed initial state per task with multiple rollout environments and stochastic action sampling. These episodes are not distinct initial states. Report the reset-state coverage, number of episodes, decoding settings, and seeds with any success rate.

The evaluation wrapper also does not forward extra overrides. To change initial-state coverage, call the Python entry point directly, for example:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

python examples/embodiment/eval_embodied_agent.py \
  --config-path "$EMBODIED_PATH/config/" \
  --config-name libero_object_task6_lora_residual_sac_openvlaoft \
  runner.eval_policy_path="$MERGED" \
  runner.only_eval=True \
  runner.logger.log_path=/workspace/RLinf/results/object6_multi_reset \
  env.eval.specific_reset_id=6 \
  env.eval.max_trials_per_task=50
```

This changes the configured reset-state pool. Check actual completed episodes and state coverage in the outputs before reporting a benchmark result.

## Code map

| Component | Location |
| --- | --- |
| Residual policy models and weight loading | `rlinf/models/embodiment/residual_policy/` |
| Task configurations | `examples/embodiment/config/` |
| Residual rollout worker | `rlinf/workers/rollout/hf/residual_rollout_worker.py` |
| Residual SAC training worker | `rlinf/workers/actor/residual_fsdp_sac_policy_worker.py` |
| Correction-field analysis | `examples/embodiment/eval/correction_field_analysis/` |
| Merge utilities | `toolkits/merge_lora_policies/` |

## License and acknowledgments

The repository is distributed under the Apache License 2.0; see [LICENSE](LICENSE). The original RLinf license is preserved in [LICENSE-RLinf](LICENSE-RLinf), and [UPSTREAM.md](UPSTREAM.md) records the source commit and packaging changes. Preserve upstream copyright notices and license files when redistributing code. Third-party datasets, models, and dependencies remain subject to their own licenses.

This implementation builds on RLinf, OpenVLA-OFT, and LIBERO. We thank their authors and contributors.
