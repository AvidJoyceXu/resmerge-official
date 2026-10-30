# ResMerge

**Residual Policy Learning and Merging for Continual Adaptation of Pre-trained Robot Policies**

[Paper](https://openreview.net/forum?id=wLtzr7PNBI) · [Project Website](https://resmerge.github.io/)

ResMerge learns lightweight residual policies on top of a frozen robot policy and merges compatible experts using Residual-Field Consistency (RFC).

## Quick Start

The example below trains and merges two residual policies on LIBERO Object tasks 6 and 8 with OpenVLA-OFT.

### 1. Set up the environment

Use a Linux machine with NVIDIA GPUs, Docker, and the NVIDIA Container Toolkit.

```bash
git clone https://github.com/ResMerge/resmerge-official.git
cd resmerge-official
mkdir -p "$HOME/.cache/huggingface"

docker run -it --gpus all --shm-size 64g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e HF_HOME=/workspace/hf \
  -v "$PWD:/workspace/RLinf" \
  -v "$HOME/.cache/huggingface:/workspace/hf" \
  rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 \
  /bin/bash
```

Run the remaining commands inside the container:

```bash
cd /workspace/RLinf
export PATH=/opt/venv/openvla-oft/bin:$PATH
export PYTHONPATH=/workspace/RLinf
export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export WANDB_MODE=offline
python -m pip install -r requirements/resmerge-analysis.txt

export RESMERGE_OBJECT_BASE="$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Haozhan72/Openvla-oft-SFT-libero-object-traj1"))')"
```

### 2. Train residual policies

Run the two training jobs sequentially. Adjust GPU placement, batch sizes, and the training budget in the corresponding [task configurations](examples/embodiment/config/) for your machine.

```bash
bash examples/embodiment/run_embodiment.sh \
  libero_object_task6_lora_residual_sac_openvlaoft

bash examples/embodiment/run_embodiment.sh \
  libero_object_task8_lora_residual_sac_openvlaoft
```

Checkpoints are saved to `logs/<run>/<experiment>/checkpoints/global_step_<step>/actor/huggingface_model/`. Set the paths to the checkpoints you want to merge:

```bash
export CKPT6=/absolute/path/to/task6/actor/huggingface_model
export CKPT8=/absolute/path/to/task8/actor/huggingface_model
export MERGED=/workspace/RLinf/merged_policy_dir/object_6_8
```

### 3. Compute RFC and merge

Compute pairwise RFC using probe states collected by the frozen base policy:

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

If the pair meets your RFC threshold, merge the residual checkpoints:

```bash
python toolkits/merge_lora_policies/quick_merge.py \
  --checkpoint_paths "$CKPT6" "$CKPT8" \
  --output_path "$MERGED" --restore_norm
```

### 4. Refine the merged policy

Continue training the merged residual on both tasks. This example uses 600 epochs and saves a checkpoint every 100 epochs:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python examples/embodiment/train_embodied_agent.py \
  --config-path "$EMBODIED_PATH/config/" \
  --config-name libero_object_task6_lora_residual_sac_openvlaoft \
  actor.model.model_path="$MERGED" \
  'env.train.specific_reset_id=[6,8]' \
  runner.max_epochs=600 \
  runner.save_interval=100 \
  runner.logger.log_path=/workspace/RLinf/logs/refine_object_6_8
```

### 5. Evaluate

Choose a trained, merged, or refined checkpoint directory. The example below evaluates the merged policy on both tasks:

```bash
export POLICY="$MERGED"

bash examples/embodiment/eval_single_task.sh \
  libero_object_task6_lora_residual_sac_openvlaoft "$POLICY" 6

bash examples/embodiment/eval_single_task.sh \
  libero_object_task8_lora_residual_sac_openvlaoft "$POLICY" 8
```

Evaluation outputs are saved to `results/single_eval_<timestamp>/`. To evaluate a refined policy, set `POLICY` to its saved `actor/huggingface_model/` directory.

---

Built on [RLinf](https://github.com/RLinf/RLinf), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). Licensed under [Apache 2.0](LICENSE).
