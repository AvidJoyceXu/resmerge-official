# 🧪 ResMerge in Simulation

[📄 Paper](https://openreview.net/forum?id=wLtzr7PNBI) · [🌐 Project](https://resmerge.github.io/) · [🏠 Main README](../README.md) · [🦾 Real-Robot Guide](../real_robot/README.md)

Train and merge residual experts on **LIBERO Object** tasks 6 and 8 with a frozen **OpenVLA-OFT** base policy.

**Workflow:** Install → Train task experts → Compute RFC → Merge → Refine → Evaluate.

Run the commands below from the **repository root**, not `sim/`. The simulation implementation lives in [`examples/embodiment/`](../examples/embodiment/) and [`rlinf/`](../rlinf/).

```bash
git clone https://github.com/zhucoffee/resmerge-official.git
cd resmerge-official
```

<a id="installation"></a>

## 🛠️ Installation

Requires Linux, NVIDIA GPUs, Docker, and the NVIDIA Container Toolkit.

```bash
mkdir -p "$HOME/.cache/huggingface"

docker run -it --gpus all --shm-size 64g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e HF_HOME=/workspace/hf \
  -v "$PWD:/workspace/RLinf" \
  -v "$HOME/.cache/huggingface:/workspace/hf" \
  rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 \
  /bin/bash
```

Inside the container, activate the OpenVLA-OFT environment and download the shared base policy:

```bash
cd /workspace/RLinf
export PATH=/opt/venv/openvla-oft/bin:$PATH
export PYTHONPATH=/workspace/RLinf
export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export WANDB_MODE=offline
python -m pip install -r requirements/resmerge-analysis.txt

export RESMERGE_OBJECT_BASE="$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Haozhan72/Openvla-oft-SFT-libero-object-traj1"))')"
```

<a id="train"></a>

## 🧠 Train Task Experts

Train one residual expert per task. Run these commands one at a time:

```bash
bash examples/embodiment/run_embodiment.sh \
  libero_object_task6_lora_residual_sac_openvlaoft

bash examples/embodiment/run_embodiment.sh \
  libero_object_task8_lora_residual_sac_openvlaoft
```

Each run saves checkpoints under `logs/<run>/<experiment>/checkpoints/global_step_<step>/actor/huggingface_model/`. Select one checkpoint from each task:

```bash
export CKPT6=/absolute/path/to/task6/actor/huggingface_model
export CKPT8=/absolute/path/to/task8/actor/huggingface_model
export MERGED=/workspace/RLinf/merged_policy_dir/object_6_8
```

Adjust environment counts and training settings in the [task configs](../examples/embodiment/config/). Extra Hydra settings can also be appended to the launcher command.

<a id="rfc"></a>

## 🔎 Compute RFC

Measure compatibility on shared states collected by the frozen base policy on both tasks. The analysis compares the first-action corrections at chunk boundaries:

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

The report gives **direction consistency (DC)**, **magnitude consistency (MC)**, and **RFC = DC × MC**. Merge the pair if RFC meets the threshold (`0.5` here).

<a id="merge"></a>

## 🔀 Merge

Use checkpoints trained with the same frozen base policy, observation encoding, residual scale and architecture:

```bash
python toolkits/merge_lora_policies/quick_merge.py \
  --checkpoint_paths "$CKPT6" "$CKPT8" \
  --output_path "$MERGED" --restore_norm \
  --obs_dim 88 --action_dim 7 --num_action_chunks 8 --actor_input obs
```

The dimensions above match the supplied LIBERO Object configurations. See the [merge toolkit](../toolkits/merge_lora_policies/README.md) for checkpoint formats and weighted merging.

<a id="refine"></a>

## ✨ Optional: Refine

Train the merged residual on both tasks:

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

Use the saved `actor/huggingface_model/` checkpoint for evaluation.

<a id="evaluate"></a>

## 📊 Evaluate

Evaluate the merged policy on both tasks:

```bash
bash examples/embodiment/eval_single_task.sh \
  libero_object_task6_lora_residual_sac_openvlaoft "$MERGED" 6

bash examples/embodiment/eval_single_task.sh \
  libero_object_task8_lora_residual_sac_openvlaoft "$MERGED" 8
```

Results and videos are saved under `results/single_eval_<timestamp>/`. Replace `$MERGED` with a trained or refined checkpoint to compare policies.
