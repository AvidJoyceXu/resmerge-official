<div align="center">

<h1>🤖 ResMerge</h1>

<p><b>Residual Policy Learning and Merging for<br>Continual Adaptation of Pre-trained Robot Policies</b></p>

<a href="https://openreview.net/forum?id=wLtzr7PNBI"><img src="https://img.shields.io/badge/%F0%9F%93%84_Paper-OpenReview-b31b1b?style=for-the-badge" alt="Paper"></a>
<a href="https://resmerge.github.io/"><img src="https://img.shields.io/badge/%F0%9F%8C%90_Project-Website-2563eb?style=for-the-badge" alt="Project Website"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-10b981?style=for-the-badge" alt="License: Apache 2.0"></a>

</div>

**Learn residuals. Check compatibility. Merge robot policies.** ResMerge enables continual adaptation of a frozen robot policy with lightweight residual experts and Residual-Field Consistency (RFC).

## 🚀 Quick Start

Train and merge two residual experts on **LIBERO Object** tasks 6 and 8 with **OpenVLA-OFT**.

### 🛠️ Installation

Requires Linux, NVIDIA GPUs, Docker, and the NVIDIA Container Toolkit.

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

Inside the container:

```bash
cd /workspace/RLinf
export PATH=/opt/venv/openvla-oft/bin:$PATH
export PYTHONPATH=/workspace/RLinf
export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export WANDB_MODE=offline
python -m pip install -r requirements/resmerge-analysis.txt

export RESMERGE_OBJECT_BASE="$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Haozhan72/Openvla-oft-SFT-libero-object-traj1"))')"
```

### 🧠 Train

Train one residual expert per task:

```bash
bash examples/embodiment/run_embodiment.sh \
  libero_object_task6_lora_residual_sac_openvlaoft

bash examples/embodiment/run_embodiment.sh \
  libero_object_task8_lora_residual_sac_openvlaoft
```

Checkpoints: `logs/<run>/<experiment>/checkpoints/global_step_<step>/actor/huggingface_model/`. Set the paths to your trained experts:

```bash
export CKPT6=/absolute/path/to/task6/actor/huggingface_model
export CKPT8=/absolute/path/to/task8/actor/huggingface_model
export MERGED=/workspace/RLinf/merged_policy_dir/object_6_8
```

Training settings are in the [task configs](examples/embodiment/config/).

### 🔀 Compute RFC & Merge

Measure compatibility between the two experts:

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

Merge the pair if its RFC meets the threshold:

```bash
python toolkits/merge_lora_policies/quick_merge.py \
  --checkpoint_paths "$CKPT6" "$CKPT8" \
  --output_path "$MERGED" --restore_norm
```

<details>
<summary><b>✨ Optional: refine the merged policy</b></summary>

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

</details>

### 📊 Evaluate

Evaluate the merged policy on both tasks:

```bash
bash examples/embodiment/eval_single_task.sh \
  libero_object_task6_lora_residual_sac_openvlaoft "$MERGED" 6

bash examples/embodiment/eval_single_task.sh \
  libero_object_task8_lora_residual_sac_openvlaoft "$MERGED" 8
```

Results: `results/single_eval_<timestamp>/`. Replace `$MERGED` with any trained or refined checkpoint to evaluate it.

---

🙏 Built on [RLinf](https://github.com/RLinf/RLinf), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). Released under [Apache 2.0](LICENSE).
