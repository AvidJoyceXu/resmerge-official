<div align="center">

<h1>🤖 ResMerge</h1>

<p><b>Residual Policy Learning and Merging for<br>Continual Adaptation of Pre-trained Robot Policies</b></p>

<a href="https://openreview.net/forum?id=wLtzr7PNBI"><img src="https://img.shields.io/badge/%F0%9F%93%84_Paper-OpenReview-b31b1b?style=for-the-badge" alt="Paper"></a>
<a href="https://resmerge.github.io/"><img src="https://img.shields.io/badge/%F0%9F%8C%90_Project-Website-2563eb?style=for-the-badge" alt="Project Website"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-10b981?style=for-the-badge" alt="License: Apache 2.0"></a>

</div>

**Learn residuals. Check compatibility. Merge robot policies.** ResMerge adapts a frozen robot policy to new tasks with lightweight residual experts. Residual-Field Consistency (RFC) compares their action corrections to guide which experts to merge and refine.

## 🚀 Quick Start

```bash
git clone https://github.com/zhucoffee/resmerge-official.git
cd resmerge-official
```

Choose your setup:

| | Base policy & platform | Get started |
|---|---|---|
| 🧪 Simulation | OpenVLA-OFT · LIBERO | [Train → RFC → merge → evaluate](#simulation) |
| 🦾 Real robot | IMLE · Franka + LEAP hand | [Install](#real-robot) · [Full hardware & learning guide](real_robot/README.md) |

<a id="simulation"></a>

## 🧪 Simulation

Train and merge two residual experts on **LIBERO Object** tasks 6 and 8 with **OpenVLA-OFT**.

### 🛠️ Installation

Requires Linux, NVIDIA GPUs, Docker, and the NVIDIA Container Toolkit. The example below trains and merges **LIBERO Object tasks 6 and 8**. Run the two training commands one at a time.

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

Each run saves checkpoints under `logs/<run>/<experiment>/checkpoints/global_step_<step>/actor/huggingface_model/`. Select one checkpoint from each task:

```bash
export CKPT6=/absolute/path/to/task6/actor/huggingface_model
export CKPT8=/absolute/path/to/task8/actor/huggingface_model
export MERGED=/workspace/RLinf/merged_policy_dir/object_6_8
```

Adjust environment counts and training settings in the [task configs](examples/embodiment/config/). Extra Hydra settings can also be appended to the launcher command.

### 🔀 Compute RFC & Merge

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

The report gives **DC**, **MC**, and **RFC = DC × MC**. Merge the pair if RFC meets the threshold (`0.5` here):

```bash
python toolkits/merge_lora_policies/quick_merge.py \
  --checkpoint_paths "$CKPT6" "$CKPT8" \
  --output_path "$MERGED" --restore_norm \
  --obs_dim 88 --action_dim 7 --num_action_chunks 8 --actor_input obs
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

Results and videos are saved under `results/single_eval_<timestamp>/`. Replace `$MERGED` with a trained or refined checkpoint to compare policies. See the [merge toolkit](toolkits/merge_lora_policies/README.md) for checkpoint formats and weighted merging.

<a id="real-robot"></a>

## 🦾 Real Robot

Use a **Franka arm + 16-DoF LEAP hand**, with PICO tracking, SpaceMouse and a RealSense camera. From the repository root in a separate host terminal, create the Python 3.11 environment:

```bash
cd real_robot
bash install.sh
source .resmerge-venv/bin/activate
cp hardware.example.env hardware.local.env
```

Configure `hardware.local.env` and follow the [hardware setup](real_robot/HARDWARE.md). The **[real-robot guide →](real_robot/README.md)** walks through environment activation, demonstration collection, IMLE pretraining, residual RL, RFC analysis, merging and evaluation. Its actor and learner launchers use residual scale **α = 0.05**.

---

🙏 Built on [RLinf](https://github.com/RLinf/RLinf), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), and [ConRFT](https://github.com/cccedric/conrft).

ResMerge code uses [Apache 2.0](LICENSE). Bundled third-party components retain their [respective licenses](real_robot/THIRD_PARTY.md), including CC BY-NC 4.0 for the LEAP Hand API.
