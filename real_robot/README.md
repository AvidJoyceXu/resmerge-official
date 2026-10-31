# ResMerge: Real-Robot Reinforcement Learning

Real-robot code for ResMerge with a Franka arm and a 16-DoF LEAP right hand. This repository provides IMLE pretraining, task-specific LoRA residual SAC training, residual-field merge analysis, multi-task actor merging and real-robot evaluation. `pick_place` is the supplied task-environment example.

**Workflow:** PICO hand tracking + SpaceMouse arm teleoperation → successful demonstrations → Zarr dataset → shared IMLE base policy → task-specific LoRA residual RL → merge-feasibility analysis → actor merge → evaluation on each task.

The IMLE base policy remains frozen during RL. Label episode success or failure using the keyboard. Collect demonstrations and train checkpoints using the steps below.

## Repository layout

| Directory | Purpose |
|---|---|
| `teleop/` | PICO hand retargeting, SpaceMouse arm teleoperation, demonstration collection, base-policy evaluation and RL replay preparation |
| `imle_dex/` | IMLE pretraining, LoRA residual RL and multi-task policy merging; its main subdirectories are listed below |
| `imle_dex/precise_dex/` | Demonstration-to-Zarr conversion, IMLE policy models, training configurations and pretraining entry point |
| `imle_dex/examples/` | Runnable experiment entry points and shared example utilities |
| `imle_dex/examples/experiments/imle_lora/` | Residual RL actor/learner, task-actor export, merge-feasibility analysis, merging and evaluation |
| `imle_dex/examples/experiments/pick_place/` | Example task environment, reset poses, workspace bounds and camera settings |
| `imle_dex/serl_launcher/` | Shared RL networks, replay buffers and learner utilities |
| `imle_dex/serl_robot_infra/` | Franka HTTP service and robot observation/action wrappers |
| `LEAP_Hand_API/` | Python communication and motor control for the LEAP hand |
| `XRoboToolkit-Teleop-Sample-Python/` | PICO/XR tracking client and teleoperation integration |
| `dex-retargeting/` | Converts tracked human-hand keypoints into LEAP hand joint targets |
| `dex-urdf/` | LEAP right-hand URDF model, meshes and materials used by retargeting |
| `requirements/` | Python dependency declarations, installation lock file and Git dependency specification |

Root entry points and documentation:

- `install.sh`: installs the Python environment and local packages.
- `start_franka_server.sh`: starts the Franka service in your ROS environment.
- `hardware.example.env`: machine configuration template; copy it to `hardware.local.env`.
- [HARDWARE.md](HARDWARE.md): hardware interfaces and calibration conventions.
- [THIRD_PARTY.md](THIRD_PARTY.md): upstream projects and dependency roles.

`imle_lora` implements the algorithm; `pick_place` supplies the example environment through `experiments/mappings.py`. Training creates datasets and checkpoints at the paths you configure.

## 1. Prepare the hardware

Configure the hardware stack for your machine using the upstream instructions:

| Component | Setup reference | Required result |
|---|---|---|
| Franka arm/controller | [ConRFT](https://github.com/cccedric/conrft) and its linked robot setup | A working ROS workspace with `serl_franka_controllers` |
| LEAP hand | [LEAP Hand API](https://github.com/leap-hand/LEAP_Hand_API) | Python motor communication and correct joint calibration |
| PICO tracking | [XRoboToolkit Teleop](https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python) and [PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) | PC Service receives right-hand tracking; its matching Python SDK is available |
| Hand retargeting | [dex-retargeting](https://github.com/dexsuite/dex-retargeting) | LEAP right-hand vector retargeting; the required configuration and meshes are included here |
| Camera and arm input | Your RealSense and SpaceMouse device setup | Camera access and USB/input permissions |

Choose OS, drivers, ROS and PICO service versions for your hardware. Keep the ROS/controller environment separate from the Python environment below. Install the matching `xrobotoolkit_sdk` binding into the ResMerge environment after creating it. Use `install.sh` to install the bundled packages used by this workflow.

Before collection, verify the arm controller, camera, LEAP readback and PICO tracking independently. Adjust reset poses, workspace bounds, camera crop and joint calibration to your physical setup. See [hardware interfaces](HARDWARE.md) for the expected API and data conventions.

## 2. Install the Python environment

Use Linux x86_64, Python 3.11 with venv support, a C/C++ build toolchain and an NVIDIA driver compatible with the packaged CUDA 12 libraries. On Ubuntu, the toolchain is available through `build-essential`. Install Python using the method appropriate for your OS.

From the official repository's root:

```bash
cd real_robot
bash install.sh
source .resmerge-venv/bin/activate
export REPO_ROOT="$PWD"
export PYTHONPATH="$REPO_ROOT/imle_dex/examples:$REPO_ROOT/imle_dex/precise_dex:$REPO_ROOT/teleop:${PYTHONPATH:-}"
cp hardware.example.env hardware.local.env
```

Edit `hardware.local.env`, then load it in each runtime terminal:

```bash
set -a
source "$REPO_ROOT/hardware.local.env"
set +a
```

Set `ROBOT_IP`, `CATKIN_WS`, `FRANKA_SERVER_URL`, `REALSENSE_SERIAL` and `LEAP_SERIAL_PORT`. Set `HF_HOME` to a writable model cache. Configure task poses, bounds and camera crop in `imle_dex/examples/experiments/pick_place/config.py`. The local `.env` file is ignored by Git.

The installer installs Python dependencies from `requirements/` and the local packages in this repository. It does not install OS drivers, ROS or PICO PC Service. The default image encoder downloads pretrained timm weights on first use. For loading a complete local base checkpoint offline, `RESMERGE_PRETRAINED=0` avoids that download; do not use it for normal pretraining, since it initializes vision weights randomly.

## 3. Teleoperate and collect demonstrations

In a terminal using your configured **ROS environment**, load the hardware configuration and start the Franka service:

```bash
bash start_franka_server.sh
```

In the **ResMerge Python environment**, start PICO PC Service/tracking and run:

```bash
python teleop/record_demos_franka_leap_imle_manual.py \
  --exp_name=pick_place --imle_root="$REPO_ROOT/imle_dex" \
  --stack_obs_num=2 --demos_needed=20 --save_images \
  --output_dir="$REPO_ROOT/data/demos"
```

PICO controls the hand through retargeting; SpaceMouse controls the arm. Press `r` to start recording, `s` to save a successful trajectory, `d` to discard, and `q` to exit. The recorder writes episode pickle files with `transitions` and `meta`, plus HDF5 measurements. Keep images enabled for visual pretraining.

## 4. Convert data and pretrain IMLE

Convert the saved episode directory into Zarr:

```bash
export DATASET_ZARR="$REPO_ROOT/data/demos.zarr"
python imle_dex/precise_dex/pkl2zarr.py \
  --input "$REPO_ROOT/data/demos" \
  --output "$DATASET_ZARR"
```

The converter searches recursively for episode pickle files. It expects the recorder's format, including measured hand positions. The example uses 22 action dimensions (6 arm + 16 hand), 34 state dimensions, and two stacked camera/state observations. Keep these conventions consistent across collection, training and RL. Existing Zarr output requires explicit `--overwrite` to replace.

Train the base policy:

```bash
bash imle_dex/precise_dex/train_imle.sh \
  --dataset "$DATASET_ZARR" \
  logging.mode=disabled logging.entity=null
```

The task configuration is `psi_policy/config/task/franka_leap_image_singlergb.yaml`; the workspace configuration is `psi_policy/config/psi_policy_workspace.yaml`, both under `imle_dex/precise_dex`. Hydra prints the output location. Select a saved checkpoint for evaluation:

```bash
export CKPT_PATH=/path/to/base-policy.ckpt
python teleop/franka_leap_imle_eval.py \
  --exp_name=pick_place --imle_root="$REPO_ROOT/imle_dex" \
  --precise_dex_root="$REPO_ROOT/imle_dex/precise_dex" \
  --ckpt_path="$CKPT_PATH" --stack_obs_num=2
```

This executes the policy on hardware. After reset, `r` starts and `q` exits; space labels success and `f` labels failure. Evaluate the base policy before starting RL.

## 5. Prepare demonstration replay for RL

Pretraining consumes Zarr. Residual RL can additionally consume a replay pickle containing features and base actions computed with the frozen checkpoint:

```bash
export DEMO_PATH="$REPO_ROOT/data/replay.pkl"
python teleop/postprocess_demo_pkl_for_residual_rl.py \
  --input_zarr "$DATASET_ZARR" --output_pkl "$DEMO_PATH" \
  --ckpt_path "$CKPT_PATH" \
  --precise_dex_root "$REPO_ROOT/imle_dex/precise_dex" \
  --stack_obs_num 2 --chunk_exec_steps 4 --latency_step 0
```

Use successful demonstrations collected above: the default terminal-reward mode treats episode endings as success. Do not mix failures/timeouts into that mode. The default output drops RGB after extracting `imle_preproj`. Regenerate replay if the base checkpoint, feature settings or action timing changes.

This step is optional for online-only RL. The learner command below explicitly enables demonstration replay; omit `--demo_paths` to train from online replay only.

## 6. Run LoRA residual RL

In both ResMerge terminals, set the same `CKPT_PATH`, `DEMO_PATH`, `CHECKPOINT_PATH`, hardware configuration and model settings:

```bash
export CHECKPOINT_PATH="$REPO_ROOT/data/rl_checkpoints"
export LORA_RANK=16
```

Start the learner:

```bash
bash imle_dex/examples/experiments/imle_lora/run_learner_res_sac_imle_lora.sh \
  --demo_paths="$DEMO_PATH" --debug
```

Start the robot actor in the other terminal:

```bash
export LEARNER_IP=localhost
HIL=1 bash imle_dex/examples/experiments/imle_lora/run_actor_res_sac_imle_lora.sh --debug
```

`HIL=1` enables SpaceMouse arm intervention and stores the executed intervention action in replay. Use `HIL=0` for policy control with keyboard labels only. After reset, press `r` to start or `t` to reset again; space ends an episode successfully and `f` ends it as failure.

The learner waits for sufficient online replay, then mixes online and demonstration samples 50/50 when `--demo_paths` is supplied. `--debug` disables W&B; use your own external credentials/entity for online logging. Actor `max_steps` counts environment interactions; learner `max_steps` counts gradient updates.

The launchers agree on observation stack 2, action horizon 16, chunk execution 4, latency 0 and residual scale 0.05. Keep these settings, LoRA rank and the base checkpoint consistent. For separate actor/learner machines, configure reachable addresses and networking. To resume, set `RESUME_RUN` to the previous run directory containing its checkpoints and replay snapshot.

## 7. Train task-specific residuals and export their actors

Repeat the RL stage for each task/object condition, retaining a separate run directory and its task configuration. The supplied `pick_place` configuration is a template; configure the physical task/reset setup for each run. Keep the **same frozen IMLE checkpoint**, observation encoding, action/joint conventions, LoRA rank and residual scale across tasks you intend to merge. Matching tensor shapes alone does not establish compatible feature semantics.

The learner periodically saves `task_<exp_name>_step<step>_for_lora_merge.pt` and sampled `states_imle_preproj_step<step>.npy` in its run directory. These contain the actor export and feature samples used below. Run directories distinguish task conditions even when both use the `pick_place` environment name.

You can also export a selected RL checkpoint explicitly. Run this once for task A and again for task B with their respective run directories:

```bash
export RUN_DIR=/path/to/task_A_run
python imle_dex/examples/experiments/imle_lora/export_lora_merge_pt.py \
  --run_dir="$RUN_DIR" --ckpt_path="$CKPT_PATH" \
  --lora_rank="$LORA_RANK" --residual_alpha=0.05 --out="$RUN_DIR/actor.pt"

export TASK_A_PT=/path/to/task_A_run/actor.pt
export TASK_B_PT=/path/to/task_B_run/actor.pt
export LORA_DIR="$REPO_ROOT/imle_dex/examples/experiments/imle_lora"
```

Evaluate each unmerged actor first using the evaluation command in step 10 with its own `.pt` file. This provides the per-task baseline for assessing the merged actor.

## 8. Analyze whether task residuals are compatible

`analyze_correction_field.py` queries two actors on the **same** IMLE feature states and compares their deterministic corrections `residual_alpha * tanh(mean_u)`. It computes the paper's **RFC = DC × MC**: DC is mean stabilized direction cosine and MC is mean bounded magnitude agreement, `2 min(‖u‖, ‖v‖) / (‖u‖ + ‖v‖ + ε)`. Absolute log norm ratio and the fraction of states whose direction agreement falls below `delta_dir` are also reported as diagnostics.

Use states covering both tasks, rather than only a convenient subset from one task. For example, combine learner-saved feature samples from the two runs:

```bash
export STATES_A=/path/to/task_A_run/states_imle_preproj_stepSTEP.npy
export STATES_B=/path/to/task_B_run/states_imle_preproj_stepSTEP.npy
export MERGE_STATES="$REPO_ROOT/data/merge_states.npy"
python - <<'PY_STATES'
import os
from pathlib import Path
import numpy as np
arrays = [np.load(os.environ[key]) for key in ("STATES_A", "STATES_B")]
assert all(a.ndim == 2 and len(a) > 0 for a in arrays)
assert arrays[0].shape[1] == arrays[1].shape[1]
# Equal representation of the two sampled task distributions.
rng = np.random.default_rng(0)
n = min(map(len, arrays))
states = np.concatenate([a[rng.choice(len(a), n, replace=False)] for a in arrays])
output = Path(os.environ["MERGE_STATES"])
output.parent.mkdir(parents=True, exist_ok=True)
np.save(output, states.astype(np.float32))
PY_STATES

python "$LORA_DIR/analyze_correction_field.py" \
  --task_i_pt="$TASK_A_PT" --task_j_pt="$TASK_B_PT" \
  --states_npy="$MERGE_STATES" --residual_alpha=0.05 --tau=0.5 \
  --output_dir="$REPO_ROOT/data/merge_analysis"
```

Alternatively, pass `--demo_pkl=/path/to/replay.pkl` instead of `--states_npy`, using feature-enriched replay from step 5. Raw demonstration pickles without `imle_preproj` cannot be used directly.

The analysis writes a plot (`.png`), raw arrays (`.npz`) and a decision report (`.json`). The default rule recommends merging when **RFC ≥ 0.5**. Use `--tau` to set the RFC threshold. `--decision_rule=legacy` selects the earlier three-threshold diagnostic (`threshold_dir`, `threshold_scale`, `threshold_bad`). Evaluate the merged policy on each task in step 10. For sequential adaptation, compare a new task actor against each current expert and merge only with an expert that passes the RFC gate.

## 9. Merge the task actors

Run `quick_merge.py` to combine task actors. The algorithms are implemented in `merge_lora_actors.py`; the default method is `RobustMergeLoRA` with pruning ratio 0.2:

```bash
export MERGED_PT="$REPO_ROOT/data/merged_actor.pt"
python "$LORA_DIR/quick_merge.py" \
  --checkpoint_paths "$TASK_A_PT" "$TASK_B_PT" \
  --task_weights 0.5 0.5 --method robust --prune_ratio 0.2 \
  --output_path "$MERGED_PT"
```

Supply two or more single-task exports or previously merged actors. Omit `--task_weights` for uniform weights over the **input experts**; a consolidated expert counts as one input. For example, weights `0.5 0.5` for `(AB, C)` weight expert AB and expert C equally, rather than weighting all three original tasks equally. Run the compatibility analysis in step 8 first. The merge entry point also checks parameter shapes, rank, finite values and available base-policy/residual-scale metadata. Existing output requires `--overwrite`.

Select alternative methods with `--method optimized` or `--method directional`. Performance-aware and hierarchical merging are available as Python APIs in `merge_lora_actors.py`.

The output contains the merged actor and LayerNorm parameters, shared base/scale metadata when available for all inputs, and original-export lineage. Pass this file directly to evaluation or the next RFC analysis and merge.

## 10. Evaluate the merged policy on every task

Keep the same frozen IMLE base and residual scale used for training/analysis. Configure the robot for each task in turn and run:

```bash
HIL=0 EVAL_MERGED_PT="$MERGED_PT" EVAL_EPISODES=20 \
  bash "$LORA_DIR/run_actor_res_sac_imle_lora.sh" --debug
```

Evaluation does not connect to the learner or write replay. After reset, `r` starts, space marks success, `f` marks failure, and `t` resets again while waiting. Repeat on each constituent task and compare success rates against its unmerged actor under the same conditions. Use `HIL=0 EVAL_BASE_ONLY=1` to evaluate the frozen IMLE base alone.

The `EVAL_MERGED_PT` interface also accepts a single-task actor export, so the same evaluation entry point covers both baselines and merges.

Keep the position-embedding configuration consistent across pretraining and policy loading: the default uses `policy.use_rotary_emb=true`; checkpoints containing learned `model.pos_emb` require `policy.use_rotary_emb=false` consistently across loading and replay generation.
