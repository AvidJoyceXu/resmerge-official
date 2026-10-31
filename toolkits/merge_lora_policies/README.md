# 🔀 LoRA Residual Policy Merging

Combine compatible simulation residual actors into one checkpoint. For training → RFC → merge → evaluation, start with the [main Quick Start](../../README.md#simulation). For Franka + LEAP actors, use the [real-robot guide](../../real_robot/README.md#9-merge-the-task-actors).

## Quick Start

Run from the repository root in the simulation environment. This example uses LIBERO Object privileged-state observations: observation dimension 88, action dimension 7, eight actions per chunk and `actor_input=obs`.

```bash
python toolkits/merge_lora_policies/quick_merge.py \
  --checkpoint_paths /path/to/task6/actor/huggingface_model /path/to/task8/actor/huggingface_model \
  --output_path merged_policy_dir/object_6_8 \
  --obs_dim 88 --action_dim 7 --num_action_chunks 8 --actor_input obs \
  --prune_ratio 0.2 --restore_norm
```

Use the same frozen base policy, observation encoding and residual scale for all inputs. Compute RFC on a shared probe set before merging, then evaluate the merged actor on each constituent task.

### Weighted merging

```bash
python toolkits/merge_lora_policies/quick_merge.py \
  --checkpoint_paths /path/to/expert_A /path/to/expert_B /path/to/expert_C \
  --output_path merged_policy_dir/weighted \
  --obs_dim 88 --action_dim 7 --num_action_chunks 8 --actor_input obs \
  --task_weights 0.5 0.3 0.2 --restore_norm
```

Weights apply to **input experts**, including previously merged experts. They must be finite and nonnegative, with at least one positive weight, and are normalized to sum to one. Zero-weight inputs do not affect the merge. A single positive-weight input is copied without pruning or rescaling.

| Option | Purpose |
|---|---|
| `--checkpoint_paths` | Two or more RLinf checkpoint directories, `.safetensors` files or actor-export `.pt` files |
| `--output_path` | Output directory for safetensors, or a filename ending in `.pt` |
| `--obs_dim`, `--action_dim`, `--num_action_chunks`, `--actor_input` | Architecture and input semantics; required for tensor-only RLinf checkpoints; optional with complete matching metadata |
| `--task_weights` | Optional weight per input expert; defaults to uniform |
| `--prune_ratio` | Fraction of low-magnitude entries to prune from each LoRA A matrix; default `0.2` |
| `--restore_norm` | Restore weighted source norms after averaging LoRA factors and output heads |

Use the dimensions and input mode from your training configuration. Matching tensor shapes alone cannot establish matching observation semantics.

## Merge operator

Each factorized layer `W = BA` is processed as follows:

1. Magnitude-prune entries of each source's **A** matrix.
2. Scale B columns by pre-pruning / retained absolute row mass of A. Fully pruned rows contribute zero.
3. Normalize source `BA` Frobenius norms to their shared mean.
4. Average A, B and biases using the input weights; average the full-rank mean and log-standard-deviation heads with the same weights.
5. With `--restore_norm`, rescale merged layers and output heads to their weighted source norms.

See [merge_lora_actors.py](../../rlinf/models/embodiment/residual_policy/merge_lora_actors.py). `RobustMergeLoRA` implements steps 1–4; `RobustMergeLoRAOptimized` adds step 5.

## Checkpoints

- **RLinf input:** a directory of `*.safetensors` shards, or a single safetensors file.
- **Actor-export input:** `.pt` produced by `LoRAResidualPolicy.save_for_merge()` or a previous merge. Load `.pt` files only from sources you trust.
- **Safetensors output:** a directory containing actor weights and `metadata.json`, usable as `actor.model.model_path` for refinement or `runner.eval_policy_path` for evaluation.
- **PyTorch output:** an explicit `.pt` path, with actor tensors under `merged_params`.

The actor checkpoint supplies the residual network; configure the shared frozen base separately. Training-time critics are not merged.

## Python API

For exports with complete architecture metadata:

```python
from rlinf.models.embodiment.residual_policy import RobustMergeLoRAOptimized

merger = RobustMergeLoRAOptimized(prune_ratio=0.2, restore_norm=True)
merged_params = merger.merge_actors(
    checkpoint_paths=["task6.pt", "task8.pt"],
    task_weights=[0.5, 0.5],
    output_path="merged.pt",
)
```

Use `LoRAResidualPolicy.set_lora_parameters(merged_params)` on a policy initialized with the matching architecture, or use the evaluation launcher in the [main README](../../README.md#simulation).
