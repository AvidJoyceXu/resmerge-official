# LoRA Residual Policy Merge Tools

This toolkit provides utilities for merging multiple LoRA Residual Policy checkpoints trained on different tasks, enabling multi-task policy support.

## Overview

The merge tools implement the **RobustMerge** algorithm for LoRA parameters, which includes:
1. **Pruning & Complementary Scaling**: Prunes small parameters and compensates with scaling
2. **Cross-Task Normalization**: Normalizes parameters across tasks
3. **Weighted Averaging**: Merges parameters with task-specific weights
4. **Norm Restoration** (optional): Restores original parameter norms after merging

## Quick Start

### Basic Usage

Merge two checkpoints with uniform weights:

```bash
# Save as safetensors format (recommended, default)
python toolkits/merge_lora_policies/quick_merge.py \
    --checkpoint_paths path/to/task0 path/to/task1 \
    --output_path merged_policy_dir/

# Save as .pt format (backward compatibility)
python toolkits/merge_lora_policies/quick_merge.py \
    --checkpoint_paths path/to/task0.pt path/to/task1.pt \
    --output_path merged_policy.pt
```

**Note**: 
- If `output_path` is a directory or doesn't end with `.pt`, the merged checkpoint will be saved in **safetensors format** (compatible with RLinf's checkpoint format: `model-00001-of-00001.safetensors`)
- If `output_path` ends with `.pt`, it will be saved as a PyTorch `.pt` file (backward compatibility)

### Advanced Usage

Merge multiple checkpoints with custom weights and norm restoration:

```bash
# Safetensors format (recommended)
python toolkits/merge_lora_policies/quick_merge.py \
    --checkpoint_paths path/to/task0 path/to/task1 path/to/task2 \
    --output_path merged_policy_dir/ \
    --task_weights 0.5 0.3 0.2 \
    --prune_ratio 0.2 \
    --restore_norm

# .pt format
python toolkits/merge_lora_policies/quick_merge.py \
    --checkpoint_paths path/to/task0.pt path/to/task1.pt path/to/task2.pt \
    --output_path merged_policy.pt \
    --task_weights 0.5 0.3 0.2 \
    --prune_ratio 0.2 \
    --restore_norm
```

### Parameters

- `--checkpoint_paths`: List of checkpoint file paths (saved via `LoRAResidualPolicy.save_for_merge()`)
- `--output_path`: Path to save the merged checkpoint
- `--task_weights`: Optional weights for each task (will be normalized). If not provided, uniform weights are used.
- `--prune_ratio`: Pruning ratio for RobustMerge (default: 0.2)
- `--restore_norm`: Use optimized version with norm restoration (recommended)

## Programmatic Usage

You can also use the merge classes directly in Python:

```python
from rlinf.models.embodiment.residual_policy import RobustMergeLoRAOptimized

# Initialize merger
merger = RobustMergeLoRAOptimized(prune_ratio=0.2, restore_norm=True)

# Merge checkpoints
merged_params = merger.merge_actors(
    checkpoint_paths=[
        'path/to/task0.pt',
        'path/to/task1.pt'
    ],
    task_weights=[0.5, 0.5],  # Optional
    output_path='merged_policy.pt'
)

# Load merged parameters into a policy
from rlinf.models.embodiment.residual_policy import LoRAResidualPolicy

policy = LoRAResidualPolicy(
    obs_dim=512,
    action_dim=7,
    num_action_chunks=1,
    rank=16
)

# Load merged parameters
checkpoint = torch.load('merged_policy.pt', map_location='cpu')
policy.set_lora_parameters(checkpoint['merged_params'], device='cuda')
```

## Checkpoint Format

### Loading Checkpoints

The merge tool supports multiple checkpoint formats:

1. **Safetensors format** (recommended, compatible with RLinf):
   - Directory containing `model-00001-of-00001.safetensors` files
   - Or single `model.safetensors` file
   - Automatically detected when path is a directory or ends with `.safetensors`

2. **PyTorch .pt format** (backward compatibility):
   - Saved using `LoRAResidualPolicy.save_for_merge()`:
   ```python
   policy = LoRAResidualPolicy(...)
   policy.save_for_merge(
       save_path='task0.pt',
       task_id=0,
       additional_info={'training_steps': 1000000}
   )
   ```

### Saving Checkpoints

The merge tool can save in two formats:

1. **Safetensors format** (default, recommended):
   - Specify a directory path: `--output_path merged_policy_dir/`
   - Creates `model-00001-of-00001.safetensors` files (compatible with RLinf)
   - Also saves `metadata.json` with merge information

2. **PyTorch .pt format** (backward compatibility):
   - Specify a file path ending with `.pt`: `--output_path merged_policy.pt`

### Checkpoint Structure

The checkpoint format is:
```python
{
    'task_id': int,
    'params': {
        'lora_layers': {
            'fc1': {'A': tensor, 'B': tensor, 'B_bias': tensor},
            'fc2': {'A': tensor, 'B': tensor, 'B_bias': tensor},
            'fc3': {'A': tensor, 'B': tensor, 'B_bias': tensor}
        },
        'output_layers': {
            'fc_mean': {'weight': tensor, 'bias': tensor},
            'fc_logstd': {'weight': tensor, 'bias': tensor}
        },
        'meta': {
            'rank': int,
            'obs_dim': int,
            'action_dim': int,
            'num_action_chunks': int,
            'actor_input': str
        }
    },
    'state_dict': dict,  # Full state_dict as backup
    'additional_info': dict
}
```

## Merge Methods

### RobustMergeLoRA

Standard RobustMerge implementation:
- Pruning & Complementary Scaling
- Cross-Task Normalization
- Weighted Averaging

### RobustMergeLoRAOptimized

Optimized version with additional features:
- All features of RobustMergeLoRA
- Norm restoration for LoRA layers
- Norm restoration for output layers

**Recommendation**: Use `RobustMergeLoRAOptimized` with `restore_norm=True` for better performance.

## Example Workflow

1. **Train policies on different tasks**:
   ```python
   # Train task 0
   policy_task0 = train_policy(task=0)
   policy_task0.save_for_merge('task0.pt', task_id=0)
   
   # Train task 1
   policy_task1 = train_policy(task=1)
   policy_task1.save_for_merge('task1.pt', task_id=1)
   ```

2. **Merge policies**:
   ```bash
   python toolkits/merge_lora_policies/quick_merge.py \
       --checkpoint_paths task0.pt task1.pt \
       --output_path merged.pt \
       --restore_norm
   ```

3. **Load and use merged policy**:
   ```python
   # Create policy instance
   merged_policy = LoRAResidualPolicy(...)
   
   # Load merged parameters (safetensors format)
   from rlinf.models.embodiment.residual_policy.merge_lora_actors import _load_checkpoint_safetensors, _load_lora_params_from_checkpoint
   
   checkpoint = _load_checkpoint_safetensors('merged_policy_dir/')
   merged_params = _load_lora_params_from_checkpoint(checkpoint)
   merged_policy.set_lora_parameters(merged_params, device='cuda')
   
   # Or load from .pt format
   # checkpoint = torch.load('merged.pt', map_location='cpu')
   # merged_policy.set_lora_parameters(checkpoint['merged_params'], device='cuda')
   
   # Use for inference or further training
   action = merged_policy(observation)
   ```

## Notes

- All checkpoints must have the same architecture (same rank, dimensions, etc.)
- The merged checkpoint format is compatible with `LoRAResidualPolicy.set_lora_parameters()`
- The merge process preserves the parameter structure and metadata from the first checkpoint
- Task weights are automatically normalized if they don't sum to 1.0

