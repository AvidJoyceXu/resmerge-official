#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Quick merge script for RLinf LoRA Residual Policies

This script provides a convenient way to merge multiple LoRA Residual Policy checkpoints
trained on different tasks.

Usage:
    # Safetensors format (recommended, default)
    python toolkits/merge_lora_policies/quick_merge.py \
        --checkpoint_paths path/to/task0 path/to/task1 \
        --output_path merged_policy_dir/ \
        --prune_ratio 0.2 \
        --restore_norm
    
    # PyTorch .pt format (backward compatibility)
    python toolkits/merge_lora_policies/quick_merge.py \
        --checkpoint_paths path/to/task0.pt path/to/task1.pt \
        --output_path merged_policy.pt \
        --prune_ratio 0.2 \
        --restore_norm

Note: If output_path is a directory or doesn't end with .pt, saves as safetensors format
      (model-00001-of-00001.safetensors). If output_path ends with .pt, saves as .pt format.
"""

import torch
import argparse
import sys
import os

# Add parent directory to path to import rlinf modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from rlinf.models.embodiment.residual_policy.merge_lora_actors import (
    RobustMergeLoRA,
    RobustMergeLoRAOptimized
)


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple LoRA Residual Policy checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--checkpoint_paths',
        type=str,
        nargs='+',
        required=True,
        help='Paths to checkpoint files/directories to merge. Supports both safetensors format '
             '(directory with model-*.safetensors files) and .pt format (PyTorch checkpoint).'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save the merged checkpoint. If directory or doesn\'t end with .pt, '
             'saves as safetensors format (model-00001-of-00001.safetensors). '
             'If ends with .pt, saves as PyTorch .pt format.'
    )
    
    parser.add_argument(
        '--task_weights',
        type=float,
        nargs='+',
        default=None,
        help='Weights for each task (will be normalized). If not provided, uniform weights are used.'
    )
    
    parser.add_argument(
        '--prune_ratio',
        type=float,
        default=0.2,
        help='Pruning ratio for RobustMerge (default: 0.2)'
    )
    
    parser.add_argument(
        '--restore_norm',
        action='store_true',
        help='Use optimized version with norm restoration (recommended)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.checkpoint_paths) < 2:
        print("Error: At least 2 checkpoint paths are required for merging")
        sys.exit(1)
    
    for path in args.checkpoint_paths:
        if not os.path.exists(path):
            print(f"Error: Checkpoint file not found: {path}")
            sys.exit(1)
    
    if args.task_weights is not None and len(args.task_weights) != len(args.checkpoint_paths):
        print(f"Error: Number of task weights ({len(args.task_weights)}) must match number of checkpoints ({len(args.checkpoint_paths)})")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Choose merge method
    if args.restore_norm:
        print("Using RobustMergeLoRAOptimized (with norm restoration)")
        merger = RobustMergeLoRAOptimized(
            prune_ratio=args.prune_ratio,
            restore_norm=True
        )
    else:
        print("Using RobustMergeLoRA (standard version)")
        merger = RobustMergeLoRA(prune_ratio=args.prune_ratio)
    
    # Execute merge
    merged_params = merger.merge_actors(
        checkpoint_paths=args.checkpoint_paths,
        task_weights=args.task_weights,
        output_path=args.output_path
    )
    
    print(f"\n✅ Merge completed!")
    print(f"Output file: {args.output_path}")
    
    # Print parameter statistics
    rank = merged_params['meta']['rank']
    total_params = 0
    print("\nLoRA Layer Parameters:")
    for layer_name, layer_params in merged_params['lora_layers'].items():
        A_params = layer_params['A'].numel()
        B_params = layer_params['B'].numel()
        B_bias_params = layer_params['B_bias'].numel() if layer_params['B_bias'] is not None else 0
        total_params += A_params + B_params + B_bias_params
        print(f"  {layer_name}: A={A_params:,}, B={B_params:,}, B_bias={B_bias_params:,}")
    
    print(f"\nTotal LoRA parameters: {total_params:,}")
    print(f"LoRA rank: {rank}")
    
    # Print output layer parameters
    print("\nOutput Layer Parameters:")
    for output_name, output_params in merged_params['output_layers'].items():
        weight_params = output_params['weight'].numel()
        bias_params = output_params['bias'].numel()
        print(f"  {output_name}: weight={weight_params:,}, bias={bias_params:,}")


if __name__ == '__main__':
    main()

