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
LoRA Residual Actor Merge Methods
用于融合多个任务训练的 LoRA Residual Actor

本文件包含融合多个 LoRA Residual Policy 的方法，兼容 RLinf 的 LoRA Residual Policy。
"""

import torch
import numpy as np
import math
import os
import glob
import json
from typing import List, Dict, Optional

try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available. Install with: pip install safetensors")


def _load_checkpoint_safetensors(load_path: str) -> Dict:
    """
    Load checkpoint from safetensors format (directory with .safetensors files).
    
    Args:
        load_path: Path to directory containing safetensors files or checkpoint file
        
    Returns:
        dict: Checkpoint dictionary
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required but not installed. Install with: pip install safetensors")
    
    if os.path.isdir(load_path):
        # Check for safetensors files
        safetensor_files = sorted(glob.glob(os.path.join(load_path, "*.safetensors")))
        # Filter out index files
        safetensor_files = [f for f in safetensor_files if not f.endswith(".index.json")]
        
        if safetensor_files:
            model_dict = {}
            for safetensor_file in safetensor_files:
                model_dict.update(safetensors.torch.load_file(safetensor_file))
            
            # Try to load metadata if available
            metadata_path = os.path.join(load_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return {'state_dict': model_dict, **metadata}
            
            return {'state_dict': model_dict}
        else:
            raise FileNotFoundError(
                f"No safetensors files found in checkpoint directory: {load_path}"
            )
    else:
        # Single safetensors file
        if load_path.endswith('.safetensors'):
            model_dict = safetensors.torch.load_file(load_path)
            return {'state_dict': model_dict}
        else:
            # Fall back to torch.load
            return torch.load(load_path, map_location='cpu')


def _load_lora_params_from_checkpoint(checkpoint: Dict) -> Dict:
    """
    Extract LoRA parameters from checkpoint (supports both .pt and safetensors formats).
    
    Args:
        checkpoint: Checkpoint dictionary
        
    Returns:
        dict: LoRA parameters in get_lora_parameters() format
    """
    # If checkpoint has 'params' key (from save_for_merge)
    if 'params' in checkpoint:
        return checkpoint['params']
    
    # If checkpoint has 'merged_params' key (from previous merge)
    if 'merged_params' in checkpoint:
        return checkpoint['merged_params']
    
    # If checkpoint has 'state_dict', try to extract LoRA parameters
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Try to reconstruct LoRA parameters from state_dict
        # This assumes state_dict keys match LoRAResidualPolicy parameter names
        lora_layers = {}
        output_layers = {}
        
        for layer_name in ['fc1', 'fc2', 'fc3']:
            A_key = f'{layer_name}_A.weight'
            B_key = f'{layer_name}_B.weight'
            B_bias_key = f'{layer_name}_B.bias'
            
            if A_key in state_dict and B_key in state_dict:
                lora_layers[layer_name] = {
                    'A': state_dict[A_key],
                    'B': state_dict[B_key],
                    'B_bias': state_dict.get(B_bias_key, None)
                }
        
        # Extract output layers
        if 'actor_mean.weight' in state_dict:
            output_layers['fc_mean'] = {
                'weight': state_dict['actor_mean.weight'],
                'bias': state_dict.get('actor_mean.bias', None)
            }
        if 'actor_logstd.weight' in state_dict:
            output_layers['fc_logstd'] = {
                'weight': state_dict['actor_logstd.weight'],
                'bias': state_dict.get('actor_logstd.bias', None)
            }
        
        # Extract meta information if available
        meta = checkpoint.get('meta', {})
        if not meta:
            # Try to infer from state_dict shapes
            if 'fc1_A.weight' in state_dict:
                rank = state_dict['fc1_A.weight'].shape[0]
                meta = {'rank': rank}
        
        return {
            'lora_layers': lora_layers,
            'output_layers': output_layers,
            'meta': meta
        }
    
    # If checkpoint is directly params dict
    if 'lora_layers' in checkpoint and 'output_layers' in checkpoint:
        return checkpoint
    
    raise ValueError("Cannot extract LoRA parameters from checkpoint. "
                    "Checkpoint should have 'params', 'merged_params', or 'state_dict' key.")


def _save_checkpoint_safetensors(
    merged_params: Dict,
    output_path: str,
    task_weights: Optional[List[float]] = None,
    n_tasks: int = 0,
    prune_ratio: float = 0.2,
    merge_method: str = 'robust_merge',
    reference_checkpoint_path: Optional[str] = None
) -> None:
    """
    Save merged checkpoint in safetensors format.
    
    Args:
        merged_params: Merged parameters dict
        output_path: Output directory path (will create safetensors files here)
        task_weights: Task weights used for merging
        n_tasks: Number of tasks merged
        prune_ratio: Pruning ratio used
        merge_method: Merge method name
        reference_checkpoint_path: Optional path to a reference checkpoint to copy
                                   non-LoRA parameters (q_head, action_scale, action_bias, etc.)
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required but not installed. Install with: pip install safetensors")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Convert merged_params to state_dict format
    state_dict = {}
    
    # Convert LoRA layers
    for layer_name in ['fc1', 'fc2', 'fc3']:
        layer_params = merged_params['lora_layers'][layer_name]
        state_dict[f'{layer_name}_A.weight'] = layer_params['A']
        state_dict[f'{layer_name}_B.weight'] = layer_params['B']
        if layer_params['B_bias'] is not None:
            state_dict[f'{layer_name}_B.bias'] = layer_params['B_bias']
    
    # Convert output layers
    state_dict['actor_mean.weight'] = merged_params['output_layers']['fc_mean']['weight']
    state_dict['actor_mean.bias'] = merged_params['output_layers']['fc_mean']['bias']
    state_dict['actor_logstd.weight'] = merged_params['output_layers']['fc_logstd']['weight']
    state_dict['actor_logstd.bias'] = merged_params['output_layers']['fc_logstd']['bias']
    
    # Copy non-LoRA parameters from reference checkpoint if provided
    # These include: q_head, action_scale, action_bias, and other buffers/parameters
    if reference_checkpoint_path is not None:
        try:
            # Load reference checkpoint
            if os.path.isdir(reference_checkpoint_path) or reference_checkpoint_path.endswith('.safetensors'):
                ref_checkpoint = _load_checkpoint_safetensors(reference_checkpoint_path)
                ref_state_dict = ref_checkpoint.get('state_dict', ref_checkpoint)
            else:
                ref_checkpoint = torch.load(reference_checkpoint_path, map_location='cpu')
                if 'state_dict' in ref_checkpoint:
                    ref_state_dict = ref_checkpoint['state_dict']
                elif 'params' in ref_checkpoint and 'state_dict' in ref_checkpoint:
                    ref_state_dict = ref_checkpoint['state_dict']
                else:
                    ref_state_dict = ref_checkpoint
            
            # Copy non-LoRA parameters
            non_lora_keys = [
                'action_scale', 'action_bias',  # Buffers
            ]
            # Copy q_head parameters
            for key in ref_state_dict.keys():
                if 'q_head' in key:
                    non_lora_keys.append(key)
            
            for key in non_lora_keys:
                if key in ref_state_dict:
                    state_dict[key] = ref_state_dict[key].clone()
                    print(f"  Copied {key} from reference checkpoint")
        except Exception as e:
            print(f"  Warning: Failed to copy non-LoRA parameters from reference checkpoint: {e}")
            print(f"  The merged checkpoint will only contain LoRA parameters.")
    
    # Save using safetensors (single file or sharded)
    # Use save_state_dict_sharded_safetensors for consistency with RLinf format
    try:
        from rlinf.hybrid_engines.fsdp.utils import save_state_dict_sharded_safetensors
        # Save as sharded safetensors (format: model-00001-of-00001.safetensors)
        num_shards, total_size = save_state_dict_sharded_safetensors(
            state_dict=state_dict,
            out_dir=output_path,
            base_name="model",
            max_shard_size=4 * 1024**3  # 4GB per shard
        )
        print(f"Saved {num_shards} sharded safetensors files (total size: {total_size / 1024**2:.2f} MB)")
    except ImportError:
        # Fallback to single safetensors file
        safetensors.torch.save_file(
            state_dict,
            os.path.join(output_path, "model.safetensors"),
            metadata={"format": "pt"}
        )
        print(f"Saved single safetensors file: model.safetensors")
    
    # Save metadata (without tensors, as they're in safetensors files)
    metadata = {
        'task_weights': task_weights,
        'n_tasks': n_tasks,
        'prune_ratio': prune_ratio,
        'merge_method': merge_method,
        'meta': merged_params.get('meta', {})  # Save meta info only
    }
    
    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"Saved metadata to {metadata_path}")


class RobustMergeLoRA:
    """
    借鉴 RobustMerge 思想融合多个 LoRA Residual Actors
    
    核心步骤：
    1. Pruning & Complementary Scaling: 修剪小参数并补偿缩放
    2. Cross-Task Normalization: 跨任务归一化
    3. Weighted Averaging: 加权平均融合
    """
    
    def __init__(self, prune_ratio=0.2):
        """
        Args:
            prune_ratio: float, 修剪掉每层后 k% 的小参数
        """
        self.prune_ratio = prune_ratio
    
    def merge_actors(
        self, 
        checkpoint_paths: List[str], 
        task_weights: Optional[List[float]] = None,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        融合多个 LoRA Residual Actor
        
        Args:
            checkpoint_paths: List[str], 每个任务的checkpoint路径
            task_weights: Optional[List[float]], 每个任务的权重，默认均匀
            output_path: Optional[str], 保存融合结果的路径
        
        Returns:
            merged_params: Dict, 融合后的参数，格式与 LoRAResidualPolicy.get_lora_parameters() 一致
        """
        n_tasks = len(checkpoint_paths)
        
        # 默认均匀权重
        if task_weights is None:
            task_weights = [1.0 / n_tasks] * n_tasks
        else:
            # 归一化权重
            total_weight = sum(task_weights)
            task_weights = [w / total_weight for w in task_weights]
        
        print(f"Merging {n_tasks} LoRA Residual Actors...")
        print(f"Task weights: {task_weights}")
        
        # 1. 加载所有任务的参数
        all_task_params = []
        for i, path in enumerate(checkpoint_paths):
            print(f"Loading task {i} from {path}...")
            
            # Try to load as safetensors first, then fall back to torch.load
            try:
                if os.path.isdir(path) or (os.path.isfile(path) and path.endswith('.safetensors')):
                    checkpoint = _load_checkpoint_safetensors(path)
                else:
                    checkpoint = torch.load(path, map_location='cpu')
            except Exception as e:
                # Fall back to torch.load if safetensors loading fails
                print(f"  Warning: Failed to load as safetensors, trying torch.load: {e}")
                checkpoint = torch.load(path, map_location='cpu')
            
            # Extract LoRA parameters from checkpoint
            params = _load_lora_params_from_checkpoint(checkpoint)
            all_task_params.append(params)
        
        # 2. 融合 LoRA 层（使用 RobustMerge）
        merged_lora_layers = {}
        for layer_name in ['fc1', 'fc2', 'fc3']:
            print(f"Merging {layer_name}...")
            
            # 提取所有任务的 A 和 B 矩阵
            A_matrices = [p['lora_layers'][layer_name]['A'] for p in all_task_params]
            B_matrices = [p['lora_layers'][layer_name]['B'] for p in all_task_params]
            B_biases = [p['lora_layers'][layer_name]['B_bias'] for p in all_task_params]
            
            # 应用 RobustMerge
            A_merged, B_merged, B_bias_merged = self._robust_merge_lora_layer(
                A_matrices, B_matrices, B_biases, task_weights
            )
            
            merged_lora_layers[layer_name] = {
                'A': A_merged,
                'B': B_merged,
                'B_bias': B_bias_merged
            }
        
        # 3. 融合输出层（简单加权平均）
        merged_output_layers = {}
        for output_name in ['fc_mean', 'fc_logstd']:
            print(f"Merging {output_name}...")
            
            weights = [p['output_layers'][output_name]['weight'] for p in all_task_params]
            biases = [p['output_layers'][output_name]['bias'] for p in all_task_params]
            
            merged_weight = sum(w * weight for w, weight in zip(task_weights, weights))
            merged_bias = sum(w * bias for w, bias in zip(task_weights, biases))
            
            merged_output_layers[output_name] = {
                'weight': merged_weight,
                'bias': merged_bias
            }
        
        # 4. 组装最终参数（格式与 get_lora_parameters() 一致）
        merged_params = {
            'lora_layers': merged_lora_layers,
            'output_layers': merged_output_layers,
            'meta': all_task_params[0]['meta']  # 元信息从第一个任务复制
        }
        
        # 5. 保存（如果指定了路径）
        if output_path is not None:
            # Check if output_path is a directory (for safetensors) or file (for .pt)
            if os.path.isdir(output_path) or output_path.endswith('/') or (not os.path.exists(output_path) and not output_path.endswith('.pt')):
                # Save as safetensors format
                # Determine the actual directory path
                if os.path.isdir(output_path):
                    save_dir = output_path
                elif output_path.endswith('/'):
                    save_dir = output_path.rstrip('/')
                else:
                    # Create directory from output_path
                    save_dir = output_path if os.path.dirname(output_path) else output_path
                    os.makedirs(save_dir, exist_ok=True)
                
                _save_checkpoint_safetensors(
                    merged_params=merged_params,
                    output_path=save_dir,
                    task_weights=task_weights,
                    n_tasks=n_tasks,
                    prune_ratio=self.prune_ratio,
                    merge_method='robust_merge',
                    reference_checkpoint_path=checkpoint_paths[0] if checkpoint_paths else None
                )
            else:
                # Save as .pt format (backward compatibility)
                save_dict = {
                    'merged_params': merged_params,
                    'task_weights': task_weights,
                    'n_tasks': n_tasks,
                    'prune_ratio': self.prune_ratio,
                    'merge_method': 'robust_merge'
                }
                torch.save(save_dict, output_path)
                print(f"Saved merged model to {output_path}")
        
        return merged_params
    
    def _robust_merge_lora_layer(
        self, 
        A_matrices: List[torch.Tensor], 
        B_matrices: List[torch.Tensor],
        B_biases: List[Optional[torch.Tensor]],
        task_weights: List[float]
    ):
        """
        对单层的 LoRA 参数进行 RobustMerge
        带详细的范数追踪
        
        Args:
            A_matrices: List[torch.Tensor], 所有任务的 A 矩阵
            B_matrices: List[torch.Tensor], 所有任务的 B 矩阵
            B_biases: List[Optional[torch.Tensor]], 所有任务的 B bias
            task_weights: List[float], 任务权重
        
        Returns:
            A_merged: torch.Tensor, 融合后的 A 矩阵
            B_merged: torch.Tensor, 融合后的 B 矩阵
            B_bias_merged: Optional[torch.Tensor], 融合后的 B bias
        """
        n_tasks = len(A_matrices)
        
        print(f"\n  [Debug] Starting merge for {n_tasks} tasks")
        
        # ===== 记录原始范数 =====
        original_norms = []
        for i in range(n_tasks):
            W_original = torch.mm(B_matrices[i], A_matrices[i])
            norm = torch.norm(W_original, p='fro').item()
            original_norms.append(norm)
            print(f"  [Debug] Task {i} - Original norm: {norm:.4f}")
        print(f"  [Debug] Original average norm: {np.mean(original_norms):.4f}")
        
        # Step 1: Pruning & Complementary Scaling
        A_scaled = []
        B_scaled = []
        
        print(f"\n  [Debug] Step 1: Pruning (ratio={self.prune_ratio}) & Complementary Scaling")
        
        for i in range(n_tasks):
            A = A_matrices[i].clone()
            B = B_matrices[i].clone()
            
            # 1.1 修剪前
            W_before_prune = torch.mm(B, A)
            norm_before_prune = torch.norm(W_before_prune, p='fro').item()
            
            # 1.1 修剪
            A_pruned = self._prune_matrix(A, self.prune_ratio)
            
            # 修剪后
            W_after_prune = torch.mm(B, A_pruned)
            norm_after_prune = torch.norm(W_after_prune, p='fro').item()
            
            # 1.2 互补缩放
            S = self._compute_scaling_matrix(A_pruned, B)
            
            # 1.3 应用缩放
            B_scaled_task = B * S.unsqueeze(0)
            
            # 缩放后
            W_after_scaling = torch.mm(B_scaled_task, A_pruned)
            norm_after_scaling = torch.norm(W_after_scaling, p='fro').item()
            
            print(f"    Task {i}:")
            print(f"      After Pruning:   {norm_after_prune:.4f} ({norm_after_prune/norm_before_prune*100:.1f}% of original)")
            print(f"      After Scaling:   {norm_after_scaling:.4f} ({norm_after_scaling/norm_before_prune*100:.1f}% of original)")
            print(f"      S matrix - min: {S.min().item():.4f}, max: {S.max().item():.4f}, mean: {S.mean().item():.4f}")
            
            A_scaled.append(A_pruned)
            B_scaled.append(B_scaled_task)
        
        # Step 2: Cross-Task Normalization
        print(f"\n  [Debug] Step 2: Cross-Task Normalization")
        
        # 计算缩放后的范数
        norms_before_normalize = []
        for i in range(n_tasks):
            W_effective = torch.mm(B_scaled[i], A_scaled[i])
            norm = torch.norm(W_effective, p='fro').item()
            norms_before_normalize.append(norm)
            print(f"    Task {i} norm before normalize: {norm:.4f}")
        
        # 归一化
        avg_norm = np.mean(norms_before_normalize)
        print(f"  [Debug] Target norm (average): {avg_norm:.4f}")
        
        norms_after_normalize = []
        for i in range(n_tasks):
            if norms_before_normalize[i] > 1e-6:
                scale = avg_norm / norms_before_normalize[i]
                A_scaled[i] = A_scaled[i] * math.sqrt(scale)
                B_scaled[i] = B_scaled[i] * math.sqrt(scale)
                
                # 验证归一化后的范数
                W_normalized = torch.mm(B_scaled[i], A_scaled[i])
                norm_normalized = torch.norm(W_normalized, p='fro').item()
                norms_after_normalize.append(norm_normalized)
                
                print(f"    Task {i}:")
                print(f"      Scale factor: {scale:.4f} (sqrt={math.sqrt(scale):.4f})")
                print(f"      After normalize: {norm_normalized:.4f}")
        
        # Step 3: 加权融合
        print(f"\n  [Debug] Step 3: Weighted Averaging (weights: {task_weights})")
        
        A_merged = sum(w * A for w, A in zip(task_weights, A_scaled))
        B_merged = sum(w * B for w, B in zip(task_weights, B_scaled))
        
        # 最终融合结果
        W_merged = torch.mm(B_merged, A_merged)
        norm_merged = torch.norm(W_merged, p='fro').item()
        
        print(f"  [Debug] Final merged norm: {norm_merged:.4f}")
        print(f"  [Debug] Ratio to original avg: {norm_merged/np.mean(original_norms)*100:.1f}%")
        
        # 融合 bias
        if B_biases[0] is not None:
            B_bias_merged = sum(w * b for w, b in zip(task_weights, B_biases))
        else:
            B_bias_merged = None
        
        return A_merged, B_merged, B_bias_merged
    
    def _prune_matrix(self, matrix: torch.Tensor, prune_ratio: float) -> torch.Tensor:
        """
        修剪矩阵：将后 prune_ratio% 的小参数置零
        
        Args:
            matrix: torch.Tensor, 要修剪的矩阵
            prune_ratio: float, 修剪比例
        
        Returns:
            torch.Tensor, 修剪后的矩阵
        """
        if prune_ratio <= 0:
            return matrix
        
        matrix_flat = matrix.abs().flatten()
        threshold = torch.quantile(matrix_flat, prune_ratio)
        
        mask = (matrix.abs() >= threshold).float()
        return matrix * mask
    
    def _compute_scaling_matrix(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        计算互补缩放矩阵 S（对角矩阵）
        
        公式：S^i = Σ|A[i,:]| / Σ|M_A[i,:] ⊙ A[i,:]|
        其中 M_A 是 A 的掩码（非零为1）
        
        Args:
            A: torch.Tensor, LoRA A 矩阵
            B: torch.Tensor, LoRA B 矩阵
        
        Returns:
            torch.Tensor, 缩放系数向量（对角矩阵的对角元素）
        """
        # 计算每行的绝对值和
        A_abs = A.abs()
        row_sums = A_abs.sum(dim=1)  # Σ|A[i,:]|
        
        # 计算掩码加权和
        mask = (A != 0).float()
        masked_sums = (mask * A_abs).sum(dim=1)  # Σ|M_A[i,:] ⊙ A[i,:]|
        
        # 计算缩放系数
        S = row_sums / (masked_sums + 1e-8)
        
        return S


class RobustMergeLoRAOptimized:
    """
    RobustMergeLoRA 优化版本
    
    新增功能：
    1. 范数恢复：融合后恢复原始平均范数
    2. 输出层范数恢复：输出层也恢复原始平均范数
    """
    
    def __init__(self, prune_ratio=0.2, restore_norm=True):
        """
        Args:
            prune_ratio: float, 修剪掉每层后 k% 的小参数
            restore_norm: bool, 是否在融合后恢复原始平均范数
        """
        self.prune_ratio = prune_ratio
        self.restore_norm = restore_norm
    
    def merge_actors(
        self, 
        checkpoint_paths: List[str], 
        task_weights: Optional[List[float]] = None,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        融合多个 LoRA Residual Actor（优化版本）
        
        Args:
            checkpoint_paths: List[str], 每个任务的checkpoint路径
            task_weights: Optional[List[float]], 每个任务的权重，默认均匀
            output_path: Optional[str], 保存融合结果的路径
        
        Returns:
            merged_params: Dict, 融合后的参数
        """
        n_tasks = len(checkpoint_paths)
        
        # 默认均匀权重
        if task_weights is None:
            task_weights = [1.0 / n_tasks] * n_tasks
        else:
            # 归一化权重
            total_weight = sum(task_weights)
            task_weights = [w / total_weight for w in task_weights]
        
        print(f"Merging {n_tasks} LoRA Residual Actors (Optimized Version)...")
        print(f"Task weights: {task_weights}")
        if self.restore_norm:
            print(f"Norm restoration: ENABLED")
        else:
            print(f"Norm restoration: DISABLED")
        
        # 1. 加载所有任务的参数
        all_task_params = []
        for i, path in enumerate(checkpoint_paths):
            print(f"Loading task {i} from {path}...")
            
            # Try to load as safetensors first, then fall back to torch.load
            try:
                if os.path.isdir(path) or (os.path.isfile(path) and path.endswith('.safetensors')):
                    checkpoint = _load_checkpoint_safetensors(path)
                else:
                    checkpoint = torch.load(path, map_location='cpu')
            except Exception as e:
                # Fall back to torch.load if safetensors loading fails
                print(f"  Warning: Failed to load as safetensors, trying torch.load: {e}")
                checkpoint = torch.load(path, map_location='cpu')
            
            # Extract LoRA parameters from checkpoint
            params = _load_lora_params_from_checkpoint(checkpoint)
            all_task_params.append(params)
        
        # 2. 融合 LoRA 层（使用优化版RobustMerge）
        merged_lora_layers = {}
        for layer_name in ['fc1', 'fc2', 'fc3']:
            print(f"Merging {layer_name}...")
            
            # 提取所有任务的 A 和 B 矩阵
            A_matrices = [p['lora_layers'][layer_name]['A'] for p in all_task_params]
            B_matrices = [p['lora_layers'][layer_name]['B'] for p in all_task_params]
            B_biases = [p['lora_layers'][layer_name]['B_bias'] for p in all_task_params]
            
            # 应用优化版RobustMerge
            A_merged, B_merged, B_bias_merged = self._robust_merge_lora_layer_optimized(
                A_matrices, B_matrices, B_biases, task_weights
            )
            
            merged_lora_layers[layer_name] = {
                'A': A_merged,
                'B': B_merged,
                'B_bias': B_bias_merged
            }
        
        # 3. 融合输出层（优化版本：考虑范数恢复）
        merged_output_layers = {}
        for output_name in ['fc_mean', 'fc_logstd']:
            print(f"Merging {output_name}...")
            
            weights = [p['output_layers'][output_name]['weight'] for p in all_task_params]
            biases = [p['output_layers'][output_name]['bias'] for p in all_task_params]
            
            # 记录原始范数
            if self.restore_norm:
                original_output_norms = [torch.norm(w, p='fro').item() for w in weights]
                avg_output_norm = np.mean(original_output_norms)
            
            merged_weight = sum(w * weight for w, weight in zip(task_weights, weights))
            merged_bias = sum(w * bias for w, bias in zip(task_weights, biases))
            
            # 恢复输出层的范数
            if self.restore_norm:
                merged_weight_norm = torch.norm(merged_weight, p='fro').item()
                if merged_weight_norm > 1e-6:
                    scale_factor = avg_output_norm / merged_weight_norm
                    merged_weight = merged_weight * scale_factor
                    print(f"  Output layer norm recovery: {merged_weight_norm:.4f} → {torch.norm(merged_weight, p='fro').item():.4f} (target: {avg_output_norm:.4f})")
            
            merged_output_layers[output_name] = {
                'weight': merged_weight,
                'bias': merged_bias
            }
        
        # 4. 组装最终参数
        merged_params = {
            'lora_layers': merged_lora_layers,
            'output_layers': merged_output_layers,
            'meta': all_task_params[0]['meta']  # 元信息从第一个任务复制
        }
        
        # 5. 保存（如果指定了路径）
        if output_path is not None:
            # Check if output_path is a directory (for safetensors) or file (for .pt)
            if os.path.isdir(output_path) or output_path.endswith('/') or (not os.path.exists(output_path) and not output_path.endswith('.pt')):
                # Save as safetensors format
                # Determine the actual directory path
                if os.path.isdir(output_path):
                    save_dir = output_path
                elif output_path.endswith('/'):
                    save_dir = output_path.rstrip('/')
                else:
                    # Create directory from output_path
                    save_dir = output_path if os.path.dirname(output_path) else output_path
                    os.makedirs(save_dir, exist_ok=True)
                
                _save_checkpoint_safetensors(
                    merged_params=merged_params,
                    output_path=save_dir,
                    task_weights=task_weights,
                    n_tasks=n_tasks,
                    prune_ratio=self.prune_ratio,
                    merge_method='robust_merge_optimized',
                    reference_checkpoint_path=checkpoint_paths[0] if checkpoint_paths else None
                )
            else:
                # Save as .pt format (backward compatibility)
                save_dict = {
                    'merged_params': merged_params,
                    'task_weights': task_weights,
                    'n_tasks': n_tasks,
                    'prune_ratio': self.prune_ratio,
                    'restore_norm': self.restore_norm,
                    'merge_method': 'robust_merge_optimized'
                }
                torch.save(save_dict, output_path)
                print(f"Saved merged model to {output_path}")
        
        return merged_params
    
    def _robust_merge_lora_layer_optimized(
        self, 
        A_matrices: List[torch.Tensor], 
        B_matrices: List[torch.Tensor],
        B_biases: List[Optional[torch.Tensor]],
        task_weights: List[float]
    ):
        """
        对单层的 LoRA 参数进行优化版RobustMerge
        包含范数恢复功能
        """
        n_tasks = len(A_matrices)
        
        print(f"\n  [Debug] Starting optimized merge for {n_tasks} tasks")
        
        # ===== 记录原始范数 =====
        original_norms = []
        for i in range(n_tasks):
            W_original = torch.mm(B_matrices[i], A_matrices[i])
            norm = torch.norm(W_original, p='fro').item()
            original_norms.append(norm)
            print(f"  [Debug] Task {i} - Original norm: {norm:.4f}")
        print(f"  [Debug] Original average norm: {np.mean(original_norms):.4f}")
        
        # Step 1: Pruning & Complementary Scaling
        A_scaled = []
        B_scaled = []
        
        print(f"\n  [Debug] Step 1: Pruning (ratio={self.prune_ratio}) & Complementary Scaling")
        
        for i in range(n_tasks):
            A = A_matrices[i].clone()
            B = B_matrices[i].clone()
            
            # 1.1 修剪前
            W_before_prune = torch.mm(B, A)
            norm_before_prune = torch.norm(W_before_prune, p='fro').item()
            
            # 1.1 修剪
            A_pruned = self._prune_matrix(A, self.prune_ratio)
            
            # 修剪后
            W_after_prune = torch.mm(B, A_pruned)
            norm_after_prune = torch.norm(W_after_prune, p='fro').item()
            
            # 1.2 互补缩放
            S = self._compute_scaling_matrix(A_pruned, B)
            
            # 1.3 应用缩放
            B_scaled_task = B * S.unsqueeze(0)
            
            # 缩放后
            W_after_scaling = torch.mm(B_scaled_task, A_pruned)
            norm_after_scaling = torch.norm(W_after_scaling, p='fro').item()
            
            print(f"    Task {i}:")
            print(f"      After Pruning:   {norm_after_prune:.4f} ({norm_after_prune/norm_before_prune*100:.1f}% of original)")
            print(f"      After Scaling:   {norm_after_scaling:.4f} ({norm_after_scaling/norm_before_prune*100:.1f}% of original)")
            print(f"      S matrix - min: {S.min().item():.4f}, max: {S.max().item():.4f}, mean: {S.mean().item():.4f}")
            
            A_scaled.append(A_pruned)
            B_scaled.append(B_scaled_task)
        
        # Step 2: Cross-Task Normalization
        print(f"\n  [Debug] Step 2: Cross-Task Normalization")
        
        # 计算缩放后的范数
        norms_before_normalize = []
        for i in range(n_tasks):
            W_effective = torch.mm(B_scaled[i], A_scaled[i])
            norm = torch.norm(W_effective, p='fro').item()
            norms_before_normalize.append(norm)
            print(f"    Task {i} norm before normalize: {norm:.4f}")
        
        # 归一化
        avg_norm = np.mean(norms_before_normalize)
        print(f"  [Debug] Target norm (average): {avg_norm:.4f}")
        
        norms_after_normalize = []
        for i in range(n_tasks):
            if norms_before_normalize[i] > 1e-6:
                scale = avg_norm / norms_before_normalize[i]
                A_scaled[i] = A_scaled[i] * math.sqrt(scale)
                B_scaled[i] = B_scaled[i] * math.sqrt(scale)
                
                # 验证归一化后的范数
                W_normalized = torch.mm(B_scaled[i], A_scaled[i])
                norm_normalized = torch.norm(W_normalized, p='fro').item()
                norms_after_normalize.append(norm_normalized)
                
                print(f"    Task {i}:")
                print(f"      Scale factor: {scale:.4f} (sqrt={math.sqrt(scale):.4f})")
                print(f"      After normalize: {norm_normalized:.4f}")
        
        # Step 3: 加权融合
        print(f"\n  [Debug] Step 3: Weighted Averaging (weights: {task_weights})")
        
        A_merged = sum(w * A for w, A in zip(task_weights, A_scaled))
        B_merged = sum(w * B for w, B in zip(task_weights, B_scaled))
        
        # 计算融合后的范数
        W_merged = torch.mm(B_merged, A_merged)
        norm_merged = torch.norm(W_merged, p='fro').item()
        
        # ✅ 优化：范数恢复（如果启用）
        if self.restore_norm:
            target_norm = np.mean(original_norms)  # 使用之前记录的原始范数
            
            if norm_merged > 1e-6:
                scale_factor = target_norm / norm_merged
                A_merged = A_merged * math.sqrt(scale_factor)
                B_merged = B_merged * math.sqrt(scale_factor)
                
                # 验证恢复后的范数
                W_restored = torch.mm(B_merged, A_merged)
                norm_restored = torch.norm(W_restored, p='fro').item()
                print(f"  [Debug] Norm recovery: {norm_merged:.4f} → {norm_restored:.4f} (target: {target_norm:.4f}, scale={scale_factor:.4f})")
                print(f"  [Debug] Recovery ratio: {norm_restored/target_norm*100:.1f}%")
            else:
                print(f"  [Debug] Warning: Merged norm too small ({norm_merged:.6f}), skipping norm recovery")
        else:
            print(f"  [Debug] Final merged norm: {norm_merged:.4f}")
            print(f"  [Debug] Ratio to original avg: {norm_merged/np.mean(original_norms)*100:.1f}%")
        
        # 融合 bias
        if B_biases[0] is not None:
            B_bias_merged = sum(w * b for w, b in zip(task_weights, B_biases))
        else:
            B_bias_merged = None
        
        return A_merged, B_merged, B_bias_merged
    
    def _prune_matrix(self, matrix: torch.Tensor, prune_ratio: float) -> torch.Tensor:
        """修剪矩阵：将后 prune_ratio% 的小参数置零"""
        if prune_ratio <= 0:
            return matrix
        
        matrix_flat = matrix.abs().flatten()
        threshold = torch.quantile(matrix_flat, prune_ratio)
        
        mask = (matrix.abs() >= threshold).float()
        return matrix * mask
    
    def _compute_scaling_matrix(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """计算互补缩放矩阵 S（对角矩阵）"""
        # 计算每行的绝对值和
        A_abs = A.abs()
        row_sums = A_abs.sum(dim=1)  # Σ|A[i,:]|
        
        # 计算掩码加权和
        mask = (A != 0).float()
        masked_sums = (mask * A_abs).sum(dim=1)  # Σ|M_A[i,:] ⊙ A[i,:]|
        
        # 计算缩放系数
        S = row_sums / (masked_sums + 1e-8)
        
        return S

