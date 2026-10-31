# Modified by the ResMerge contributors, 2026-10-31; see real_robot/THIRD_PARTY.md.
"""
LoRA residual actor merge algorithms: RobustMergeLoRA (pruning, complementary scaling and
cross-task normalization), RobustMergeLoRAOptimized (norm restoration and optional adaptive
weights), DirectionalInterpolationMergeLoRA (direction-space interpolation), and
PerformanceAwareMergeLoRA (performance-based weighting). These implementations are independent.
Use quick_merge.py for the supported command-line workflow.
"""

import torch
import numpy as np
import os
from typing import List, Dict, Optional
import math


def get_actor_params(checkpoint):
    """Read a single-task export or a previously merged actor."""
    for key in ("params", "merged_params"):
        if key in checkpoint:
            return checkpoint[key]
    raise ValueError("Expected an actor export containing params or merged_params")


def prepare_merge_inputs(checkpoint_paths, task_weights=None):
    """Normalize convex weights and exclude zero-weight inputs before scale estimation."""
    paths = list(checkpoint_paths)
    weights = list(task_weights) if task_weights is not None else [1.0] * len(paths)
    if (not paths or len(weights) != len(paths)
            or any(not math.isfinite(w) or w < 0 for w in weights)
            or not any(w > 0 for w in weights)):
        raise ValueError("Provide one finite nonnegative weight per checkpoint, with a positive sum")
    active = [(path, float(weight)) for path, weight in zip(paths, weights) if weight > 0]
    # Scaling first avoids overflow when individually finite weights have a huge sum.
    scale = max(weight for _, weight in active)
    total = math.fsum(weight / scale for _, weight in active)
    return [path for path, _ in active], [weight / scale / total for _, weight in active]


def _single_input_params(path, output_path=None):
    """A single contributing expert needs no pruning, refactorization or norm restoration."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    params = get_actor_params(checkpoint)
    if output_path is not None:
        result = dict(checkpoint)
        result.pop("params", None)
        result.update(merged_params=params, task_weights=[1.0], n_inputs=1,
                      source_checkpoints=[str(path)], merge_method="single_input")
        torch.save(result, output_path)
    return params


class DirectionalInterpolationMergeLoRA:
    """
    Norm-preserving directional interpolation: normalize effective matrices W=BA into
    directions, interpolate those directions, then restore the weighted average magnitude.
    """
    
    def __init__(self):
        """
        Initialize directional interpolation.
        """
        pass
    
    def merge_actors(
        self, 
        checkpoint_paths: List[str], 
        task_weights: Optional[List[float]] = None,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Merge actor checkpoints with directional interpolation. task_weights defaults to
        uniform; optionally save to output_path and return merged parameters.
        """
        checkpoint_paths, task_weights = prepare_merge_inputs(checkpoint_paths, task_weights)
        n_tasks = len(checkpoint_paths)
        if n_tasks == 1:
            return _single_input_params(checkpoint_paths[0], output_path)
        
        print(f"Merging {n_tasks} LoRA Residual Actors using Directional Interpolation...")
        print(f"Task weights: {task_weights}")
        
        
        all_task_params = []
        for i, path in enumerate(checkpoint_paths):
            print(f"Loading task {i} from {path}...")
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            all_task_params.append(get_actor_params(checkpoint))
        
        
        merged_lora_layers = {}
        for layer_name in ['fc1', 'fc2', 'fc3']:
            print(f"\nMerging {layer_name} using directional interpolation...")
            
            
            A_matrices = [p['lora_layers'][layer_name]['A'] for p in all_task_params]
            B_matrices = [p['lora_layers'][layer_name]['B'] for p in all_task_params]
            B_biases = [p['lora_layers'][layer_name]['B_bias'] for p in all_task_params]
            
            
            A_merged, B_merged, B_bias_merged = self._directional_interpolation_merge_lora_layer(
                A_matrices, B_matrices, B_biases, task_weights,
                min_direction_consistency=0.3  
            )
            
            merged_lora_layers[layer_name] = {
                'A': A_merged,
                'B': B_merged,
                'B_bias': B_bias_merged
            }
        
        
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
        
        
        merged_params = {
            'lora_layers': merged_lora_layers,
            'output_layers': merged_output_layers,
            'meta': all_task_params[0]['meta']  
        }
        
        
        if output_path is not None:
            save_dict = {
                'merged_params': merged_params,
                'task_weights': task_weights,
                'n_tasks': n_tasks,
                'merge_method': 'directional_interpolation'
            }
            torch.save(save_dict, output_path)
            print(f"\n✅ Saved merged model to {output_path}")
        
        return merged_params
    
    def _directional_interpolation_merge_lora_layer(
        self, 
        A_matrices: List[torch.Tensor], 
        B_matrices: List[torch.Tensor],
        B_biases: List[Optional[torch.Tensor]],
        task_weights: List[float],
        min_direction_consistency: float = 0.3
    ):
        """
        Merge one LoRA layer: compute effective W=BA, check pairwise directions and fall back
        to weighted factor averaging below min_direction_consistency. Otherwise interpolate
        normalized W directions, restore their weighted mean norm and recover A/B by SVD.
        """
        n_tasks = len(A_matrices)
        epsilon = 1e-8
        
        print(f"  [Directional Interpolation] Merging {n_tasks} tasks")
        
        
        W_matrices = []
        W_norms = []
        
        for i in range(n_tasks):
            W = torch.mm(B_matrices[i], A_matrices[i])  # [out_dim, in_dim]
            W_norm = torch.norm(W, p='fro').item()
            W_matrices.append(W)
            W_norms.append(W_norm)
            print(f"    Task {i}: W norm = {W_norm:.6f}")
        
        avg_norm = np.mean(W_norms)
        print(f"    Average W norm: {avg_norm:.6f}")
        
        
        if n_tasks == 2:
            W_0 = W_matrices[0]
            W_1 = W_matrices[1]
            W_0_norm = W_norms[0]
            W_1_norm = W_norms[1]
            
            if W_0_norm > epsilon and W_1_norm > epsilon:
                u_0 = W_0 / W_0_norm
                u_1 = W_1 / W_1_norm
                direction_consistency = torch.nn.functional.cosine_similarity(
                    u_0.flatten().unsqueeze(0), u_1.flatten().unsqueeze(0)
                ).item()
                print(f"    Direction consistency: {direction_consistency:.6f}")
                
                
                if direction_consistency < min_direction_consistency:
                    print(f"    Warning: Direction consistency ({direction_consistency:.6f}) < threshold ({min_direction_consistency:.6f})")
                    print(f"    Falling back to weighted average merge")
                    
                    A_merged = sum(w * A for w, A in zip(task_weights, A_matrices))
                    B_merged = sum(w * B for w, B in zip(task_weights, B_matrices))
                    
                    
                    W_avg = torch.mm(B_merged, A_merged)
                    W_avg_norm = torch.norm(W_avg, p='fro').item()
                    if W_avg_norm > epsilon:
                        scale = avg_norm / W_avg_norm
                        A_merged = A_merged * math.sqrt(scale)
                        B_merged = B_merged * math.sqrt(scale)
                    
                    
                    if B_biases[0] is not None:
                        B_bias_merged = sum(w * b for w, b in zip(task_weights, B_biases))
                    else:
                        B_bias_merged = None
                    
                    return A_merged, B_merged, B_bias_merged
        
        
        W_unit_directions = []
        for i in range(n_tasks):
            W = W_matrices[i]
            W_norm = W_norms[i]
            
            if W_norm < epsilon:
                
                W_unit = torch.zeros_like(W)
            else:
                W_unit = W / W_norm
            
            W_unit_directions.append(W_unit)
        
        
        # u_merge = normalize(sum(w_i * u_i))
        W_unit_merged = sum(w * u for w, u in zip(task_weights, W_unit_directions))
        W_unit_merged_norm = torch.norm(W_unit_merged, p='fro').item()
        
        if W_unit_merged_norm < epsilon:
            
            print(f"    Warning: Merged direction is zero, using first task direction")
            W_unit_merged = W_unit_directions[0]
            W_unit_merged_norm = 1.0
        
        W_unit_merged = W_unit_merged / W_unit_merged_norm
        print(f"    Merged unit direction norm: {torch.norm(W_unit_merged, p='fro').item():.6f}")
        
        
        # mag = sum(w_i * ||W_i||_F)
        merged_magnitude = sum(w * norm for w, norm in zip(task_weights, W_norms))
        W_merged = merged_magnitude * W_unit_merged
        W_merged_norm = torch.norm(W_merged, p='fro').item()
        
        print(f"    Merged magnitude: {merged_magnitude:.6f}")
        print(f"    Final W_merged norm: {W_merged_norm:.6f}")


        rank = A_matrices[0].shape[0]
        out_dim, in_dim = W_merged.shape
        
        
        U, S, Vt = torch.linalg.svd(W_merged, full_matrices=False)
        
        
        U_rank = U[:, :rank]  # [out_dim, rank]
        S_rank = S[:rank]  # [rank]
        Vt_rank = Vt[:rank, :]  # [rank, in_dim]


        sqrt_S = torch.sqrt(S_rank + epsilon)
        B_merged = U_rank @ torch.diag(sqrt_S)  # [out_dim, rank]
        A_merged = torch.diag(sqrt_S) @ Vt_rank  # [rank, in_dim]
        
        
        W_verify = torch.mm(B_merged, A_merged)
        verify_error = torch.norm(W_verify - W_merged, p='fro').item()
        print(f"    Decomposition error (SVD): {verify_error:.8f}")
        
        
        if B_biases[0] is not None:
            B_bias_merged = sum(w * b for w, b in zip(task_weights, B_biases))
        else:
            B_bias_merged = None
        
        return A_merged, B_merged, B_bias_merged

class RobustMergeLoRA:
    """
    Merge residual actors using pruning and complementary scaling inspired by RobustMerge.
    """
    
    def __init__(self, prune_ratio=0.2):
        """
        prune_ratio is the fraction of smallest-magnitude parameters pruned per layer.
        """
        self.prune_ratio = prune_ratio
    
    def merge_actors(
        self, 
        checkpoint_paths: List[str], 
        task_weights: Optional[List[float]] = None,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Merge checkpoint_paths with optional task_weights (uniform by default), optionally save
        to output_path, and return merged parameters.
        """
        checkpoint_paths, task_weights = prepare_merge_inputs(checkpoint_paths, task_weights)
        n_tasks = len(checkpoint_paths)
        if n_tasks == 1:
            return _single_input_params(checkpoint_paths[0], output_path)
        
        print(f"Merging {n_tasks} LoRA Residual Actors...")
        print(f"Task weights: {task_weights}")
        
        
        all_task_params = []
        for i, path in enumerate(checkpoint_paths):
            print(f"Loading task {i} from {path}...")
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            all_task_params.append(get_actor_params(checkpoint))
        
        
        merged_lora_layers = {}
        for layer_name in ['fc1', 'fc2', 'fc3']:
            print(f"Merging {layer_name}...")
            
            
            A_matrices = [p['lora_layers'][layer_name]['A'] for p in all_task_params]
            B_matrices = [p['lora_layers'][layer_name]['B'] for p in all_task_params]
            B_biases = [p['lora_layers'][layer_name]['B_bias'] for p in all_task_params]
            
            
            A_merged, B_merged, B_bias_merged = self._robust_merge_lora_layer(
                A_matrices, B_matrices, B_biases, task_weights
            )
            
            merged_lora_layers[layer_name] = {
                'A': A_merged,
                'B': B_merged,
                'B_bias': B_bias_merged
            }
        
        
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
        
        
        merged_params = {
            'lora_layers': merged_lora_layers,
            'output_layers': merged_output_layers,
            'meta': all_task_params[0]['meta']  
        }
        
        
        if output_path is not None:
            save_dict = {
                'merged_params': merged_params,
                'task_weights': task_weights,
                'n_tasks': n_tasks,
                'prune_ratio': self.prune_ratio
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
        Apply RobustMerge to a LoRA layer and report intermediate norms.
        """
        n_tasks = len(A_matrices)
        
        print(f"\n  [Debug] Starting merge for {n_tasks} tasks")
        
        
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
            
            
            W_before_prune = torch.mm(B, A)
            norm_before_prune = torch.norm(W_before_prune, p='fro').item()
            
            
            A_pruned = self._prune_matrix(A, self.prune_ratio)
            
            
            W_after_prune = torch.mm(B, A_pruned)
            norm_after_prune = torch.norm(W_after_prune, p='fro').item()
            
            
            S = self._compute_scaling_matrix(A, A_pruned)
            
            
            B_scaled_task = B * S.unsqueeze(0)
            
            
            W_after_scaling = torch.mm(B_scaled_task, A_pruned)
            norm_after_scaling = torch.norm(W_after_scaling, p='fro').item()
            
            print(f"    Task {i}:")
            print(f"      After Pruning:   {norm_after_prune:.4f} ({norm_after_prune/max(norm_before_prune, 1e-8)*100:.1f}% of original)")
            print(f"      After Scaling:   {norm_after_scaling:.4f} ({norm_after_scaling/max(norm_before_prune, 1e-8)*100:.1f}% of original)")
            print(f"      S matrix - min: {S.min().item():.4f}, max: {S.max().item():.4f}, mean: {S.mean().item():.4f}")
            
            A_scaled.append(A_pruned)
            B_scaled.append(B_scaled_task)
        
        # Step 2: Cross-Task Normalization
        print(f"\n  [Debug] Step 2: Cross-Task Normalization")
        
        
        norms_before_normalize = []
        for i in range(n_tasks):
            W_effective = torch.mm(B_scaled[i], A_scaled[i])
            norm = torch.norm(W_effective, p='fro').item()
            norms_before_normalize.append(norm)
            print(f"    Task {i} norm before normalize: {norm:.4f}")
        
        
        avg_norm = np.mean(norms_before_normalize)
        print(f"  [Debug] Target norm (average): {avg_norm:.4f}")
        
        norms_after_normalize = []
        for i in range(n_tasks):
            if norms_before_normalize[i] > 1e-6:
                scale = avg_norm / norms_before_normalize[i]
                A_scaled[i] = A_scaled[i] * math.sqrt(scale)
                B_scaled[i] = B_scaled[i] * math.sqrt(scale)
                
                
                W_normalized = torch.mm(B_scaled[i], A_scaled[i])
                norm_normalized = torch.norm(W_normalized, p='fro').item()
                norms_after_normalize.append(norm_normalized)
                
                print(f"    Task {i}:")
                print(f"      Scale factor: {scale:.4f} (sqrt={math.sqrt(scale):.4f})")
                print(f"      After normalize: {norm_normalized:.4f}")
        
        
        print(f"\n  [Debug] Step 3: Weighted Averaging (weights: {task_weights})")
        
        A_merged = sum(w * A for w, A in zip(task_weights, A_scaled))
        B_merged = sum(w * B for w, B in zip(task_weights, B_scaled))
        
        
        W_merged = torch.mm(B_merged, A_merged)
        norm_merged = torch.norm(W_merged, p='fro').item()


        if B_biases[0] is not None:
            B_bias_merged = sum(w * b for w, b in zip(task_weights, B_biases))
        else:
            B_bias_merged = None
        
        return A_merged, B_merged, B_bias_merged
    
    def _prune_matrix(self, matrix: torch.Tensor, prune_ratio: float) -> torch.Tensor:
        """
        Zero the smallest-magnitude prune_ratio fraction of matrix entries.
        """
        if prune_ratio <= 0:
            return matrix
        
        matrix_flat = matrix.abs().flatten()
        threshold = torch.quantile(matrix_flat, prune_ratio)
        
        mask = (matrix.abs() >= threshold).float()
        return matrix * mask
    
    def _compute_scaling_matrix(self, original_A: torch.Tensor,
                                pruned_A: torch.Tensor) -> torch.Tensor:
        """Restore each retained row's absolute mass using its pre-pruning value."""
        original_sums = original_A.abs().sum(dim=1)
        retained_sums = pruned_A.abs().sum(dim=1)
        # A fully pruned row has no contribution to restore; keep its B column unchanged.
        return torch.where(retained_sums > 0,
                           original_sums / retained_sums.clamp_min(1e-8),
                           torch.ones_like(retained_sums))

class PerformanceAwareMergeLoRA:
    """
    Choose merge weights from model success rates across tasks, using balanced, per-task,
    generalization or task-specific weighting.
    """
    
    def __init__(self, prune_ratio=0.2, restore_norm=True, weighting_strategy='balanced'):
        """
        Configure pruning, norm restoration and the weighting strategy: balanced,
        best_per_task, generalization or task_specific.
        """
        self.prune_ratio = prune_ratio
        self.restore_norm = restore_norm
        self.weighting_strategy = weighting_strategy
    
    def merge_actors(
        self,
        checkpoint_paths: List[str],
        performance_matrix: np.ndarray,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Merge checkpoints using a performance_matrix shaped (n_models,n_tasks), whose (i,j)
        entry is model i success on task j. Optionally save and return merged parameters.
        """
        n_models = len(checkpoint_paths)
        n_tasks = performance_matrix.shape[1]
        
        if performance_matrix.shape[0] != n_models:
            raise ValueError(f"Performance matrix shape mismatch: {performance_matrix.shape[0]} != {n_models}")
        
        print(f"="*80)
        print(f"Performance-Aware Merge")
        print(f"="*80)
        print(f"Models: {n_models}, Tasks: {n_tasks}")
        print(f"Weighting strategy: {self.weighting_strategy}")
        print(f"\nPerformance Matrix:")
        print(f"  {'Task':<10}", end="")
        for j in range(n_tasks):
            print(f"Task {j:<8}", end="")
        print()
        for i in range(n_models):
            print(f"  Model {i:<7}", end="")
            for j in range(n_tasks):
                print(f"{performance_matrix[i, j]:>8.3f}", end="")
            print()
        
        
        task_weights_list = self._compute_task_weights(performance_matrix)
        
        print(f"\nComputed Task Weights:")
        for task_id, weights in enumerate(task_weights_list):
            print(f"  Task {task_id}: {[f'{w:.3f}' for w in weights]}")
        
        
        all_model_params = []
        for i, path in enumerate(checkpoint_paths):
            print(f"\nLoading model {i} from {path}...")
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            all_model_params.append(get_actor_params(checkpoint))


        if self.weighting_strategy == 'task_specific':
            
            print("\n⚠️  Task-specific strategy requires returning multiple models.")
            print("   Using averaged weights instead.")
            
            avg_weights = np.mean(task_weights_list, axis=0)
            task_weights = avg_weights.tolist()
        else:
            
            if len(task_weights_list) == 1:
                task_weights = task_weights_list[0].tolist()
            else:
                
                avg_weights = np.mean(task_weights_list, axis=0)
                task_weights = avg_weights.tolist()
        
        print(f"\nFinal merge weights: {[f'{w:.3f}' for w in task_weights]}")
        
        
        merger = RobustMergeLoRAOptimized(
            prune_ratio=self.prune_ratio,
            restore_norm=self.restore_norm,
            adaptive_weights=False  
        )
        
        merged_params = merger.merge_actors(
            checkpoint_paths=checkpoint_paths,
            task_weights=task_weights,
            output_path=output_path
        )
        
        
        if output_path is not None:
            checkpoint = torch.load(output_path, map_location='cpu')
            checkpoint['performance_matrix'] = performance_matrix
            checkpoint['task_weights'] = task_weights
            checkpoint['weighting_strategy'] = self.weighting_strategy
            torch.save(checkpoint, output_path)
        
        return merged_params
    
    def _compute_task_weights(self, performance_matrix: np.ndarray) -> List[np.ndarray]:
        """
        Compute a list of per-task model weight arrays from the performance matrix.
        """
        n_models, n_tasks = performance_matrix.shape
        task_weights_list = []
        
        if self.weighting_strategy == 'balanced':
            
            
            model_scores = performance_matrix.mean(axis=1)
            
            model_scores = model_scores - model_scores.min() + 1e-6
            weights = model_scores / model_scores.sum()
            
            task_weights_list = [weights] * n_tasks
            
        elif self.weighting_strategy == 'best_per_task':
            
            
            for task_id in range(n_tasks):
                task_performances = performance_matrix[:, task_id]
                
                exp_scores = np.exp(task_performances * 5)  
                weights = exp_scores / exp_scores.sum()
                task_weights_list.append(weights)
                
        elif self.weighting_strategy == 'generalization':
            
            
            mean_perf = performance_matrix.mean(axis=1)
            std_perf = performance_matrix.std(axis=1)
            
            generalization_scores = mean_perf / (std_perf + 1e-6)
            
            generalization_scores = generalization_scores - generalization_scores.min() + 1e-6
            weights = generalization_scores / generalization_scores.sum()
            
            task_weights_list = [weights] * n_tasks
            
        elif self.weighting_strategy == 'task_specific':
            
            
            for task_id in range(n_tasks):
                task_performances = performance_matrix[:, task_id]
                
                exp_scores = np.exp(task_performances * 3)  
                weights = exp_scores / exp_scores.sum()
                task_weights_list.append(weights)
                
        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")
        
        return task_weights_list

class RobustMergeLoRAOptimized:
    """
    RobustMerge variant adding norm restoration, optional performance-based adaptive weighting
    and output-head norm restoration.
    """
    
    def __init__(self, prune_ratio=0.2, restore_norm=True, adaptive_weights=False):
        """
        Configure pruning, post-merge norm restoration and optional adaptive weights requiring
        performance values.
        """
        self.prune_ratio = prune_ratio
        self.restore_norm = restore_norm
        self.adaptive_weights = adaptive_weights
    
    def merge_actors(
        self, 
        checkpoint_paths: List[str], 
        task_weights: Optional[List[float]] = None,
        output_path: Optional[str] = None,
        task_performances: Optional[List[float]] = None
    ) -> Dict:
        """
        Merge actor checkpoints with optional weights/performance scores and output path;
        return the merged parameters.
        """
        n_tasks = len(checkpoint_paths)
        if not n_tasks:
            raise ValueError("Provide at least one actor checkpoint")
        if self.adaptive_weights and task_performances is not None:
            if len(task_performances) != n_tasks:
                print(f"Warning: task_performances length ({len(task_performances)}) != n_tasks ({n_tasks}), using uniform weights")
                task_weights = [1.0 / n_tasks] * n_tasks
            else:
                
                performances = np.asarray(task_performances, dtype=np.float64)
                if not np.isfinite(performances).all():
                    raise ValueError("task_performances must be finite")
                
                performances = performances - performances.min() + 1e-6
                task_weights = (performances / performances.sum()).tolist()
                print(f"Adaptive weights based on performance: {task_weights}")
        checkpoint_paths, task_weights = prepare_merge_inputs(checkpoint_paths, task_weights)
        n_tasks = len(checkpoint_paths)
        if n_tasks == 1:
            return _single_input_params(checkpoint_paths[0], output_path)
        
        print(f"Merging {n_tasks} LoRA Residual Actors (Optimized Version)...")
        print(f"Task weights: {task_weights}")
        if self.restore_norm:
            print(f"Norm restoration: ENABLED")
        else:
            print(f"Norm restoration: DISABLED")
        
        
        all_task_params = []
        for i, path in enumerate(checkpoint_paths):
            print(f"Loading task {i} from {path}...")
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            all_task_params.append(get_actor_params(checkpoint))
        
        
        merged_lora_layers = {}
        for layer_name in ['fc1', 'fc2', 'fc3']:
            print(f"Merging {layer_name}...")
            
            
            A_matrices = [p['lora_layers'][layer_name]['A'] for p in all_task_params]
            B_matrices = [p['lora_layers'][layer_name]['B'] for p in all_task_params]
            B_biases = [p['lora_layers'][layer_name]['B_bias'] for p in all_task_params]
            
            
            A_merged, B_merged, B_bias_merged = self._robust_merge_lora_layer_optimized(
                A_matrices, B_matrices, B_biases, task_weights
            )
            
            merged_lora_layers[layer_name] = {
                'A': A_merged,
                'B': B_merged,
                'B_bias': B_bias_merged
            }
        
        
        merged_output_layers = {}
        for output_name in ['fc_mean', 'fc_logstd']:
            print(f"Merging {output_name}...")
            
            weights = [p['output_layers'][output_name]['weight'] for p in all_task_params]
            biases = [p['output_layers'][output_name]['bias'] for p in all_task_params]
            
            
            if self.restore_norm:
                original_output_norms = [torch.norm(w, p='fro').item() for w in weights]
                avg_output_norm = np.mean(original_output_norms)
            
            merged_weight = sum(w * weight for w, weight in zip(task_weights, weights))
            merged_bias = sum(w * bias for w, bias in zip(task_weights, biases))
            
            
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
        
        
        merged_params = {
            'lora_layers': merged_lora_layers,
            'output_layers': merged_output_layers,
            'meta': all_task_params[0]['meta']  
        }
        
        
        if output_path is not None:
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
        Merge one low-rank layer with the optimized RobustMerge procedure and norm restoration.
        """
        n_tasks = len(A_matrices)
        
        print(f"\n  [Debug] Starting optimized merge for {n_tasks} tasks")
        
        
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
            
            
            W_before_prune = torch.mm(B, A)
            norm_before_prune = torch.norm(W_before_prune, p='fro').item()
            
            
            A_pruned = self._prune_matrix(A, self.prune_ratio)
            
            
            W_after_prune = torch.mm(B, A_pruned)
            norm_after_prune = torch.norm(W_after_prune, p='fro').item()
            
            
            S = self._compute_scaling_matrix(A, A_pruned)
            
            
            B_scaled_task = B * S.unsqueeze(0)
            
            
            W_after_scaling = torch.mm(B_scaled_task, A_pruned)
            norm_after_scaling = torch.norm(W_after_scaling, p='fro').item()
            
            print(f"    Task {i}:")
            print(f"      After Pruning:   {norm_after_prune:.4f} ({norm_after_prune/max(norm_before_prune, 1e-8)*100:.1f}% of original)")
            print(f"      After Scaling:   {norm_after_scaling:.4f} ({norm_after_scaling/max(norm_before_prune, 1e-8)*100:.1f}% of original)")
            print(f"      S matrix - min: {S.min().item():.4f}, max: {S.max().item():.4f}, mean: {S.mean().item():.4f}")
            
            A_scaled.append(A_pruned)
            B_scaled.append(B_scaled_task)
        
        # Step 2: Cross-Task Normalization
        print(f"\n  [Debug] Step 2: Cross-Task Normalization")
        
        
        norms_before_normalize = []
        for i in range(n_tasks):
            W_effective = torch.mm(B_scaled[i], A_scaled[i])
            norm = torch.norm(W_effective, p='fro').item()
            norms_before_normalize.append(norm)
            print(f"    Task {i} norm before normalize: {norm:.4f}")
        
        
        avg_norm = np.mean(norms_before_normalize)
        print(f"  [Debug] Target norm (average): {avg_norm:.4f}")
        
        norms_after_normalize = []
        for i in range(n_tasks):
            if norms_before_normalize[i] > 1e-6:
                scale = avg_norm / norms_before_normalize[i]
                A_scaled[i] = A_scaled[i] * math.sqrt(scale)
                B_scaled[i] = B_scaled[i] * math.sqrt(scale)
                
                
                W_normalized = torch.mm(B_scaled[i], A_scaled[i])
                norm_normalized = torch.norm(W_normalized, p='fro').item()
                norms_after_normalize.append(norm_normalized)
                
                print(f"    Task {i}:")
                print(f"      Scale factor: {scale:.4f} (sqrt={math.sqrt(scale):.4f})")
                print(f"      After normalize: {norm_normalized:.4f}")
        
        
        print(f"\n  [Debug] Step 3: Weighted Averaging (weights: {task_weights})")
        
        A_merged = sum(w * A for w, A in zip(task_weights, A_scaled))
        B_merged = sum(w * B for w, B in zip(task_weights, B_scaled))
        
        
        W_merged = torch.mm(B_merged, A_merged)
        norm_merged = torch.norm(W_merged, p='fro').item()
        
        
        if self.restore_norm:
            target_norm = np.mean(original_norms)  
            
            if norm_merged > 1e-6:
                scale_factor = target_norm / norm_merged
                A_merged = A_merged * math.sqrt(scale_factor)
                B_merged = B_merged * math.sqrt(scale_factor)
                
                
                W_restored = torch.mm(B_merged, A_merged)
                norm_restored = torch.norm(W_restored, p='fro').item()
                print(f"  [Debug] Norm recovery: {norm_merged:.4f} → {norm_restored:.4f} (target: {target_norm:.4f}, scale={scale_factor:.4f})")
                print(f"  [Debug] Recovery ratio: {norm_restored/target_norm*100:.1f}%")
            else:
                print(f"  [Debug] Warning: Merged norm too small ({norm_merged:.6f}), skipping norm recovery")
        else:
            print(f"  [Debug] Final merged norm: {norm_merged:.4f}")
            print(f"  [Debug] Ratio to original avg: {norm_merged/np.mean(original_norms)*100:.1f}%")
        
        
        if B_biases[0] is not None:
            B_bias_merged = sum(w * b for w, b in zip(task_weights, B_biases))
        else:
            B_bias_merged = None
        
        return A_merged, B_merged, B_bias_merged
    
    def _prune_matrix(self, matrix: torch.Tensor, prune_ratio: float) -> torch.Tensor:
        """
        Zero the smallest-magnitude prune_ratio fraction of matrix entries.
        """
        if prune_ratio <= 0:
            return matrix
        
        matrix_flat = matrix.abs().flatten()
        threshold = torch.quantile(matrix_flat, prune_ratio)
        
        mask = (matrix.abs() >= threshold).float()
        return matrix * mask
    
    def _compute_scaling_matrix(self, original_A: torch.Tensor,
                                pruned_A: torch.Tensor) -> torch.Tensor:
        """Restore each retained row's absolute mass using its pre-pruning value."""
        original_sums = original_A.abs().sum(dim=1)
        retained_sums = pruned_A.abs().sum(dim=1)
        # A fully pruned row has no contribution to restore; keep its B column unchanged.
        return torch.where(retained_sums > 0,
                           original_sums / retained_sums.clamp_min(1e-8),
                           torch.ones_like(retained_sums))
    

    def merge_actors_hierarchical(
        self,
        checkpoint_paths: List[str],
        task_weights: Optional[List[float]] = None,
        layers_to_merge: List[str] = ['fc1', 'fc2'],  
        layers_to_keep_expert: List[str] = ['fc3'],   
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Hierarchical merge: merge layers_to_merge while preserving task experts for
        layers_to_keep_expert.
        """
        n_tasks = len(checkpoint_paths)
        
        
        if task_weights is None:
            task_weights = [1.0 / n_tasks] * n_tasks
        else:
            total_weight = sum(task_weights)
            task_weights = [w / total_weight for w in task_weights]
        
        print(f"="*70)
        print(f"Hierarchical merge mode")
        print(f"="*70)
        print(f"Merging {n_tasks} tasks")
        print(f"Task weights: {task_weights}")
        print(f"Merged layers: {layers_to_merge}")
        print(f"Expert layers retained: {layers_to_keep_expert}")
        print(f"="*70)
        
        
        all_task_params = []
        task_ids = []
        for i, path in enumerate(checkpoint_paths):
            print(f"Loading task {i} from {path}...")
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            all_task_params.append(get_actor_params(checkpoint))
            task_ids.append(checkpoint.get('task_id', i))
        
        
        merged_lora_layers = {}
        for layer_name in layers_to_merge:
            if layer_name not in ['fc1', 'fc2', 'fc3']:
                print(f"Skipping unknown layer: {layer_name}")
                continue
            
            print(f"Merging {layer_name}...")
            
            
            A_matrices = [p['lora_layers'][layer_name]['A'] for p in all_task_params]
            B_matrices = [p['lora_layers'][layer_name]['B'] for p in all_task_params]
            B_biases = [p['lora_layers'][layer_name]['B_bias'] for p in all_task_params]
            
            
            A_merged, B_merged, B_bias_merged = self._robust_merge_lora_layer(
                A_matrices, B_matrices, B_biases, task_weights
            )
            
            merged_lora_layers[layer_name] = {
                'A': A_merged,
                'B': B_merged,
                'B_bias': B_bias_merged,
                'type': 'merged'
            }
        
        
        expert_layers = {}
        for layer_name in layers_to_keep_expert:
            if layer_name not in ['fc1', 'fc2', 'fc3']:
                print(f"Skipping unknown layer: {layer_name}")
                continue
            
            print(f"Retaining {layer_name}: {n_tasks} experts...")
            
            experts = []
            for i, params in enumerate(all_task_params):
                expert_params = {
                    'A': params['lora_layers'][layer_name]['A'].clone(),
                    'B': params['lora_layers'][layer_name]['B'].clone(),
                    'B_bias': params['lora_layers'][layer_name]['B_bias'].clone() 
                            if params['lora_layers'][layer_name]['B_bias'] is not None else None,
                    'task_id': task_ids[i]
                }
                experts.append(expert_params)
            
            expert_layers[layer_name] = {
                'experts': experts,
                'n_experts': n_tasks,
                'type': 'expert'
            }
        
        
        merged_output_layers = {}
        output_layer_names = ['fc_mean', 'fc_logstd']
        
        
        for output_name in output_layer_names:
            if output_name in layers_to_keep_expert:
                print(f"Retaining {output_name}: {n_tasks} experts...")
                experts = []
                for i, params in enumerate(all_task_params):
                    expert_params = {
                        'weight': params['output_layers'][output_name]['weight'].clone(),
                        'bias': params['output_layers'][output_name]['bias'].clone(),
                        'task_id': task_ids[i]
                    }
                    experts.append(expert_params)
                
                expert_layers[output_name] = {
                    'experts': experts,
                    'n_experts': n_tasks,
                    'type': 'expert'
                }
            else:
                print(f"Merging {output_name}...")
                weights = [p['output_layers'][output_name]['weight'] for p in all_task_params]
                biases = [p['output_layers'][output_name]['bias'] for p in all_task_params]
                
                merged_weight = sum(w * weight for w, weight in zip(task_weights, weights))
                merged_bias = sum(w * bias for w, bias in zip(task_weights, biases))
                
                merged_output_layers[output_name] = {
                    'weight': merged_weight,
                    'bias': merged_bias,
                    'type': 'merged'
                }
        
        
        hierarchical_params = {
            'merged_lora_layers': merged_lora_layers,    
            'expert_layers': expert_layers,              
            'merged_output_layers': merged_output_layers, 
            'meta': {
                'rank': all_task_params[0]['meta']['rank'],
                'n_tasks': n_tasks,
                'task_ids': task_ids,
                'layers_to_merge': layers_to_merge,
                'layers_to_keep_expert': layers_to_keep_expert,
                'architecture': 'hierarchical'
            }
        }
        
        
        if output_path is not None:
            save_dict = {
                'hierarchical_params': hierarchical_params,
                'task_weights': task_weights,
                'prune_ratio': self.prune_ratio
            }
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            torch.save(save_dict, output_path)
            print(f"Hierarchical merged model saved to: {output_path}")
        
        return hierarchical_params

def merge_task_actors(
    checkpoint_dir: Optional[str] = None,
    task_ids: Optional[List[int]] = None,
    output_path: str = "merged_actor.pt",
    task_weights: Optional[List[float]] = None,
    prune_ratio: float = 0.2,
    checkpoint_paths: Optional[List[str]] = None,
    method: str = "robust",
    overwrite: bool = False,
):
    """Merge explicitly selected exports using the same self-contained format as quick_merge."""
    if checkpoint_dir is not None or task_ids is not None or not checkpoint_paths:
        raise ValueError(
            "Directory/task-id discovery is no longer supported: export filenames identify "
            "runs and steps. Pass checkpoint_paths=[task_A.pt, task_B.pt, ...], or run "
            "quick_merge.py --checkpoint_paths task_A.pt task_B.pt --output_path merged_actor.pt."
        )
    from quick_merge import merge_exports

    result = merge_exports(
        checkpoint_paths, output_path, task_weights=task_weights,
        prune_ratio=prune_ratio, method=method, overwrite=overwrite,
    )
    return result["merged_params"]

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge explicit actor exports; delegates to quick_merge.py.")
    parser.add_argument('--checkpoint_paths', nargs='+', help='Explicit single-task or merged .pt exports')
    parser.add_argument('--checkpoint_dir', help='Obsolete; use --checkpoint_paths')
    parser.add_argument('--task_ids', type=int, nargs='+', help='Obsolete; use --checkpoint_paths')
    parser.add_argument('--output_path', type=str, default='merged_actor.pt')
    parser.add_argument('--prune_ratio', type=float, default=0.2)
    parser.add_argument('--task_weights', type=float, nargs='+')
    parser.add_argument('--method', choices=('robust', 'optimized', 'directional'), default='robust')
    parser.add_argument('--overwrite', action='store_true')
    
    args = parser.parse_args()
    if args.checkpoint_dir is not None or args.task_ids is not None or not args.checkpoint_paths:
        parser.error(
            "Use --checkpoint_paths task_A.pt task_B.pt (also supported by quick_merge.py). "
            "The obsolete --checkpoint_dir/--task_ids interface cannot select run/step exports."
        )

    merged_params = merge_task_actors(
        checkpoint_paths=args.checkpoint_paths,
        output_path=args.output_path,
        prune_ratio=args.prune_ratio,
        task_weights=args.task_weights,
        method=args.method,
        overwrite=args.overwrite,
    )
    
    print("Merge completed!")
