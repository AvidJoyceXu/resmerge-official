#!/usr/bin/env python3
# Modified by the ResMerge contributors, 2026-10-31; see real_robot/THIRD_PARTY.md.
"""Merge compatible task LoRA exports and retain LayerNorm for direct actor evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import torch

from merge_lora_actors import (
    RobustMergeLoRA, RobustMergeLoRAOptimized, DirectionalInterpolationMergeLoRA,
    get_actor_params,
)


def validate_export(checkpoint, path):
    """Check the flat actor parameter schema used for merging."""
    if not isinstance(checkpoint, dict):
        raise ValueError(f'{path}: expected an exported or merged actor dictionary')
    params = get_actor_params(checkpoint)
    meta = params['meta']
    rank, obs_dim, action_dim = (int(meta[k]) for k in ('rank', 'obs_dim', 'action_dim'))
    if min(rank, obs_dim, action_dim) <= 0:
        raise ValueError(f'{path}: dimensions must be positive')
    hidden = params['lora_layers']['fc1']['B'].shape[0]
    for layer, input_dim in [('fc1', obs_dim), ('fc2', hidden), ('fc3', hidden)]:
        for key, shape in [('A', (rank, input_dim)), ('B', (hidden, rank)), ('B_bias', (hidden,))]:
            tensor = params['lora_layers'][layer][key]
            if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                raise ValueError(f'{path}: invalid {layer}.{key}, expected finite {shape}')
    for head in ('fc_mean', 'fc_logstd'):
        for key, shape in [('weight', (action_dim, hidden)), ('bias', (action_dim,))]:
            tensor = params['output_layers'][head][key]
            if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                raise ValueError(f'{path}: invalid {head}.{key}, expected finite {shape}')
    extras = checkpoint.get('_jax_extras')
    if extras is None:
        raise ValueError(f'{path}: missing _jax_extras; re-export the original task actor with lora_agent')
    for layer in ('enc_LN', 'trunk_LN_0', 'trunk_LN_1'):
        for key in ('scale', 'bias'):
            tensor = extras[layer][key]
            if tuple(tensor.shape) != (hidden,) or not torch.isfinite(tensor).all():
                raise ValueError(f'{path}: invalid LayerNorm {layer}.{key}')
    return rank, obs_dim, action_dim, hidden


def merge_exports(checkpoint_paths, output_path, task_weights=None, prune_ratio=0.2,
                  method='robust', overwrite=False):
    """Merge task actors and bundle their weighted LayerNorm parameters."""
    paths = [Path(p).expanduser().resolve() for p in checkpoint_paths]
    output = Path(output_path).expanduser().resolve()
    if len(paths) < 2:
        raise ValueError('Provide at least two task exports')
    if output in paths:
        raise ValueError('Output must not overwrite an input task export')
    if output.exists() and not overwrite:
        raise FileExistsError(f'{output} exists; use --overwrite to replace it')
    if not math.isfinite(prune_ratio) or not 0 <= prune_ratio < 1:
        raise ValueError('prune_ratio must be in [0,1)')
    weights = list(task_weights) if task_weights is not None else [1.0] * len(paths)
    if len(weights) != len(paths) or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError('Provide one finite nonnegative weight per task, with a positive sum')
    weights = [w / sum(weights) for w in weights]
    checkpoints = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    schemas = [validate_export(c, p) for c, p in zip(checkpoints, paths)]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError(f'Incompatible rank/observation/action/hidden dimensions: {schemas}')
    # Compare base-policy and residual-scale metadata when present.
    shared_metadata = {}
    for key in ('base_checkpoint', 'residual_alpha'):
        values = [c.get('additional_info', {}).get(key) for c in checkpoints]
        supplied = [v for v in values if v is not None and v != '']
        if key == 'residual_alpha' and any(not math.isfinite(v) or v <= 0 for v in supplied):
            raise ValueError('residual_alpha metadata must be finite and positive; re-export the actor')
        if supplied and any(v != supplied[0] for v in supplied[1:]):
            raise ValueError(f'Conflicting {key} metadata: {supplied}')
        if len(supplied) == len(checkpoints):
            shared_metadata[key] = supplied[0]
    if method == 'robust':
        merger = RobustMergeLoRA(prune_ratio=prune_ratio)
    elif method == 'optimized':
        merger = RobustMergeLoRAOptimized(prune_ratio=prune_ratio, restore_norm=True)
    elif method == 'directional':
        merger = DirectionalInterpolationMergeLoRA()
    else:
        raise ValueError(f'Unknown merge method: {method}')
    merged_params = merger.merge_actors([str(p) for p in paths], task_weights=weights)
    extras = {
        layer: {key: sum(w * c['_jax_extras'][layer][key] for w, c in zip(weights, checkpoints))
                for key in ('scale', 'bias')}
        for layer in ('enc_LN', 'trunk_LN_0', 'trunk_LN_1')
    }
    validate_export({'params': merged_params, '_jax_extras': extras}, 'merged output')
    # Weights apply to the input experts, including any already consolidated expert.
    # Preserve original export lineage without silently changing these weights.
    lineage = list(dict.fromkeys(
        source for path, checkpoint in zip(paths, checkpoints)
        for source in checkpoint.get('source_exports', checkpoint.get('source_checkpoints', [str(path)]))
    ))
    result = dict(merged_params=merged_params, _jax_extras=extras, task_weights=weights,
                  n_inputs=len(paths), n_tasks=len(lineage), method=method, prune_ratio=prune_ratio,
                  source_checkpoints=[str(p) for p in paths], source_exports=lineage,
                  additional_info=shared_metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(f'Saved {method} merge of {len(paths)} actors to {output}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint_paths', nargs='+', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--task_weights', type=float, nargs='+')
    parser.add_argument('--prune_ratio', type=float, default=0.2)
    parser.add_argument('--method', choices=['robust', 'optimized', 'directional'], default='robust')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    merge_exports(**vars(args))


if __name__ == '__main__':
    main()
