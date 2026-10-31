"""
Prioritized replay using a NumPy sum-tree without changing parent buffer storage. Samples
include max-normalized is_weights and leaf _indices. update_priorities writes
(abs(TD)+eps)^alpha and tracks maximum priority. Residual action-base fields pass through to
the parent. DataStore callers must hold the shared lock for insert/sample/update_priorities;
buffers do not lock internally.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import gymnasium as gym
import numpy as np
from flax.core import frozen_dict

from serl_launcher.data.dataset import DatasetDict, _sample
from serl_launcher.data.memory_efficient_replay_buffer import (
    MemoryEfficientReplayBuffer,
)
from serl_launcher.data.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class _SumTree:
    """
    Capacity-aligned sum-tree with 2*next_pow2(capacity)-1 entries. Leaves store already
    exponentiated priorities; internal nodes sum their children. The buffer applies alpha
    outside this tree.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        
        self.tree_capacity = 1
        while self.tree_capacity < self.capacity:
            self.tree_capacity <<= 1
        self.leaf_offset = self.tree_capacity - 1
        # tree[i] = sum of leaves in subtree of node i
        self.tree = np.zeros(2 * self.tree_capacity - 1, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, leaf_index: int, value: float) -> None:
        idx = self.leaf_offset + int(leaf_index)
        delta = float(value) - self.tree[idx]
        self.tree[idx] = float(value)
        idx = (idx - 1) // 2
        while idx >= 0:
            self.tree[idx] += delta
            if idx == 0:
                break
            idx = (idx - 1) // 2

    def update_batch(self, leaf_indices: np.ndarray, values: np.ndarray) -> None:
        
        for li, v in zip(leaf_indices, values):
            self.update(int(li), float(v))

    def get_leaf(self, s: float) -> Tuple[int, float]:
        """Find a leaf for cumulative weight s in [0,total); return (leaf_index, priority)."""
        idx = 0
        while idx < self.leaf_offset:
            left = 2 * idx + 1
            right = left + 1
            left_sum = self.tree[left]
            if s <= left_sum:
                idx = left
            else:
                s -= left_sum
                idx = right
        leaf_index = idx - self.leaf_offset
        return int(leaf_index), float(self.tree[idx])

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stratified sampling: draw once from each of B equal segments of [0,total)."""
        total = self.total()
        if total <= 0.0:
            
            indices = rng.integers(0, max(self.capacity, 1), size=batch_size)
            ps = np.zeros(batch_size, dtype=np.float64)
            return indices.astype(np.int64), ps
        seg = total / float(batch_size)
        indices = np.empty(batch_size, dtype=np.int64)
        ps = np.empty(batch_size, dtype=np.float64)
        # uniform within each segment
        offsets = rng.uniform(0.0, 1.0, size=batch_size) * seg
        bases = np.arange(batch_size, dtype=np.float64) * seg
        samples = bases + offsets
        for i, s in enumerate(samples):
            
            s = min(float(s), total - 1e-9)
            li, p = self.get_leaf(s)
            indices[i] = li
            ps[i] = p
        return indices, ps

    def min_nonzero(self) -> float:
        """
        Scan leaves for the approximate minimum nonzero priority used to bound importance
        weights.
        """
        leaves = self.tree[self.leaf_offset : self.leaf_offset + self.capacity]
        
        positive = leaves[leaves > 0]
        if positive.size == 0:
            return 1.0
        return float(positive.min())


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class _PERMixin:
    """
    Subclasses must call _init_per(capacity, alpha, beta, eps) after parent initialization.
    """

    def _init_per(
        self,
        capacity: int,
        alpha: float,
        beta: float,
        eps: float,
        max_priority_init: float = 1.0,
    ) -> None:
        self._per_alpha = float(alpha)
        self._per_beta = float(beta)
        self._per_eps = float(eps)
        self._sum_tree = _SumTree(capacity)
        self._max_priority = float(max_priority_init)
        
        self._per_rng = np.random.default_rng(0)

    def per_seed(self, seed: int) -> None:
        self._per_rng = np.random.default_rng(int(seed))

    def _per_priority_from(self, td_abs: float) -> float:
        return (abs(float(td_abs)) + self._per_eps) ** self._per_alpha

    def _per_record_insert(self, leaf_index: int) -> None:
        """
        Insert new transitions with the current maximum priority to prioritize initial
        sampling.
        """
        p = self._max_priority
        
        self._sum_tree.update(leaf_index, p)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        td_errors = np.asarray(td_errors, dtype=np.float64).reshape(-1)
        assert indices.shape == td_errors.shape, (indices.shape, td_errors.shape)
        priorities_alpha = (np.abs(td_errors) + self._per_eps) ** self._per_alpha
        self._sum_tree.update_batch(indices, priorities_alpha)
        cur_max = float(priorities_alpha.max())
        if cur_max > self._max_priority:
            self._max_priority = cur_max

    def _per_sample_indices_and_weights(
        self, batch_size: int, beta: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        beta_used = float(self._per_beta if beta is None else beta)
        indices, ps = self._sum_tree.sample_batch(batch_size, self._per_rng)
        total = self._sum_tree.total()
        if total <= 0.0 or self.__len__() == 0:
            
            return indices, np.ones(batch_size, dtype=np.float32)
        n = max(int(self.__len__()), 1)
        
        probs = np.maximum(ps, 1e-12) / total
        weights = (1.0 / (n * probs)) ** beta_used
        weights = weights / max(weights.max(), 1e-12)
        return indices, weights.astype(np.float32)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class PrioritizedReplayBuffer(_PERMixin, ReplayBuffer):
    """Residual-compatible ReplayBuffer storage with an additional priority sum-tree."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        *,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_eps: float = 1e-6,
        include_action_base: bool = False,
        include_q_selector: bool = False,
        q_selector_max_k: int = 16,
        next_observation_space: Optional[gym.Space] = None,
        **kwargs,
    ):
        ReplayBuffer.__init__(
            self,
            observation_space,
            action_space,
            capacity,
            next_observation_space=next_observation_space,
            include_action_base=include_action_base,
            include_q_selector=include_q_selector,
            q_selector_max_k=q_selector_max_k,
            **kwargs,
        )
        self._init_per(capacity, per_alpha, per_beta, per_eps)

    def insert(self, data_dict: DatasetDict):
        leaf = int(self._insert_index)
        super().insert(data_dict)
        
        self._per_record_insert(leaf)

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        *,
        beta: Optional[float] = None,
    ):
        if indx is None:
            indx, is_weights = self._per_sample_indices_and_weights(batch_size, beta=beta)
        else:
            indx = np.asarray(indx, dtype=np.int64).reshape(-1)
            
            is_weights = np.ones(batch_size, dtype=np.float32)
        batch = ReplayBuffer.sample(self, batch_size, keys=keys, indx=indx).unfreeze()
        batch["is_weights"] = is_weights
        batch["_indices"] = indx.astype(np.int64)
        return frozen_dict.freeze(batch)


# ---------------------------------------------------------------------------
# Pixel-stacked buffer
# ---------------------------------------------------------------------------
class MemoryEfficientPrioritizedReplayBuffer(_PERMixin, MemoryEfficientReplayBuffer):
    """
    Prioritized replay for stacked pixels. Recheck parent frame/episode validity after priority
    sampling and resample invalid indices in place, approximately preserving the priority
    distribution.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        *,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_eps: float = 1e-6,
        include_action_base: bool = False,
        include_q_selector: bool = False,
        q_selector_max_k: int = 16,
        **kwargs,
    ):
        MemoryEfficientReplayBuffer.__init__(
            self,
            observation_space,
            action_space,
            capacity,
            pixel_keys=pixel_keys,
            include_action_base=include_action_base,
            include_q_selector=include_q_selector,
            q_selector_max_k=q_selector_max_k,
            **kwargs,
        )
        self._init_per(capacity, per_alpha, per_beta, per_eps)

    def insert(self, data_dict: DatasetDict):
        
        
        
        
        before = int(self._insert_index)
        super().insert(data_dict)
        after = int(self._insert_index)
        
        
        if after >= before:
            inserted = list(range(before, after))
        else:
            # wrap around
            inserted = list(range(before, self._capacity)) + list(range(0, after))
        for li in inserted:
            self._per_record_insert(li)

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        pack_obs: bool = False,
        *,
        beta: Optional[float] = None,
    ):
        if indx is not None:
            
            raise NotImplementedError("MemoryEfficientPrioritizedReplayBuffer does not support explicit indx")
        indices, is_weights = self._per_sample_indices_and_weights(batch_size, beta=beta)
        
        
        max_resample_rounds = 8
        for _ in range(max_resample_rounds):
            bad = ~self._is_correct_index[indices]
            if not bad.any():
                break
            n_bad = int(bad.sum())
            new_idx, new_w = self._per_sample_indices_and_weights(n_bad, beta=beta)
            indices[bad] = new_idx
            is_weights[bad] = new_w
        
        bad = ~self._is_correct_index[indices]
        if bad.any():
            for i in np.where(bad)[0]:
                while not self._is_correct_index[indices[i]]:
                    indices[i] = self._per_rng.integers(len(self))
                is_weights[i] = 1.0

        
        if keys is None:
            keys = self.dataset_dict.keys()
        else:
            assert "observations" in keys
        keys = list(keys)
        keys.remove("observations")
        
        batch = ReplayBuffer.sample(self, batch_size, keys, indices).unfreeze()

        obs_keys = self.dataset_dict["observations"].keys()
        obs_keys = list(obs_keys)
        for pixel_key in self.pixel_keys:
            obs_keys.remove(pixel_key)

        batch["observations"] = {}
        for k in obs_keys:
            batch["observations"][k] = _sample(self.dataset_dict["observations"][k], indices)

        for pixel_key in self.pixel_keys:
            obs_pixels = self.dataset_dict["observations"][pixel_key]
            obs_pixels = np.lib.stride_tricks.sliding_window_view(
                obs_pixels, self._num_stack + 1, axis=0
            )
            obs_pixels = obs_pixels[indices - self._num_stack]
            obs_pixels = obs_pixels.transpose((0, 4, 1, 2, 3))
            if pack_obs:
                batch["observations"][pixel_key] = obs_pixels
            else:
                batch["observations"][pixel_key] = obs_pixels[:, :-1, ...]
                if "next_observations" in keys:
                    batch["next_observations"][pixel_key] = obs_pixels[:, 1:, ...]

        batch["is_weights"] = is_weights
        batch["_indices"] = indices.astype(np.int64)
        return frozen_dict.freeze(batch)
