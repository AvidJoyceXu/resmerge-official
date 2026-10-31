from threading import Lock
from typing import Union, Iterable

import gymnasium as gym
import jax
import numpy as np
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.data.memory_efficient_replay_buffer import (
    MemoryEfficientReplayBuffer,
)
from serl_launcher.data.prioritized_replay_buffer import (
    MemoryEfficientPrioritizedReplayBuffer,
    PrioritizedReplayBuffer,
)

from agentlace.data.data_store import DataStoreBase


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        include_action_base: bool = False,
        include_q_selector: bool = False,
        q_selector_max_k: int = 16,
        q_selector_action_dim: int = -1,
    ):
        ReplayBuffer.__init__(
            self,
            observation_space,
            action_space,
            capacity,
            include_action_base=include_action_base,
            include_q_selector=include_q_selector,
            q_selector_max_k=q_selector_max_k,
            q_selector_action_dim=q_selector_action_dim,
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


class PrioritizedReplayBufferDataStore(PrioritizedReplayBuffer, DataStoreBase):
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
        q_selector_action_dim: int = -1,
        **kwargs,
    ):
        rb_kwargs = dict(kwargs)
        if int(q_selector_action_dim) >= 0:
            rb_kwargs["q_selector_action_dim"] = int(q_selector_action_dim)
        PrioritizedReplayBuffer.__init__(
            self,
            observation_space,
            action_space,
            capacity,
            per_alpha=per_alpha,
            per_beta=per_beta,
            per_eps=per_eps,
            include_action_base=include_action_base,
            include_q_selector=include_q_selector,
            q_selector_max_k=q_selector_max_k,
            **rb_kwargs,
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    def insert(self, *args, **kwargs):
        with self._lock:
            super(PrioritizedReplayBufferDataStore, self).insert(*args, **kwargs)

    def sample(self, *args, **kwargs):
        with self._lock:
            return super(PrioritizedReplayBufferDataStore, self).sample(*args, **kwargs)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        with self._lock:
            super(PrioritizedReplayBufferDataStore, self).update_priorities(
                indices, td_errors
            )

    def latest_data_id(self):
        return self._insert_index

    def get_latest_data(self, from_id: int):
        raise NotImplementedError


class MemoryEfficientPrioritizedReplayBufferDataStore(
    MemoryEfficientPrioritizedReplayBuffer, DataStoreBase
):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        *,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_eps: float = 1e-6,
        include_action_base: bool = False,
        include_q_selector: bool = False,
        q_selector_max_k: int = 16,
        q_selector_action_dim: int = -1,
        **kwargs,
    ):
        rb_kwargs = dict(kwargs)
        if int(q_selector_action_dim) >= 0:
            rb_kwargs["q_selector_action_dim"] = int(q_selector_action_dim)
        MemoryEfficientPrioritizedReplayBuffer.__init__(
            self,
            observation_space,
            action_space,
            capacity,
            pixel_keys=tuple(image_keys),
            per_alpha=per_alpha,
            per_beta=per_beta,
            per_eps=per_eps,
            include_action_base=include_action_base,
            include_q_selector=include_q_selector,
            q_selector_max_k=q_selector_max_k,
            **rb_kwargs,
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    def insert(self, *args, **kwargs):
        with self._lock:
            super(MemoryEfficientPrioritizedReplayBufferDataStore, self).insert(
                *args, **kwargs
            )

    def sample(self, *args, **kwargs):
        with self._lock:
            return super(MemoryEfficientPrioritizedReplayBufferDataStore, self).sample(
                *args, **kwargs
            )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        with self._lock:
            super(
                MemoryEfficientPrioritizedReplayBufferDataStore, self
            ).update_priorities(indices, td_errors)

    def latest_data_id(self):
        return self._insert_index

    def get_latest_data(self, from_id: int):
        raise NotImplementedError


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        **kwargs,
    ):
        MemoryEfficientReplayBuffer.__init__(
            self, observation_space, action_space, capacity, pixel_keys=image_keys, **kwargs
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(MemoryEfficientReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(MemoryEfficientReplayBufferDataStore, self).sample(
                *args, **kwargs
            )

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


def populate_data_store(
    data_store: DataStoreBase,
    demos_path: str,
):
    """
    Utility function to populate demonstrations data into data_store.
    :return data_store
    """
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                data_store.insert(transition)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store


def populate_data_store_with_z_axis_only(
    data_store: DataStoreBase,
    demos_path: str,
):
    """
    Utility function to populate demonstrations data into data_store.
    This will remove the x and y cartesian coordinates from the state.
    :return data_store
    """
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                tmp = deepcopy(transition)
                tmp["observations"]["state"] = np.concatenate(
                    (
                        tmp["observations"]["state"][:, :4],
                        tmp["observations"]["state"][:, 6][None, ...],
                        tmp["observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                tmp["next_observations"]["state"] = np.concatenate(
                    (
                        tmp["next_observations"]["state"][:, :4],
                        tmp["next_observations"]["state"][:, 6][None, ...],
                        tmp["next_observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                data_store.insert(tmp)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store
