import collections
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple, Union

import gymnasium as gym
import jax
import numpy as np
from serl_launcher.data.dataset import Dataset, DatasetDict


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        include_next_actions: Optional[bool] = False,
        include_label: Optional[bool] = False,
        include_grasp_penalty: Optional[bool] = False,
        include_octo_embeddings: Optional[bool] = False,
        include_mc_returns: Optional[bool] = False,
        include_action_base: Optional[bool] = False,
        include_q_selector: Optional[bool] = False,
        q_selector_max_k: int = 16,
        q_selector_action_dim: int = -1,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )
        
        if include_mc_returns:
            dataset_dict['mc_returns'] = np.empty((capacity,), dtype=np.float32)
        
        if include_octo_embeddings:
            dataset_dict['embeddings'] = np.empty((capacity, 384), dtype=np.float32)
            dataset_dict['next_embeddings'] = np.empty((capacity, 384), dtype=np.float32)

        if include_next_actions:
            dataset_dict['next_actions'] = np.empty((capacity, *action_space.shape), dtype=action_space.dtype)
            dataset_dict['next_intvn'] = np.empty((capacity,), dtype=bool)

        if include_label:
            dataset_dict['labels'] = np.empty((capacity,), dtype=int)

        if include_grasp_penalty:
            dataset_dict['grasp_penalty'] = np.empty((capacity,), dtype=np.float32)

        # Residual RL: actions=a_taken(full), action_base=a_base(s), next_action_base=a_base(s')
        
        if include_action_base:
            dataset_dict["action_base"] = np.empty(
                (capacity, *action_space.shape), dtype=action_space.dtype
            )
            dataset_dict["next_action_base"] = np.empty(
                (capacity, *action_space.shape), dtype=action_space.dtype
            )

        self._include_q_selector = bool(include_q_selector)
        self._q_selector_max_k = int(q_selector_max_k)
        self._q_selector_action_dim = int(q_selector_action_dim)
        if self._include_q_selector:
            adim = (
                self._q_selector_action_dim
                if self._q_selector_action_dim > 0
                else int(np.prod(action_space.shape))
            )
            K = self._q_selector_max_k
            dataset_dict["selector_chosen_cand"] = np.empty(
                (capacity, adim), dtype=np.float32
            )
            dataset_dict["selector_next_cands"] = np.empty(
                (capacity, K, adim), dtype=np.float32
            )
            dataset_dict["selector_h_scores"] = np.empty(
                (capacity, K), dtype=np.float32
            )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        self._include_action_base = bool(include_action_base)

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        if self._include_action_base:
            if "action_base" not in data_dict or "next_action_base" not in data_dict:
                raise KeyError(
                    "ReplayBuffer(include_action_base=True) requires data_dict keys "
                    "'action_base' and 'next_action_base' with the same shape as actions."
                )
        if self._include_q_selector:
            for key in (
                "selector_chosen_cand",
                "selector_next_cands",
                "selector_h_scores",
            ):
                if key not in data_dict:
                    raise KeyError(
                        "ReplayBuffer(include_q_selector=True) requires data_dict keys "
                        f"{key!r}。"
                    )
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(
        self,
        queue_size: int = 2,
        sample_args: dict = {},
        device=None,
        batch_postprocess: Optional[Callable[[Any], Any]] = None,
    ):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                if batch_postprocess is not None:
                    data = batch_postprocess(data)
                queue.append(jax.device_put(data, device=device))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def download(self, from_idx: int, to_idx: int):
        indices = np.arange(from_idx, to_idx)
        data_dict = self.sample(batch_size=len(indices), indx=indices)
        return to_idx, data_dict

    def get_download_iterator(self):
        last_idx = 0
        while True:
            if last_idx >= self._size:
                raise RuntimeError(
                    f"last_idx {last_idx} >= self._size {self._size}")
            last_idx, batch = self.download(last_idx, self._size)
            yield batch
