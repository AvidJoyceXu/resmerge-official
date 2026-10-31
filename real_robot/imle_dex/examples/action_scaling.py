"""LeapHand 22D action scaling: arm 0-5 unchanged, hand 6-21 in [-1,1] for training."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

LEAP_HAND_PHYS_MIN = np.array(
    [
        2.09459,
        2.82759,
        2.63559,
        2.77559,
        2.09459,
        2.82759,
        2.63559,
        2.77559,
        2.09459,
        2.82759,
        2.63559,
        2.77559,
        2.79259,
        2.67159,
        1.94159,
        1.80159,
    ],
    dtype=np.float32,
)

LEAP_HAND_PHYS_MAX = np.array(
    [
        4.18859,
        5.37159,
        5.02659,
        5.18359,
        4.18859,
        5.37159,
        5.02659,
        5.18359,
        4.18859,
        5.37159,
        5.02659,
        5.18359,
        5.23559,
        5.58459,
        5.04159,
        5.02159,
    ],
    dtype=np.float32,
)

_SPAN = LEAP_HAND_PHYS_MAX - LEAP_HAND_PHYS_MIN


def normalize_leap_hand_physical(hand16: np.ndarray) -> np.ndarray:
    x = np.asarray(hand16, dtype=np.float32).reshape(16).copy()
    x = np.clip(x, LEAP_HAND_PHYS_MIN, LEAP_HAND_PHYS_MAX)
    return (2.0 * (x - LEAP_HAND_PHYS_MIN) / _SPAN - 1.0).astype(np.float32)


def denormalize_leap_hand_normalized(hand16: np.ndarray) -> np.ndarray:
    t = np.asarray(hand16, dtype=np.float32).reshape(16).copy()
    t = np.clip(t, -1.0, 1.0)
    return (LEAP_HAND_PHYS_MIN + (t + 1.0) * 0.5 * _SPAN).astype(np.float32)


def normalize_action_22d(action: np.ndarray) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if a.shape[0] < 22:
        return a
    a[6:22] = normalize_leap_hand_physical(a[6:22])
    return a


def denormalize_action_22d(action: np.ndarray) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if a.shape[0] < 22:
        return a
    a[6:22] = denormalize_leap_hand_normalized(a[6:22])
    return a


def leap_hand_looks_physical(hand16: np.ndarray) -> bool:
    h = np.asarray(hand16, dtype=np.float64).reshape(-1)
    if h.size != 16:
        return False
    return bool(np.any(h > 1.05) or np.any(h < -1.05))


def maybe_normalize_leap_hand_actions_in_transition(transition: dict) -> None:
    if "actions" not in transition:
        return
    a = np.asarray(transition["actions"])
    if a.size < 22:
        return
    if leap_hand_looks_physical(a[6:22]):
        transition["actions"] = normalize_action_22d(a)


def actions_for_replay_buffer(actions: np.ndarray) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float32).reshape(-1).copy()
    if a.size >= 22 and leap_hand_looks_physical(a[6:22]):
        return normalize_action_22d(a)
    return a.astype(np.float32)


class LeapHandNormalizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        low = env.action_space.low.astype(np.float32).copy()
        high = env.action_space.high.astype(np.float32).copy()
        low[6:22] = -1.0
        high[6:22] = 1.0
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def action(self, act):
        a = np.asarray(act, dtype=np.float32).copy()
        if a.shape[-1] >= 22:
            a[..., 6:22] = denormalize_leap_hand_normalized(a[..., 6:22])
        return a

    def step(self, action, action_base=None):
        """
        Preserve the step(..., action_base) wrapper chain for IMLEActionBaseWrapper and
        AugmentDictObsWrapper.
        """
        act = self.action(action)
        if action_base is None:
            return self.env.step(act)
        return self.env.step(act, action_base)
