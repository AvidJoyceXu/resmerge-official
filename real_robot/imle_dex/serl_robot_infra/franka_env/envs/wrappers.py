import threading
import time
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict
import copy
from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
import requests
from scipy.spatial.transform import Rotation as R
from franka_env.envs.franka_env import FrankaEnv
from typing import Optional

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def apply_leap_reset_pose(
    leap_node,
    pose: np.ndarray,
    *,
    pulse_steps: int = 25,
    pulse_dt: float = 0.01,
    max_delta_per_step: Optional[float] = None,
    leap_joint_low: Optional[np.ndarray] = None,
    leap_joint_high: Optional[np.ndarray] = None,
) -> None:
    """
    Send a fixed LEAP joint target repeatedly (like main.py hold loop) so goals are
    not lost on the wire and servos have time to move.
    """
    if leap_node is None or pose is None:
        return
    q = np.asarray(pose, dtype=np.float32).reshape(16)
    if leap_joint_low is not None and leap_joint_high is not None:
        lo = np.asarray(leap_joint_low, dtype=np.float32).reshape(16)
        hi = np.asarray(leap_joint_high, dtype=np.float32).reshape(16)
        q = np.clip(q, lo, hi)
    max_delta = None if max_delta_per_step is None else float(max_delta_per_step)
    if max_delta is not None and max_delta <= 0:
        max_delta = None

    # Start from current hardware pose if available; fallback to direct target.
    curr = None
    try:
        curr = np.asarray(leap_node.read_pos(), dtype=np.float32).reshape(16)
    except Exception:
        curr = None
    cmd = q.copy() if curr is None else curr.copy()
    start_cmd = cmd.copy()
    print(
        "[LEAP RESET] start->target with smoothing "
        f"(max_delta_per_step={max_delta}, pulse_steps={pulse_steps}, pulse_dt={pulse_dt})"
    )
    print(
        "[LEAP RESET] start[:4]="
        + np.array2string(start_cmd[:4], precision=3, separator=", ")
        + " target[:4]="
        + np.array2string(q[:4], precision=3, separator=", ")
    )

    last_err: Optional[Exception] = None
    for _ in range(pulse_steps):
        try:
            if max_delta is not None:
                delta = q - cmd
                cmd = cmd + np.clip(delta, -max_delta, max_delta)
                if leap_joint_low is not None and leap_joint_high is not None:
                    lo = np.asarray(leap_joint_low, dtype=np.float32).reshape(16)
                    hi = np.asarray(leap_joint_high, dtype=np.float32).reshape(16)
                    cmd = np.clip(cmd, lo, hi)
                leap_node.set_leap(cmd.tolist())
            else:
                leap_node.set_leap(q.tolist())
        except Exception as e:
            last_err = e
            break
        time.sleep(pulse_dt)
    if last_err is not None:
        print(f"[LEAP] apply_leap_reset_pose set_leap failed: {last_err}")
        return
    # Read once at the end for observability.
    try:
        end_pos = np.asarray(leap_node.read_pos(), dtype=np.float32).reshape(16)
        end_err = np.max(np.abs(end_pos - q))
        print(
            "[LEAP RESET] done. end[:4]="
            + np.array2string(end_pos[:4], precision=3, separator=", ")
            + f" max_abs_err_to_target={end_err:.3f}"
        )
    except Exception:
        pass

class ZOnlyWrapper(gym.ObservationWrapper):
    """
    Removal of X and Y coordinates
    """

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space["state"] = spaces.Box(-np.inf, np.inf, shape=(14,))

    def observation(self, observation):
        observation["state"] = np.concatenate(
            (
                observation["state"][:4],
                np.array(observation["state"][6])[..., None],
                observation["state"][10:],
            ),
            axis=-1,
        )
        return observation

    def step(self, action, action_base=None):
        if action_base is None:
            obs, rew, done, trunc, info = self.env.step(action)
        else:
            obs, rew, done, trunc, info = self.env.step(action, action_base)
        return self.observation(obs), rew, done, trunc, info

class ZOnlyNoFTWrapper(gym.ObservationWrapper):
    """
    Removal of X and Y coordinates and force torque sensor readings
    """

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space["state"] = spaces.Box(-np.inf, np.inf, shape=(9,))

    def observation(self, observation):
        observation["state"] = np.concatenate(
            (
                np.array(observation["state"][0])[..., None],  # gripper
                np.array(observation["state"][6])[..., None],  # z
                np.array(observation["state"][9])[..., None],  # rz
                observation["state"][-6:],  # vel
            ),
            axis=-1,
        )
        return observation

class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation

    def step(self, action, action_base=None):
        if action_base is None:
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            obs, reward, terminated, truncated, info = self.env.step(
                action, action_base
            )
        return self.observation(obs), reward, terminated, truncated, info

class Quat2R2Wrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotation matrix
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation

class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["left/tcp_pose"]
        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        tcp_pose = observation["state"]["right/tcp_pose"]
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, left_button_success: bool = True):
        super().__init__(env)

        self.expert = SpaceMouseExpert()
        self.left, self.right = False, False
        self.action_indices = action_indices
        self.left_button_success = bool(left_button_success)

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Merge SpaceMouse arm actions with the policy hand action, preserving action[6:22]. When
        left_button_success is enabled, the left button terminates successfully and supplies a
        reward if absent. Otherwise it only records info['left'], allowing keyboard labels to
        decide termination.
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = buttons[0], buttons[-1]
        intervened = False

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.action_indices is not None:
            filtered_expert_a = np.zeros(6, dtype=np.float32)
            for idx in self.action_indices:
                if idx < 6:
                    filtered_expert_a[idx] = expert_a[idx]
            expert_a = filtered_expert_a

        if intervened:
            # Preserve policy's leap dims (action[6:]) if action space is extended
            if len(action) > 6:
                expert_a = np.concatenate([expert_a[:6], action[6:]])
            return expert_a, True

        return action, False

    def step(self, action, action_base=None):

        new_action, replaced = self.action(action)

        if action_base is None:
            obs, rew, done, truncated, info = self.env.step(new_action)
        else:
            obs, rew, done, truncated, info = self.env.step(new_action, action_base)
        if self.left_button_success:
            done = bool(done or self.left)
        info = dict(info)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        if self.left_button_success and self.left:
            info["succeed"] = True
            if not rew:
                rew = True
        return obs, rew, done, truncated, info

class KeyboardSuccessWrapper(gym.Wrapper):
    """
    Keyboard episode labels applied on the next env.step. Space sets
    done/succeed/keyboard_success and supplies reward=True if absent. f sets done and
    keyboard_failure, clears succeed, and leaves reward unchanged. Actions are not modified.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._lock = threading.Lock()
        self._space_pending = False
        self._fail_pending = False
        self._listener = None
        self._start_listener()

    def _on_press(self, key) -> None:
        try:
            from pynput import keyboard

            if key == keyboard.Key.space:
                with self._lock:
                    self._space_pending = True
                return
            ch = (getattr(key, "char", None) or "").lower()
            if ch == "f":
                with self._lock:
                    self._fail_pending = True
        except Exception:
            pass

    def _start_listener(self) -> None:
        from pynput import keyboard

        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def close(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            try:
                self._listener.join(timeout=1.0)
            except Exception:
                pass
            self._listener = None
        self.env.close()

    def reset(self, **kwargs):
        with self._lock:
            self._space_pending = False
            self._fail_pending = False
        return self.env.reset(**kwargs)

    def clear_pending(self) -> None:
        """Discard pending success/failure keyboard events."""
        with self._lock:
            self._space_pending = False
            self._fail_pending = False

    def clear_pending_space(self) -> None:
        """Backward-compatible alias that clears both pending success and failure events."""
        self.clear_pending()

    def step(self, action, action_base=None):
        if action_base is None:
            obs, rew, done, truncated, info = self.env.step(action)
        else:
            obs, rew, done, truncated, info = self.env.step(action, action_base)
        info = dict(info)
        with self._lock:
            if self._space_pending:
                
                self._space_pending = False
                self._fail_pending = False
                info["keyboard_success"] = 1
                info["succeed"] = True
                done = True
                if not rew:
                    rew = True
            elif self._fail_pending:
                self._fail_pending = False
                info["keyboard_failure"] = 1
                info["succeed"] = False
                done = True
        return obs, rew, done, truncated, info

class WaitWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.wait = False

    def reset(self, **kwargs):
        if self.wait:
            input("Press Enter to continue...")
        obs, info = self.env.reset(**kwargs)
        self.wait = False
        return obs, info
    
    def step(self, action, action_base=None):
        if action_base is None:
            obs, rew, done, truncated, info = self.env.step(action)
        else:
            obs, rew, done, truncated, info = self.env.step(action, action_base)
        if rew:
            self.wait = True
        return obs, rew, done, truncated, info
    
class USBResetWrapper(gym.Wrapper, FrankaEnv):
    def __init__(self, env):
        super().__init__(env)
        self.success = False

    def reset(self, **kwargs):
        if self.success:
            requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
            
            # Move above the target pose
            target = copy.deepcopy(self.config.TARGET_POSE)
            target[2] += 0.03
            self.interpolate_move(target, timeout=0.7)
            self.interpolate_move(self.config.TARGET_POSE, timeout=0.5)

            self._update_currpos()
            reset_pose = copy.deepcopy(self.config.TARGET_POSE)
            reset_pose[1] += 0.04
            self.interpolate_move(reset_pose, timeout=0.5)

        obs, info = self.env.reset(**kwargs)
        self.success = False
        return obs, info
    
    def step(self, action, action_base=None):
        if action_base is None:
            obs, rew, done, truncated, info = self.env.step(action)
        else:
            obs, rew, done, truncated, info = self.env.step(action, action_base)
        self.success = info["succeed"]
        return obs, rew, done, truncated, info
    
    
class StackObsWrapper(gym.Wrapper):
    def __init__(self, env, num_stack=1):
        """
        A wrapper to stack observations over multiple time steps.

        Args:
            env: The environment to wrap.
            num_stack: Number of observations to stack.
        """
        super().__init__(env)
        self.num_stack = num_stack
        
        self.observation_space = self._stack_observation_space(env.observation_space)
        self._frames = {key: None for key in self.observation_space.spaces.keys()}
        
    def _stack_observation_space(self, obs_space):
        """Modify the observation space to support stacked frames."""
        stacked_spaces = {}
        for key, space in obs_space.spaces.items():
            if isinstance(space, Box):
                low = np.repeat(space.low, self.num_stack, axis=0)
                high = np.repeat(space.high, self.num_stack, axis=0)
                stacked_spaces[key] = Box(low=low, high=high, dtype=space.dtype)
            else:
                raise NotImplementedError(f"Stacking not implemented for {type(space)}")
        return Dict(stacked_spaces)
    
    def _get_stacked_obs(self):
        """Constructs the stacked observation."""
        return {key: np.stack(self._frames[key], axis=0) for key in self._frames.keys()}
    
    def reset(self, **kwargs):
        """Resets the environment and initializes the stacked frames."""
        obs, info = self.env.reset(**kwargs)
        self._frames = {key: [obs[key].squeeze(0)] * self.num_stack for key in self._frames.keys()}
        return self._get_stacked_obs(), info
    
    def step(self, action, action_base=None):
        """Steps through the environment and updates the stacked frames."""
        if action_base is None:
            next_obs, reward, done, truncated, info = self.env.step(action)
        else:
            next_obs, reward, done, truncated, info = self.env.step(action, action_base)
        for key in self._frames.keys():
            self._frames[key].pop(0)  # Remove the oldest frame
            self._frames[key].append(next_obs[key].squeeze(0))  # Add the new frame
        return self._get_stacked_obs(), reward, done, truncated, info

class LeapHandWrapper(gym.Wrapper):
    """
    Extends a 6D Franka action space to 22D by appending 16 LeapHand joint commands.

    Action layout:
        action[0:6]  -> Franka arm (position + orientation delta), forwarded to inner env
        action[6:22] -> LeapHand 16 joint positions, sent to leap_node

    Also adds "leap_pose" (16D) to obs["state"] on every step/reset by reading the
    actual hardware position (regardless of send_commands).

    Args:
        env:              Inner Franka env with 6D action space.
        leap_node:        Object with set_leap(list[float]) and read_pos() -> list[float].
        leap_joint_low:   np.ndarray shape (16,) — lower joint limits (physical units, e.g. radians).
        leap_joint_high:  np.ndarray shape (16,) — upper joint limits.
        send_commands:    If True (default, RL deployment), send action[6:22] to the hand on
                          every step. Set to False during demo recording when an external
                          controller (PicoLeapDexController) already drives the hand; in that
                          case the wrapper only reads positions for obs["state"]["leap_pose"].
        leap_reset_pose:  Optional shape (16,) joint targets in radians (LEAP convention).
                          If set and send_commands is True, sent after each inner env reset()
                          (arm motion / joint reset finished) so the hand returns to a fixed
                          pose for the new episode. Ignored when leap_node is None.
        max_delta_per_step: Optional positive float. If set, per-step command change for
                          each Leap joint is clipped to [-max_delta_per_step, +max_delta_per_step]
                          to smooth abrupt policy outputs.
        smooth_init_from_feedback: If True, initialize the smoothing baseline from current
                          hardware feedback (read_pos); else initialize from first target.
    """

    def __init__(
        self,
        env: gym.Env,
        leap_node,
        leap_joint_low: np.ndarray,
        leap_joint_high: np.ndarray,
        send_commands: bool = True,
        leap_reset_pose: Optional[np.ndarray] = None,
        max_delta_per_step: Optional[float] = None,
        smooth_init_from_feedback: bool = True,
    ):
        super().__init__(env)
        self.leap_node = leap_node
        self.send_commands = send_commands
        self.leap_reset_pose = (
            None
            if leap_reset_pose is None
            else np.asarray(leap_reset_pose, dtype=np.float32).reshape(16).copy()
        )
        self.max_delta_per_step = (
            None if max_delta_per_step is None else float(max_delta_per_step)
        )
        if self.max_delta_per_step is not None and self.max_delta_per_step <= 0:
            self.max_delta_per_step = None
        self.smooth_init_from_feedback = bool(smooth_init_from_feedback)
        self._last_sent_leap_cmd: Optional[np.ndarray] = None
        self._leap_joint_low = np.asarray(leap_joint_low, dtype=np.float32).reshape(16)
        self._leap_joint_high = np.asarray(leap_joint_high, dtype=np.float32).reshape(16)

        low = np.concatenate([env.action_space.low, leap_joint_low.astype(np.float32)])
        high = np.concatenate([env.action_space.high, leap_joint_high.astype(np.float32)])
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

        # Extend observation_space: add leap_pose to state dict
        inner_state = env.observation_space["state"]
        new_state_spaces = dict(inner_state.spaces)
        new_state_spaces["leap_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(16,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(new_state_spaces),
                "images": env.observation_space["images"],
            }
        )

    def _read_leap_pos(self) -> np.ndarray:
        if self.leap_node is None:
            return np.zeros(16, dtype=np.float32)
        try:
            return np.array(self.leap_node.read_pos(), dtype=np.float32)
        except Exception:
            return np.zeros(16, dtype=np.float32)

    def step(self, action: np.ndarray, action_base=None):
        arm_action = action[:6]
        leap_action = np.asarray(action[6:22], dtype=np.float32).reshape(16)
        leap_cmd = leap_action.copy()
        if self.leap_node is not None and self.send_commands:
            try:
                if self.max_delta_per_step is not None:
                    if self._last_sent_leap_cmd is None:
                        if self.smooth_init_from_feedback:
                            self._last_sent_leap_cmd = self._read_leap_pos().reshape(16)
                        else:
                            self._last_sent_leap_cmd = leap_cmd.copy()
                    delta = leap_cmd - self._last_sent_leap_cmd
                    leap_cmd = self._last_sent_leap_cmd + np.clip(
                        delta, -self.max_delta_per_step, self.max_delta_per_step
                    )
                leap_cmd = np.clip(
                    leap_cmd.reshape(16),
                    self._leap_joint_low,
                    self._leap_joint_high,
                )
                self.leap_node.set_leap(leap_cmd.tolist())
                self._last_sent_leap_cmd = leap_cmd.copy()
            except Exception:
                pass
        if action_base is None:
            obs, rew, done, truncated, info = self.env.step(arm_action)
        else:
            obs, rew, done, truncated, info = self.env.step(arm_action, action_base)
        obs["state"]["leap_pose"] = self._read_leap_pos()
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_sent_leap_cmd = None
        if (
            self.leap_node is not None
            and self.send_commands
            and self.leap_reset_pose is not None
        ):
            apply_leap_reset_pose(
                self.leap_node,
                self.leap_reset_pose,
                max_delta_per_step=self.max_delta_per_step,
                leap_joint_low=self._leap_joint_low,
                leap_joint_high=self._leap_joint_high,
            )
        obs["state"]["leap_pose"] = self._read_leap_pos()
        return obs, info

class IMLEActionBaseWrapper(gym.Wrapper):
    """
    Injects IMLE (or any) ``action_base`` into ``obs["state"][state_key]``, same idea as
    :class:`LeapHandWrapper` adding ``leap_pose``.

    Use explicit two-argument ``step``::

        env.step(action, action_base)

    - ``action``: **final** control already equal to ``delta + action_base`` in your
      pipeline; forwarded unchanged to the inner env.
    - ``action_base``: copied into ``obs["state"][state_key]`` only; also forwarded to
      inner wrappers that accept ``action_base``.

    With only one argument, ``env.step(action)`` behaves like the inner stack and does
    not overwrite ``state_key`` on that step (see ``reset`` for initial fill).
    """

    def __init__(
        self,
        env: gym.Env,
        state_key: str = "action_base",
        reset_action_base: Optional[np.ndarray] = None,
    ):
        super().__init__(env)
        self._state_key = state_key

        inner_as = env.action_space
        if not isinstance(inner_as, gym.spaces.Box):
            raise NotImplementedError(
                "IMLEActionBaseWrapper expects inner action_space to be gym.spaces.Box"
            )
        if len(inner_as.shape) > 1:
            raise NotImplementedError(
                "IMLEActionBaseWrapper expects a 1D (or scalar) Box action space"
            )
        self._inner_action_dim = int(np.prod(inner_as.shape))
        self.action_space = env.action_space

        inner_state = env.observation_space["state"]
        if not isinstance(inner_state, gym.spaces.Dict):
            raise NotImplementedError(
                "IMLEActionBaseWrapper requires observation_space['state'] to be a Dict"
            )

        new_state_spaces = dict(inner_state.spaces)
        new_state_spaces[self._state_key] = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(self._inner_action_dim,),
            dtype=np.float32,
        )

        images_space = env.observation_space["images"]
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(new_state_spaces),
                "images": images_space,
            }
        )

        if reset_action_base is None:
            self._reset_action_base = np.zeros(self._inner_action_dim, dtype=np.float32)
        else:
            self._reset_action_base = np.asarray(
                reset_action_base, dtype=np.float32
            ).reshape(self._inner_action_dim)

    def step(self, action, action_base=None):
        action = np.asarray(action, dtype=np.float32).reshape(self._inner_action_dim)

        if action_base is None:
            obs, rew, done, truncated, info = self.env.step(action)
        else:
            action_base = np.asarray(action_base, dtype=np.float32).reshape(
                self._inner_action_dim
            )
            obs, rew, done, truncated, info = self.env.step(action, action_base)
            obs["state"][self._state_key] = action_base.copy()
            info["action_base"] = action_base
            info["residual_delta"] = action - action_base

        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["state"][self._state_key] = self._reset_action_base.copy()
        info["action_base"] = self._reset_action_base.copy()
        return obs, info

class AugmentDictObsWrapper(gym.ObservationWrapper):
    """
    Extend a Dict observation with zero-filled keys matching extra_spaces; rollout may
    overwrite them with actual values.
    """

    def __init__(self, env: Env, extra_spaces: dict):
        super().__init__(env)
        self.extra_spaces = dict(extra_spaces)
        d = dict(self.env.observation_space.spaces)
        for k, sp in self.extra_spaces.items():
            if k in d:
                raise ValueError(f"AugmentDictObsWrapper: key {k!r} already exists in observation_space")
            d[k] = sp
        self.observation_space = gym.spaces.Dict(d)

    def observation(self, obs):
        obs = dict(obs)
        for k, sp in self.extra_spaces.items():
            if k not in obs:
                obs[k] = np.zeros(sp.shape, dtype=np.float32)
        return obs

    def step(self, action, action_base=None):
        obs, reward, terminated, truncated, info = self.env.step(action, action_base)
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
