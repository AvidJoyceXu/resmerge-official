from typing import OrderedDict
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.utils.rotations import euler_2_quat
import numpy as np
import requests
import copy
import time
from franka_env.envs.franka_env import FrankaEnv

class PickPlaceEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def compute_reward(self, obs):
        """ResMerge uses explicit human labels, never target-pose success."""
        return False

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        if not name_serial_dict:
            return

        # Some keys (e.g. demo) are aliases that should share the
        # same physical capture. Make the base capture explicit so we don't rely
        # on dict iteration order.
        base_key = (
            "side_policy_256"
            if "side_policy_256" in name_serial_dict
            else next(iter(name_serial_dict.keys()))
        )
        self.cap[base_key] = VideoCapture(RSCapture(name=base_key, **name_serial_dict[base_key]))

        for cam_name, kwargs in name_serial_dict.items():
            if cam_name == base_key:
                continue
            if cam_name in ("demo",):
                self.cap[cam_name] = self.cap[base_key]
            else:
                self.cap[cam_name] = VideoCapture(RSCapture(name=cam_name, **kwargs))
                

    def reset(self, **kwargs):
        self._recover()
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        # Move above the target pose
        target = copy.deepcopy(self.currpos)
        target[2] = self.config.TARGET_POSE[2] + 0.05
        # target[2] = self.config.RESET_POSE[2]
        self.interpolate_move(target, timeout=1)
        time.sleep(0.5)

        obs, info = super().reset(**kwargs)
        time.sleep(1)
        self.success = False
        self._update_currpos()
        obs = self._get_obs()
        return obs, info
    
    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        self._send_pos_command(goal)
        time.sleep(timeout)
        self._update_currpos()
    
    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform the reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)
            # LEAP Hand: open pose right after arm joint reset (see TrainConfig.get_environment).
            cb = getattr(self.config, "leap_on_joint_reset", None)
            if callable(cb):
                cb()

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)
        time.sleep(1.0)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
