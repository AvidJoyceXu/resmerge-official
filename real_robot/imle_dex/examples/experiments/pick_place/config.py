# Modified by the ResMerge contributors, 2026-10-31; see real_robot/THIRD_PARTY.md.
import os
import sys
import numpy as np

from franka_env.envs.wrappers import (
    IMLEActionBaseWrapper,
    KeyboardSuccessWrapper,
    Quat2EulerWrapper,
    SpacemouseIntervention,
    LeapHandWrapper,
    apply_leap_reset_pose,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig
from experiments.pick_place.wrapper import PickPlaceEnv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

def _leap_open_pose_rad():
    """Same convention as LeapNode.__init__ / main.py hold_open (allegro zeros → LEAP rad)."""
    leap_py = os.path.join(_REPO_ROOT, "LEAP_Hand_API", "python")
    if leap_py not in sys.path:
        sys.path.insert(0, leap_py)
    import leap_hand_utils.leap_hand_utils as lhu

    return np.asarray(lhu.allegro_to_LEAPhand(np.zeros(16)), dtype=np.float32)

_EXAMPLES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXAMPLES_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLES_ROOT)
from action_scaling import (
    LEAP_HAND_PHYS_MAX,
    LEAP_HAND_PHYS_MIN,
    LeapHandNormalizedActionWrapper,
)

class EnvConfig(DefaultEnvConfig):
    SERVER_URL: str = os.environ.get("FRANKA_SERVER_URL", "http://127.0.0.1:5000/") #"http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "side_policy_256": {
            "serial_number": os.environ.get("REALSENSE_SERIAL", ""),
            "dim": (1280, 720),
            "exposure": 13000,
        },
    }
    IMAGE_CROP = {"wrist_1": lambda img: img,
                  "side_policy_256": lambda img: img[100:, 300:],
                  "demo": lambda img: img}

    TARGET_POSE = np.array([0.57, -0.15, -0.03, np.pi, 0, 0])
    RESET_POSE = np.array([0.55,0.0006,0.10, np.pi, 0, 0])
    ACTION_SCALE = np.array([0.08, 0.2, 1.5])
    
    JOINT_RESET_PERIOD = 10
    RANDOM_RESET = False
    DISPLAY_IMAGE = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.03
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.3, 0.3, 0.3, 0.01, 0.01, 1.57])
    ABS_POSE_LIMIT_LOW  = TARGET_POSE - np.array([0.3, 0.3, 0.1, 0.01, 0.01, 1.57])
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.008,
        "translational_clip_y": 0.008,
        "translational_clip_z": 0.008,
        "translational_clip_neg_x": 0.008,
        "translational_clip_neg_y": 0.008,
        "translational_clip_neg_z": 0.008,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }  # for normal operation other than reset procedure
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0.0,
    }  # only for reset procedure
    MAX_EPISODE_LENGTH = 2500000


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["side_policy_256"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "leap_pose"]
    # Keep RL-related fields for future training (SERL stack), even during data collection.
    checkpoint_period = 1000
    cta_ratio = 2
    random_steps = 0
    discount = 0.95 # 0.98 may also work
    
    residual_alpha: float = 0.05
    buffer_period = 100
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-leap-hand"
    task_desc = "Put the rabbit toy to the pink plate"


    LEAP_JOINT_LOW = LEAP_HAND_PHYS_MIN.astype(np.float32).copy()
    LEAP_JOINT_HIGH = LEAP_HAND_PHYS_MAX.astype(np.float32).copy()
    # Limit per-step joint command change for smoother hand motion (at ~20Hz actor loop).
    LEAP_MAX_DELTA_PER_STEP = 0.05
    def get_environment(
        self,
        fake_env=False,
        save_video=False,
        stack_obs_num=1,
        leap_node=None,
        leap_send_commands=True,
        use_spacemouse: bool = True,
        include_imle_action_base: bool = False,
        use_keyboard_success: bool = False,
    ):
        env_config = EnvConfig()

        leap_reset_pose = None
        try:
            leap_reset_pose = _leap_open_pose_rad()
        except Exception as e:
            print(f"[LEAP] _leap_open_pose_rad failed (no reset-to-open): {e}")
        if (
            leap_node is not None
            and leap_send_commands
            and leap_reset_pose is not None
        ):
            pose = np.asarray(leap_reset_pose, dtype=np.float32).reshape(16)

            def leap_on_joint_reset():
                apply_leap_reset_pose(
                    leap_node,
                    pose,
                    max_delta_per_step=self.LEAP_MAX_DELTA_PER_STEP,
                    leap_joint_low=self.LEAP_JOINT_LOW,
                    leap_joint_high=self.LEAP_JOINT_HIGH,
                )

            env_config.leap_on_joint_reset = leap_on_joint_reset
        else:
            env_config.leap_on_joint_reset = None

        env = PickPlaceEnv(fake_env=fake_env, save_video=save_video, config=env_config)
        # LeapHandWrapper extends action 6D -> 22D and adds leap_pose to obs.
        # send_commands=False during demo recording: hand is driven by external
        # PicoLeapDexController; wrapper only reads positions for obs["state"]["leap_pose"].
        env = LeapHandWrapper(
            env,
            leap_node=leap_node,
            leap_joint_low=self.LEAP_JOINT_LOW,
            leap_joint_high=self.LEAP_JOINT_HIGH,
            send_commands=leap_send_commands,
            leap_reset_pose=leap_reset_pose,
            max_delta_per_step=self.LEAP_MAX_DELTA_PER_STEP,
            smooth_init_from_feedback=True,
        )
        env = RelativeFrame(env)
        if (not fake_env) and use_spacemouse:
            env = SpacemouseIntervention(
                env,
                left_button_success=not use_keyboard_success,
            )
        if (not fake_env) and use_keyboard_success:
            env = KeyboardSuccessWrapper(env)
            print(
                "[KEYBOARD] space: next step sets done=True/succeed=True and reward=True if no reward was present; "
                "f: next step sets done=True/succeed=False without changing reward; "
                "disable with get_environment(..., use_keyboard_success=False)."
            )

        env = Quat2EulerWrapper(env)
        proprio_keys = list(self.proprio_keys)
        if include_imle_action_base:
            
            
            
            
            env = IMLEActionBaseWrapper(env, state_key="action_base")
        env = SERLObsWrapper(env, proprio_keys=proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)
        env = LeapHandNormalizedActionWrapper(env)
        return env
