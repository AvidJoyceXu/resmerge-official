import enum
from pathlib import Path
from typing import Optional

import numpy as np

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


class RobotName(enum.Enum):
    allegro = enum.auto()
    shadow = enum.auto()
    svh = enum.auto()
    leap = enum.auto()
    ability = enum.auto()
    inspire = enum.auto()
    panda = enum.auto()


class RetargetingType(enum.Enum):
    vector = enum.auto()  # For teleoperation, no finger closing prior
    position = (
        enum.auto()
    )  # For offline data processing, especially hand-object interaction data
    dexpilot = enum.auto()  # For teleoperation, with finger closing prior


class HandType(enum.Enum):
    right = enum.auto()
    left = enum.auto()


ROBOT_NAME_MAP = {
    RobotName.allegro: "allegro_hand",
    RobotName.shadow: "shadow_hand",
    RobotName.svh: "schunk_svh_hand",
    RobotName.leap: "leap_hand",
    RobotName.ability: "ability_hand",
    RobotName.inspire: "inspire_hand",
    RobotName.panda: "panda_gripper",
}

ROBOT_NAMES = list(ROBOT_NAME_MAP.keys())


def get_default_config_path(
    robot_name: RobotName, retargeting_type: RetargetingType, hand_type: HandType
) -> Optional[Path]:
    if (robot_name is not RobotName.leap
            or hand_type is not HandType.right
            or retargeting_type is not RetargetingType.vector):
        raise ValueError("This minimal example supports only LEAP right-hand vector retargeting.")
    return Path(__file__).parent / "configs" / "teleop" / "leap_hand_right.yml"


OPERATOR2MANO = {
    HandType.right: OPERATOR2MANO_RIGHT,
    HandType.left: OPERATOR2MANO_LEFT,
}
