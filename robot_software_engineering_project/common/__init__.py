"""Booster T1 公共模块 —— 统一数据格式、工具函数。"""

from common.config_loader import get_default_config, load_config
from common.interfaces import (
    MotionCleaner,
    MuJoCoPlayer,
    PoseExtractor,
    Retargeting,
    VideoProcessor,
)
from common.logger import get_logger, setup_logger
from common.mock_factory import (
    create_mock_clean_data,
    create_mock_frames,
    create_mock_pose_data,
    create_mock_robot_motion,
)
from common.motion_data import (
    BOOSTER_T1_JOINT_NAMES,
    BOOSTER_T1_NUM_JOINTS,
    MotionData,
)

__all__ = [
    "MotionData",
    "BOOSTER_T1_JOINT_NAMES",
    "BOOSTER_T1_NUM_JOINTS",
    "VideoProcessor",
    "PoseExtractor",
    "MotionCleaner",
    "Retargeting",
    "MuJoCoPlayer",
    "create_mock_frames",
    "create_mock_pose_data",
    "create_mock_clean_data",
    "create_mock_robot_motion",
    "setup_logger",
    "get_logger",
    "load_config",
    "get_default_config",
]
