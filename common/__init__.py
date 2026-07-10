"""
Common 模块

提供项目通用的接口定义、数据结构和工具函数。
"""

from common.config_loader import ConfigLoader, get_default_config, load_config
from common.interfaces import (
    VideoProcessor,
    PoseExtractor,
    MotionCleaner,
    Retargeting,
    MuJoCoPlayer,
)
from common.logger import setup_logger
from common.mock_factory import (
    create_mock_video_processor,
    create_mock_pose_extractor,
    create_mock_motion_cleaner,
    create_mock_retargeting,
    create_mock_mujoco_player,
)
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES

__all__ = [
    "ConfigLoader",
    "get_default_config",
    "load_config",
    "VideoProcessor",
    "PoseExtractor",
    "MotionCleaner",
    "Retargeting",
    "MuJoCoPlayer",
    "setup_logger",
    "create_mock_video_processor",
    "create_mock_pose_extractor",
    "create_mock_motion_cleaner",
    "create_mock_retargeting",
    "create_mock_mujoco_player",
    "MotionData",
    "BOOSTER_T1_JOINT_NAMES",
]