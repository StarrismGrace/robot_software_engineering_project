"""
姿态提取模块 - PromptHMR 实现
当前为联调占位 Mock 版本，完全对齐 Booster T1 标准接口
TODO: 待获取预训练权重后，替换为真实深度学习推理
"""
import sys
import os
import numpy as np
from typing import List

# 自动补全项目根路径，解决模块导入问题
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 导入项目公共接口、数据结构与标准关节定义
from common.interfaces import PoseExtractor
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES
from common import setup_logger

logger = setup_logger("pose_extractor")


class PromptHMRExtractor(PoseExtractor):
    """Mock 姿态提取器，用于接口联调；权重到位后可无缝替换为真实推理"""

    def __init__(self, **kwargs):
        super().__init__()
        logger.info("已加载 Mock 姿态提取器（联调占位版本）")

    def extract(
        self,
        frames: List[np.ndarray],
        fps: int,
        **kwargs,
    ) -> MotionData:
        total_frames = len(frames)
        logger.info(f"Mock 姿态提取：模拟处理 {total_frames} 帧，帧率 {fps}")

        # 生成模拟 23 个关节 3D 坐标，与 Booster T1 标准关节数严格对齐
        dummy_joints = np.random.randn(total_frames, BOOSTER_T1_JOINT_NAMES.__len__(), 3) * 0.3

        motion_data = MotionData(
            joint_names=BOOSTER_T1_JOINT_NAMES,
            fps=fps,
            num_frames=total_frames,
            positions=dummy_joints,
        )
        return motion_data

    def get_joint_map(self) -> dict:
        """返回 SMPLX 关节到 Booster T1 标准关节的映射表"""
        joint_map = {
            "pelvis": "root",
            "left_hip": "left_hip",
            "right_hip": "right_hip",
            "left_knee": "left_knee",
            "right_knee": "right_knee",
            "left_ankle": "left_ankle",
            "right_ankle": "right_ankle",
            "spine1": "waist",
            "spine2": "chest",
            "left_shoulder": "left_shoulder",
            "right_shoulder": "right_shoulder",
            "left_elbow": "left_elbow",
            "right_elbow": "right_elbow",
            "left_wrist": "left_wrist",
            "right_wrist": "right_wrist",
            "head": "head",
        }
        return joint_map
