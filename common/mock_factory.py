"""
Mock 工厂模块

为各模块提供 Mock 实现，用于流水线测试。
"""

import numpy as np
from typing import List, Optional, Union
from pathlib import Path

from common.interfaces import (
    VideoProcessor,
    PoseExtractor,
    MotionCleaner,
    Retargeting,
    MuJoCoPlayer,
)
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES


# ============================================================
# Mock VideoProcessor
# ============================================================
class MockVideoProcessor(VideoProcessor):
    """Mock 视频处理器：生成模拟帧图像"""

    def extract_frames(
        self,
        video_path: Union[str, Path],
        fps: Optional[int] = None,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> List[np.ndarray]:
        """生成模拟帧图像"""
        num_frames = 300
        frames = []
        for i in range(num_frames):
            # 生成带简单运动的模拟图像
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # 画一个移动的圆
            x = int(320 + 100 * np.sin(i / 30.0))
            y = int(240 + 80 * np.cos(i / 20.0))
            cv2 = __import__("cv2")
            cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)
            frames.append(frame)
        return frames

    def get_video_metadata(self, video_path: Union[str, Path]) -> dict:
        return {
            "fps": 30.0,
            "total_frames": 300,
            "duration_sec": 10.0,
            "width": 640,
            "height": 480,
        }


# ============================================================
# Mock PoseExtractor
# ============================================================
class MockPoseExtractor(PoseExtractor):
    """Mock 姿态提取器：生成模拟人体姿态数据"""

    def extract(self, frames: List[np.ndarray], fps: int, **kwargs) -> MotionData:
        """生成模拟人体姿态数据"""
        T = len(frames)
        J = len(BOOSTER_T1_JOINT_NAMES)
        
        # 生成模拟位置数据：带一些噪声和运动
        positions = np.zeros((T, J, 3))
        for t in range(T):
            for j in range(J):
                # 每个关节有不同幅度的运动
                amp = 0.5 + 0.3 * np.sin(j)
                positions[t, j, 0] = 0.5 * j + 0.2 * np.sin(t / 20.0 + j)
                positions[t, j, 1] = 0.3 * (J - j) + 0.2 * np.cos(t / 15.0 + j * 0.5)
                positions[t, j, 2] = 0.1 * np.sin(t / 10.0 + j * 0.3)
        
        # 添加一些噪声
        positions += 0.02 * np.random.randn(T, J, 3)
        
        # 添加少量 NaN 来测试清洗器
        nan_indices = np.random.choice(T * J, size=int(T * J * 0.02), replace=False)
        for idx in nan_indices:
            t = idx // J
            j = idx % J
            positions[t, j, :] = np.nan
        
        timestamps = np.arange(T) / fps
        
        return MotionData(
            joint_names=BOOSTER_T1_JOINT_NAMES.copy(),
            fps=fps,
            num_frames=T,
            positions=positions,
            angles=None,
            timestamps=timestamps,
        )

    def get_joint_map(self) -> dict:
        return {name: name for name in BOOSTER_T1_JOINT_NAMES}


# ============================================================
# Mock MotionCleaner
# ============================================================
class MockMotionCleaner(MotionCleaner):
    """Mock 数据清洗器：透传数据"""

    def clean(self, motion: MotionData, **kwargs) -> MotionData:
        return motion

    def detect_outliers(self, motion: MotionData, threshold: float = 3.0) -> np.ndarray:
        return np.zeros((motion.num_frames, motion.num_joints), dtype=bool)


# ============================================================
# Mock Retargeting
# ============================================================
class MockRetargeting(Retargeting):
    """Mock 重定向器：生成模拟机器人关节角度"""

    def retarget(self, human_motion: MotionData, **kwargs) -> MotionData:
        """将人体姿态映射为机器人关节角度"""
        T = human_motion.num_frames
        J = len(BOOSTER_T1_JOINT_NAMES)
        
        # 生成模拟关节角度
        angles = np.zeros((T, J))
        for t in range(T):
            for j in range(J):
                angles[t, j] = 0.5 * np.sin(t / 30.0 + j * 0.3) + 0.1 * np.random.randn()
        
        timestamps = human_motion.timestamps.copy() if human_motion.timestamps is not None else None
        
        return MotionData(
            joint_names=BOOSTER_T1_JOINT_NAMES.copy(),
            fps=human_motion.fps,
            num_frames=T,
            positions=None,
            angles=angles,
            timestamps=timestamps,
        )

    def get_joint_limits(self) -> dict:
        limits = {}
        for name in BOOSTER_T1_JOINT_NAMES:
            limits[name] = (-1.0, 1.0)
        return limits


# ============================================================
# Mock MuJoCoPlayer
# ============================================================
class MockMuJoCoPlayer(MuJoCoPlayer):
    """Mock MuJoCo 播放器：模拟渲染"""

    def __init__(self):
        self._model_loaded = False

    def load_model(self, model_path: Union[str, Path]) -> None:
        self._model_loaded = True

    def play(
        self,
        robot_motion: MotionData,
        output_path: Optional[Union[str, Path]] = None,
        render: bool = True,
        **kwargs,
    ) -> List[np.ndarray]:
        """生成模拟渲染帧"""
        num_frames = min(robot_motion.num_frames, 10)  # 只渲染10帧
        frames = []
        for i in range(num_frames):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = 100 + 50 * np.sin(i / 5.0)  # 红色通道变化
            frames.append(frame)
        return frames

    def get_physics_state(self) -> dict:
        return {"qpos": np.zeros(23), "qvel": np.zeros(23), "time": 0.0}


# ============================================================
# 工厂函数
# ============================================================
def create_mock_video_processor() -> VideoProcessor:
    """创建 Mock 视频处理器"""
    return MockVideoProcessor()


def create_mock_pose_extractor() -> PoseExtractor:
    """创建 Mock 姿态提取器"""
    return MockPoseExtractor()


def create_mock_motion_cleaner() -> MotionCleaner:
    """创建 Mock 数据清洗器"""
    return MockMotionCleaner()


def create_mock_retargeting() -> Retargeting:
    """创建 Mock 重定向器"""
    return MockRetargeting()


def create_mock_mujoco_player() -> MuJoCoPlayer:
    """创建 Mock MuJoCo 播放器"""
    return MockMuJoCoPlayer()