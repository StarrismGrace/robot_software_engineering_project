"""
Booster T1 模块抽象接口定义

五个核心模块的契约层：
  VideoProcessor  → PoseExtractor → MotionCleaner → Retargeting → MuJoCoPlayer

所有成员基于此接口开发，确保模块间输入/输出类型一致、可替换、可独立测试。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from common.motion_data import MotionData


# ============================================================
# VideoProcessor  — 成员 E
# ============================================================
class VideoProcessor(ABC):
    """视频读取与抽帧。

    输入视频文件，按指定策略抽取帧图像，供后续 PoseExtractor 使用。
    """

    @abstractmethod
    def extract_frames(
        self,
        video_path: Union[str, Path],
        fps: Optional[int] = None,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> List[np.ndarray]:
        """从视频中抽取帧图像。

        Parameters
        ----------
        video_path : Union[str, Path]
            输入视频文件路径。
        fps : Optional[int]
            抽帧目标帧率。为 None 时使用视频原始帧率。
        start_sec : float
            起始时间（秒），默认 0。
        end_sec : Optional[float]
            结束时间（秒）。为 None 时到视频结尾。

        Returns
        -------
        List[np.ndarray]
            帧图像列表，每帧为 (H, W, 3) 的 BGR ndarray (uint8)。
        """
        ...

    @abstractmethod
    def get_video_metadata(
        self,
        video_path: Union[str, Path],
    ) -> dict:
        """读取视频元信息。

        Parameters
        ----------
        video_path : Union[str, Path]
            输入视频文件路径。

        Returns
        -------
        dict
            包含 keys: ``fps`` (float), ``total_frames`` (int),
            ``duration_sec`` (float), ``width`` (int), ``height`` (int)。
        """
        ...


# ============================================================
# PoseExtractor  — 成员 B
# ============================================================
class PoseExtractor(ABC):
    """姿态提取。

    从图像序列中检测人体关键点/关节，输出初步 MotionData（positions 填充）。
    """

    @abstractmethod
    def extract(
        self,
        frames: List[np.ndarray],
        fps: int,
        **kwargs,
    ) -> MotionData:
        """从图像序列中提取人体姿态。

        Parameters
        ----------
        frames : List[np.ndarray]
            帧图像列表，每帧为 (H, W, 3) BGR ndarray (uint8)。
        fps : int
            帧率，用于填充 MotionData.fps。
        **kwargs
            提取器特定参数（如置信度阈值、模型路径等）。

        Returns
        -------
        MotionData
            positions 已填充，angles/timestamps 可为 None。
            坐标系与关节顺序见 BOOSTER_T1_JOINT_NAMES。
        """
        ...

    @abstractmethod
    def get_joint_map(self) -> dict:
        """返回提取器输出关节到 Booster T1 标准关节的映射表。

        Returns
        -------
        dict
            {extractor_joint_name: booster_t1_joint_name} 映射。
            用于不同姿态模型的关节名称对齐。
        """
        ...


# ============================================================
# MotionCleaner  — 成员 I
# ============================================================
class MotionCleaner(ABC):
    """数据清洗。

    对原始姿态数据进行去噪、插值、平滑、异常值剔除，输出干净 MotionData。
    """

    @abstractmethod
    def clean(
        self,
        motion: MotionData,
        **kwargs,
    ) -> MotionData:
        """清洗原始动作数据。

        Parameters
        ----------
        motion : MotionData
            原始姿态数据（通常来自 PoseExtractor）。
        **kwargs
            清洗器特定参数（如平滑窗口大小、滤波截止频率等）。

        Returns
        -------
        MotionData
            清洗后的数据。positions 已平滑/插值，angles 可选填充，
            timestamps 已规范化。
        """
        ...

    @abstractmethod
    def detect_outliers(
        self,
        motion: MotionData,
        threshold: float = 3.0,
    ) -> np.ndarray:
        """检测并标记异常帧/关节。

        Parameters
        ----------
        motion : MotionData
            待检测的动作数据。
        threshold : float
            Z-score 阈值，默认 3.0。

        Returns
        -------
        np.ndarray
            布尔掩码，形状 (T, J)，True 表示异常值。
        """
        ...


# ============================================================
# Retargeting  — 成员 D
# ============================================================
class Retargeting(ABC):
    """动作重定向。

    将人体姿态映射为 Booster T1 机器人关节角度。
    输入：人体 MotionData → 输出：机器人 MotionData（angles 填充）。
    """

    @abstractmethod
    def retarget(
        self,
        human_motion: MotionData,
        **kwargs,
    ) -> MotionData:
        """将人体动作重定向到机器人关节空间。

        Parameters
        ----------
        human_motion : MotionData
            人体姿态数据（positions 需已填充）。
        **kwargs
            重定向器特定参数（如缩放因子、关节限位等）。

        Returns
        -------
        MotionData
            机器人关节角度数据。joint_names 为 Booster T1 标准关节名，
            angles 已填充（形状 (T, J) 或 (T, J, 3)），positions 可为 None。
        """
        ...

    @abstractmethod
    def get_joint_limits(self) -> dict:
        """返回机器人各关节的角度限位。

        Returns
        -------
        dict
            {joint_name: (min_rad, max_rad)} 映射，单位为弧度。
        """
        ...


# ============================================================
# MuJoCoPlayer  — 成员 F
# ============================================================
class MuJoCoPlayer(ABC):
    """MuJoCo 仿真播放。

    加载机器人动作数据，驱动 MuJoCo 物理仿真，渲染并输出视频。
    """

    @abstractmethod
    def load_model(
        self,
        model_path: Union[str, Path],
    ) -> None:
        """加载 MuJoCo 机器人模型。

        Parameters
        ----------
        model_path : Union[str, Path]
            MuJoCo XML/MJB 模型文件路径。
        """
        ...

    @abstractmethod
    def play(
        self,
        robot_motion: MotionData,
        output_path: Optional[Union[str, Path]] = None,
        render: bool = True,
        **kwargs,
    ) -> List[np.ndarray]:
        """播放机器人动作并可选输出视频。

        Parameters
        ----------
        robot_motion : MotionData
            机器人关节角度数据（angles 需已填充）。
        output_path : Optional[Union[str, Path]]
            输出视频路径。为 None 时不保存文件。
        render : bool
            是否渲染画面。False 时仅做无头仿真。
        **kwargs
            播放器特定参数（如相机角度、分辨率、播放速度等）。

        Returns
        -------
        List[np.ndarray]
            渲染帧列表，每帧为 (H, W, 3) RGB ndarray (uint8)。
        """
        ...

    @abstractmethod
    def get_physics_state(self) -> dict:
        """获取当前 MuJoCo 物理状态快照。

        Returns
        -------
        dict
            包含 qpos, qvel, time 等 MuJoCo 状态数据。
        """
        ...


# ============================================================
# 模块导出
# ============================================================
__all__ = [
    "VideoProcessor",
    "PoseExtractor",
    "MotionCleaner",
    "Retargeting",
    "MuJoCoPlayer",
]
