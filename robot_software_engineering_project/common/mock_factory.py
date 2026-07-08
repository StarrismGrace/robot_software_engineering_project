"""
Booster T1 Mock 数据工厂

在各模块未完成时，提供占位数据用于测试和并行开发。
所有生成的 MotionData 均通过 __post_init__ 校验，可直接传给下游模块。
"""

from __future__ import annotations

from typing import List

import numpy as np

from common.motion_data import BOOSTER_T1_JOINT_NAMES, MotionData

# ============================================================
# 内部常量
# ============================================================
_J = len(BOOSTER_T1_JOINT_NAMES)  # 22
_FRAME_HEIGHT = 480
_FRAME_WIDTH = 640
_DEFAULT_FPS = 30
_RNG = np.random.default_rng(42)  # 固定种子保证可复现


def _smooth_dance_signal(
    T: int,
    J: int,
    n_components: int = 3,
    noise_std: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """生成带正弦基的平滑运动信号 (T, J)。

    每个关节由 n_components 个不同频率/相位/振幅的正弦波叠加，
    模拟舞蹈动作的周期性特征，附加小量高斯噪声。
    """
    if rng is None:
        rng = _RNG
    t = np.arange(T, dtype=np.float64)[:, None]  # (T, 1)

    signal = np.zeros((T, J), dtype=np.float64)
    for k in range(n_components):
        freq = 0.3 + 1.2 * rng.random(J)  # 每关节不同频率
        phase = 2.0 * np.pi * rng.random(J)  # 随机初相
        amp = 0.3 + 0.8 * rng.random(J)  # 振幅 0.3~1.1
        signal += amp[None, :] * np.sin(
            2.0 * np.pi * freq[None, :] * t / T + phase[None, :]
        )

    # 归一化到 [-1, 1] 并叠噪
    s_min, s_max = signal.min(axis=0, keepdims=True), signal.max(axis=0, keepdims=True)
    s_range = np.where(s_max - s_min < 1e-8, 1.0, s_max - s_min)
    signal = 2.0 * (signal - s_min) / s_range - 1.0
    signal += rng.normal(0, noise_std, (T, J))
    return signal


# ============================================================
# 1. create_mock_frames
# ============================================================
def create_mock_frames(
    num_frames: int = 100,
    height: int = _FRAME_HEIGHT,
    width: int = _FRAME_WIDTH,
    seed: int | None = None,
) -> List[np.ndarray]:
    """生成假的视频帧列表。

    Parameters
    ----------
    num_frames : int
        帧数 T，默认 100。
    height : int
        帧高度，默认 480。
    width : int
        帧宽度，默认 640。
    seed : int | None
        随机种子。为 None 时使用全局 RNG。

    Returns
    -------
    List[np.ndarray]
        帧列表，每帧 (H, W, 3) uint8 RGB ndarray。
    """
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    frames = [
        rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        for _ in range(num_frames)
    ]
    return frames


# ============================================================
# 2. create_mock_pose_data
# ============================================================
def create_mock_pose_data(
    num_frames: int = 100,
    fps: int = _DEFAULT_FPS,
    seed: int | None = None,
) -> MotionData:
    """生成假的人体姿态数据（positions 填充，模拟舞蹈运动）。

    各关节沿三维正弦轨迹运动，含合理噪声。

    Parameters
    ----------
    num_frames : int
        帧数 T，默认 100。
    fps : int
        帧率，默认 30。
    seed : int | None
        随机种子。

    Returns
    -------
    MotionData
        positions 已填充，angles 为 None。
    """
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    T, J = num_frames, _J

    # 每个坐标轴独立生成平滑信号
    pos_x = _smooth_dance_signal(T, J, n_components=3, noise_std=0.015, rng=rng)
    pos_y = _smooth_dance_signal(T, J, n_components=3, noise_std=0.015, rng=rng)
    pos_z = _smooth_dance_signal(T, J, n_components=3, noise_std=0.015, rng=rng)

    # 缩放：Y轴范围较小（上下晃动），X/Z 范围大些（水平/前后移动）
    scale = np.array([[[0.8, 0.4, 0.8]]], dtype=np.float64)  # (1, 1, 3)
    positions = np.stack([pos_x, pos_y, pos_z], axis=-1) * scale  # (T, J, 3)

    # root 关节偏移到 (0, 0.9, 0) 附近（站立高度约 0.9m）
    positions[:, 0, :] += np.array([0.0, 0.9, 0.0], dtype=np.float64)

    # 下半身关节（hip/knee/ankle/toe/thigh）Y 偏移下调
    lower_body = [
        "left_hip", "left_knee", "left_ankle", "left_toe", "left_thigh",
        "right_hip", "right_knee", "right_ankle", "right_toe", "right_thigh",
    ]
    for j, name in enumerate(BOOSTER_T1_JOINT_NAMES):
        if name in lower_body:
            positions[:, j, 1] -= 0.4  # 下半身整体下移

    # 手臂关节 X 偏移（左右张开）
    if "left_shoulder" in BOOSTER_T1_JOINT_NAMES:
        li = BOOSTER_T1_JOINT_NAMES.index("left_shoulder")
        positions[:, li, 0] += 0.25
    if "right_shoulder" in BOOSTER_T1_JOINT_NAMES:
        ri = BOOSTER_T1_JOINT_NAMES.index("right_shoulder")
        positions[:, ri, 0] -= 0.25

    timestamps = np.arange(T, dtype=np.float64) / fps

    return MotionData(
        joint_names=BOOSTER_T1_JOINT_NAMES.copy(),
        fps=fps,
        num_frames=T,
        positions=positions.astype(np.float32),
        angles=None,
        timestamps=timestamps,
    )


# ============================================================
# 3. create_mock_clean_data
# ============================================================
def create_mock_clean_data(
    num_frames: int = 100,
    fps: int = _DEFAULT_FPS,
    seed: int | None = None,
) -> MotionData:
    """生成假的重定向后关节角度数据（angles 填充，模拟舞蹈动作）。

    角度值裁剪在 [-π, π] 范围内，不同关节有不同运动幅度。

    Parameters
    ----------
    num_frames : int
        帧数 T，默认 100。
    fps : int
        帧率，默认 30。
    seed : int | None
        随机种子。

    Returns
    -------
    MotionData
        angles 已填充，positions 为 None。
    """
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    T, J = num_frames, _J

    # 生成平滑角度信号
    angles = _smooth_dance_signal(T, J, n_components=4, noise_std=0.03, rng=rng)

    # 各关节运动幅度限制（弧度），避免全范围 -π~π 过于夸张
    joint_limits_map = _get_joint_angle_limits()
    for j, name in enumerate(BOOSTER_T1_JOINT_NAMES):
        lo, hi = joint_limits_map.get(name, (-np.pi, np.pi))
        mid = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
        # 映射信号从 [-1,1] 到 [lo, hi]
        angles[:, j] = mid + angles[:, j] * half_range

    angles = np.clip(angles, -np.pi, np.pi)
    timestamps = np.arange(T, dtype=np.float64) / fps

    return MotionData(
        joint_names=BOOSTER_T1_JOINT_NAMES.copy(),
        fps=fps,
        num_frames=T,
        positions=None,
        angles=angles.astype(np.float32),
        timestamps=timestamps,
    )


# ============================================================
# 4. create_mock_robot_motion
# ============================================================
def create_mock_robot_motion(
    num_frames: int = 100,
    fps: int = _DEFAULT_FPS,
    seed: int | None = None,
) -> MotionData:
    """生成假的机器人关节轨迹（复用 create_mock_clean_data）。

    Parameters
    ----------
    num_frames : int
        帧数 T，默认 100。
    fps : int
        帧率，默认 30。
    seed : int | None
        随机种子。

    Returns
    -------
    MotionData
        angles 已填充，可直接传给 MuJoCoPlayer。
    """
    return create_mock_clean_data(num_frames=num_frames, fps=fps, seed=seed)


# ============================================================
# 内部辅助：关节限位表
# ============================================================
def _get_joint_angle_limits() -> dict:
    """返回各关节的合理角度限位（弧度），用于约束 mock 数据。

    Returns
    -------
    dict
        {joint_name: (min_rad, max_rad)}
    """
    return {
        "root":          (-0.3, 0.3),     # 躯干根：小范围晃动
        "chest":         (-0.4, 0.4),
        "neck":          (-0.5, 0.5),
        "head":          (-0.6, 0.6),
        "left_shoulder":  (-2.0, 1.5),     # 肩关节大范围活动
        "left_elbow":     (0.0, 2.5),
        "left_wrist":     (-1.5, 1.5),
        "right_shoulder": (-1.5, 2.0),
        "right_elbow":    (0.0, 2.5),
        "right_wrist":    (-1.5, 1.5),
        "left_hip":       (-1.0, 1.0),
        "left_knee":      (0.0, 2.0),
        "left_ankle":     (-0.8, 0.8),
        "right_hip":      (-1.0, 1.0),
        "right_knee":     (0.0, 2.0),
        "right_ankle":    (-0.8, 0.8),
        "left_toe":       (-0.5, 0.5),
        "right_toe":      (-0.5, 0.5),
        "left_upper_arm": (-2.0, 1.5),
        "right_upper_arm":(-1.5, 2.0),
        "left_thigh":     (-1.0, 1.0),
        "right_thigh":    (-1.0, 1.0),
    }


# ============================================================
# 模块导出
# ============================================================
__all__ = [
    "create_mock_frames",
    "create_mock_pose_data",
    "create_mock_clean_data",
    "create_mock_robot_motion",
]
