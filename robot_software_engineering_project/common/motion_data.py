"""
Booster T1 统一动作数据格式

所有模块之间传递动作数据必须使用此 MotionData 类，
保证接口一致、可校验、可序列化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ============================================================
# Booster T1 标准关节定义（23个）
# ============================================================
BOOSTER_T1_JOINT_NAMES: List[str] = [
    "root",
    "chest",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_toe",
    "right_toe",
    "left_upper_arm",
    "right_upper_arm",
    "left_thigh",
    "right_thigh",
]

BOOSTER_T1_NUM_JOINTS: int = len(BOOSTER_T1_JOINT_NAMES)  # 22（实际列表中23指保留扩展位）


@dataclass
class MotionData:
    """Booster T1 统一动作数据容器。

    Parameters
    ----------
    joint_names : List[str]
        关节名称列表，顺序与 positions/angles 的第二维对齐。
    fps : int
        动作采样帧率（整数）。
    num_frames : int
        总帧数 T。
    positions : Optional[np.ndarray]
        关节位置，形状 (T, J, 3)，单位：米。可为 None。
    angles : Optional[np.ndarray]
        关节角度，形状 (T, J) 或 (T, J, 3)，单位：弧度。可为 None。
    timestamps : Optional[np.ndarray]
        每帧时间戳，形状 (T,)，单位：秒。可为 None。
    """

    joint_names: List[str]
    fps: int
    num_frames: int
    positions: Optional[np.ndarray] = None
    angles: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None

    # ---- 内部缓存（不参与比较/序列化） ----
    _validated: bool = field(default=False, repr=False, compare=False)

    # ============================================================
    # __post_init__：构造后立即执行的基础校验
    # ============================================================
    def __post_init__(self) -> None:
        """构造后自动校验：关节数量、维度匹配。"""
        errors: List[str] = []

        # 1. fps 必须为正整数
        if not isinstance(self.fps, int) or self.fps <= 0:
            errors.append(f"fps 必须为正整数，实际: {self.fps}")

        # 2. num_frames 必须为正整数
        if not isinstance(self.num_frames, int) or self.num_frames <= 0:
            errors.append(f"num_frames 必须为正整数，实际: {self.num_frames}")

        # 3. 关节名称不能为空
        if not self.joint_names or len(self.joint_names) == 0:
            errors.append("joint_names 不能为空")

        J = len(self.joint_names) if self.joint_names else 0

        # 4. positions 形状校验: (T, J, 3)
        if self.positions is not None:
            if not isinstance(self.positions, np.ndarray):
                errors.append("positions 必须是 np.ndarray")
            else:
                if self.positions.ndim != 3:
                    errors.append(
                        f"positions 必须为3维 (T, J, 3)，实际维度: {self.positions.ndim}"
                    )
                else:
                    T_pos, J_pos, D_pos = self.positions.shape
                    if T_pos != self.num_frames:
                        errors.append(
                            f"positions 第0维 ({T_pos}) 与 num_frames ({self.num_frames}) 不一致"
                        )
                    if J_pos != J:
                        errors.append(
                            f"positions 第1维 ({J_pos}) 与 joint_names 数量 ({J}) 不一致"
                        )
                    if D_pos != 3:
                        errors.append(
                            f"positions 第2维 ({D_pos}) 必须为 3 (x, y, z)"
                        )

        # 5. angles 形状校验: (T, J) 或 (T, J, 3)
        if self.angles is not None:
            if not isinstance(self.angles, np.ndarray):
                errors.append("angles 必须是 np.ndarray")
            else:
                if self.angles.ndim not in (2, 3):
                    errors.append(
                        f"angles 必须为2维 (T, J) 或3维 (T, J, 3)，实际维度: {self.angles.ndim}"
                    )
                else:
                    T_ang, J_ang = self.angles.shape[0], self.angles.shape[1]
                    if T_ang != self.num_frames:
                        errors.append(
                            f"angles 第0维 ({T_ang}) 与 num_frames ({self.num_frames}) 不一致"
                        )
                    if J_ang != J:
                        errors.append(
                            f"angles 第1维 ({J_ang}) 与 joint_names 数量 ({J}) 不一致"
                        )
                    if self.angles.ndim == 3 and self.angles.shape[2] != 3:
                        errors.append(
                            f"angles 为3维时第2维必须为3 (欧拉角)，实际: {self.angles.shape[2]}"
                        )

        # 6. timestamps 形状校验: (T,)
        if self.timestamps is not None:
            if not isinstance(self.timestamps, np.ndarray):
                errors.append("timestamps 必须是 np.ndarray")
            elif self.timestamps.ndim != 1 or self.timestamps.shape[0] != self.num_frames:
                errors.append(
                    f"timestamps 形状必须为 ({self.num_frames},)，实际: {self.timestamps.shape}"
                )

        if errors:
            raise ValueError(
                "MotionData 初始化校验失败:\n  " + "\n  ".join(errors)
            )

    # ============================================================
    # validate()：深度数据质量校验
    # ============================================================
    def validate(self) -> List[str]:
        """深度数据质量校验（NaN、关节名称合法性等）。

        Returns
        -------
        List[str]
            校验失败项列表。空列表表示全部通过。
        """
        issues: List[str] = []

        # ---- 检查关节名称是否在标准列表中 ----
        unknown_joints = [
            name for name in self.joint_names if name not in BOOSTER_T1_JOINT_NAMES
        ]
        if unknown_joints:
            issues.append(f"未知关节名称: {unknown_joints}")

        # ---- 检查关节名称是否有重复 ----
        if len(self.joint_names) != len(set(self.joint_names)):
            seen: dict = {}
            dupes = []
            for name in self.joint_names:
                seen[name] = seen.get(name, 0) + 1
            dupes = [name for name, count in seen.items() if count > 1]
            issues.append(f"重复关节名称: {dupes}")

        # ---- 检查 NaN / Inf ----
        if self.positions is not None:
            if not np.isfinite(self.positions).all():
                nan_count = int(np.isnan(self.positions).sum())
                inf_count = int(np.isinf(self.positions).sum())
                issues.append(
                    f"positions 包含 NaN: {nan_count} 个, Inf: {inf_count} 个"
                )

        if self.angles is not None:
            if not np.isfinite(self.angles).all():
                nan_count = int(np.isnan(self.angles).sum())
                inf_count = int(np.isinf(self.angles).sum())
                issues.append(
                    f"angles 包含 NaN: {nan_count} 个, Inf: {inf_count} 个"
                )

        if self.timestamps is not None:
            if not np.isfinite(self.timestamps).all():
                nan_count = int(np.isnan(self.timestamps).sum())
                inf_count = int(np.isinf(self.timestamps).sum())
                issues.append(
                    f"timestamps 包含 NaN: {nan_count} 个, Inf: {inf_count} 个"
                )

        # ---- 检查 timestamps 是否严格递增 ----
        if self.timestamps is not None and np.isfinite(self.timestamps).all():
            diffs = np.diff(self.timestamps)
            if np.any(diffs < 0):
                issues.append("timestamps 不是严格递增的")

        self._validated = (len(issues) == 0)
        return issues

    # ============================================================
    # to_dict()：日志/序列化
    # ============================================================
    def to_dict(self) -> dict:
        """转为可 JSON 序列化的字典，用于日志和调试。

        数组字段转为 shape + dtype 摘要，避免日志膨胀。
        """
        result: dict = {
            "joint_names": self.joint_names,
            "num_joints": len(self.joint_names),
            "fps": self.fps,
            "num_frames": self.num_frames,
            "duration_sec": (
                float(self.timestamps[-1] - self.timestamps[0])
                if self.timestamps is not None and len(self.timestamps) > 1
                else None
            ),
        }

        if self.positions is not None:
            result["positions_shape"] = self.positions.shape
            result["positions_dtype"] = str(self.positions.dtype)
            result["positions_range"] = (
                round(float(self.positions.min()), 4),
                round(float(self.positions.max()), 4),
            )
        else:
            result["positions_shape"] = None

        if self.angles is not None:
            result["angles_shape"] = self.angles.shape
            result["angles_dtype"] = str(self.angles.dtype)
            result["angles_range"] = (
                round(float(self.angles.min()), 4),
                round(float(self.angles.max()), 4),
            )
        else:
            result["angles_shape"] = None

        if self.timestamps is not None:
            result["timestamps_shape"] = (len(self.timestamps),)
            result["timestamps_dtype"] = str(self.timestamps.dtype)
        else:
            result["timestamps_shape"] = None

        return result

    # ============================================================
    # 便捷属性
    # ============================================================
    @property
    def num_joints(self) -> int:
        """关节数量。"""
        return len(self.joint_names)

    @property
    def has_positions(self) -> bool:
        """是否包含位置数据。"""
        return self.positions is not None

    @property
    def has_angles(self) -> bool:
        """是否包含角度数据。"""
        return self.angles is not None

    @property
    def duration(self) -> Optional[float]:
        """动作总时长（秒）。"""
        if self.timestamps is not None and len(self.timestamps) > 1:
            return float(self.timestamps[-1] - self.timestamps[0])
        if self.fps > 0:
            return self.num_frames / self.fps
        return None

    # ============================================================
    # 字符串表示
    # ============================================================
    def __repr__(self) -> str:
        pos_shape = self.positions.shape if self.positions is not None else None
        ang_shape = self.angles.shape if self.angles is not None else None
        return (
            f"MotionData(joints={self.num_joints}, fps={self.fps}, "
            f"frames={self.num_frames}, pos_shape={pos_shape}, "
            f"ang_shape={ang_shape}, validated={self._validated})"
        )
