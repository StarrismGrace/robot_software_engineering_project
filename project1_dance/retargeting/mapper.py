"""
动作重定向模块 — 成员D

将人体关键点位置（MotionData.positions）转换为机器人关节角度（MotionData.angles）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from common.interfaces import Retargeting
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES


class RetargetingImpl(Retargeting):
    """动作重定向实现类

    根据人体关节位置计算机器人关节角度。

    Parameters
    ----------
    config_path : Optional[Union[str, Path]]
        mapping.yaml 配置文件路径，默认为 config/mapping.yaml
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "mapping.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.parent_map: Dict[str, str] = config.get("parent_map", {})
        self.scale_factors: Dict[str, float] = config.get("scale_factors", {"default": 0.8})
        self.joint_limits: Dict[str, Tuple[float, float]] = config.get("joint_limits", {})

        # 机器人标准关节列表
        self.robot_joint_names = BOOSTER_T1_JOINT_NAMES

    def retarget(self, human_motion: MotionData, **kwargs) -> MotionData:
        """将人体动作映射为机器人关节角度

        Parameters
        ----------
        human_motion : MotionData
            人体姿态数据，positions 已填充，形状 (T, J, 3)
        **kwargs
            scale : float, 可选，全局缩放因子

        Returns
        -------
        MotionData
            机器人关节角度数据，angles 已填充，形状 (T, J)
        """
        if human_motion.positions is None:
            raise ValueError("human_motion.positions 不能为 None")

        positions = human_motion.positions  # (T, J, 3)
        T = positions.shape[0]
        scale = kwargs.get("scale", self.scale_factors.get("default", 0.8))

        # 构建关节名称 → 索引映射
        joint_name_to_idx = {
            name: i for i, name in enumerate(human_motion.joint_names)
        }

        # 逐帧计算角度
        robot_angles = np.zeros((T, len(self.robot_joint_names)), dtype=np.float32)

        for t in range(T):
            frame_pos = positions[t]
            angles = self._compute_frame_angles(frame_pos, joint_name_to_idx, scale)
            robot_angles[t] = angles

        # 应用关节限位
        for i, joint_name in enumerate(self.robot_joint_names):
            if joint_name in self.joint_limits:
                min_val, max_val = self.joint_limits[joint_name]
                robot_angles[:, i] = np.clip(robot_angles[:, i], min_val, max_val)

        return MotionData(
            joint_names=self.robot_joint_names.copy(),
            fps=human_motion.fps,
            num_frames=T,
            positions=None,
            angles=robot_angles,
            timestamps=human_motion.timestamps,
        )

    def _compute_frame_angles(
        self,
        pos: np.ndarray,
        idx_map: Dict[str, int],
        scale: float,
    ) -> np.ndarray:
        """计算单帧的机器人关节角度"""
        angles = np.zeros(len(self.robot_joint_names), dtype=np.float32)

        def get_vec(joint_a: str, joint_b: str) -> np.ndarray:
            """计算从 joint_a 到 joint_b 的向量"""
            if joint_a not in idx_map or joint_b not in idx_map:
                return np.zeros(3)
            return pos[idx_map[joint_b]] - pos[idx_map[joint_a]]

        def get_angle(joint_parent: str, joint_child: str, joint_grandchild: str) -> float:
            """计算 parent->child 和 child->grandchild 之间的夹角（弧度）"""
            v1 = get_vec(joint_parent, joint_child)
            v2 = get_vec(joint_child, joint_grandchild)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 1e-6 or norm2 < 1e-6:
                return 0.0
            cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
            return np.arccos(cos_angle)

        # ---- 肘关节 ----
        # 左肘：left_shoulder -> left_elbow -> left_wrist
        angles[6] = get_angle("left_shoulder", "left_elbow", "left_wrist") * scale * self.scale_factors.get("arm", 0.7)
        # 右肘
        angles[9] = get_angle("right_shoulder", "right_elbow", "right_wrist") * scale * self.scale_factors.get("arm", 0.7)

        # ---- 膝关节 ----
        # 左膝：left_hip -> left_knee -> left_ankle
        angles[12] = get_angle("left_hip", "left_knee", "left_ankle") * scale * self.scale_factors.get("leg", 0.8)
        # 右膝
        angles[15] = get_angle("right_hip", "right_knee", "right_ankle") * scale * self.scale_factors.get("leg", 0.8)

        # ---- 躯干倾斜（简化：用脊柱方向） ----
        # 用 shoulder_center - hip_center 的倾斜角度表示 chest
        if "left_shoulder" in idx_map and "right_shoulder" in idx_map and "left_hip" in idx_map:
            shoulder_center = (pos[idx_map["left_shoulder"]] + pos[idx_map["right_shoulder"]]) / 2
            hip_center = (pos[idx_map["left_hip"]] + pos[idx_map["right_hip"]]) / 2
            spine = shoulder_center - hip_center
            if np.linalg.norm(spine) > 1e-6:
                # 与垂直方向（0,0,1）的夹角
                vert = np.array([0, 0, 1])
                cos_angle = np.clip(np.dot(spine, vert) / (np.linalg.norm(spine) * np.linalg.norm(vert)), -1.0, 1.0)
                angles[2] = np.arccos(cos_angle) * scale * self.scale_factors.get("spine", 0.5)

        # ---- 其他关节暂时设为0（可扩展） ----
        # 可在此添加肩关节、髋关节等更细致的计算

        return angles

    def get_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        """返回各关节的角度限位（弧度）"""
        return self.joint_limits.copy()